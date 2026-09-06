# Cosmos Policy in Real-World ALOHA Robot Tasks

(Instructions here are adapted from [OpenVLA-OFT](https://github.com/moojink/openvla-oft).)

## Relevant Files

Evaluation
* `cosmos_policy/experiments/robot/aloha/`: ALOHA training and eval files
  * `deploy.py`: Cosmos Policy server deploy script (SERVER SIDE)
  * `run_aloha_eval.py`: ALOHA eval script (CLIENT SIDE)
  * `aloha_utils.py`: ALOHA eval utils
  * Other ALOHA robot environment files copied from the original [ALOHA GitHub repo](https://github.com/tonyzhaozh/aloha):
    * `constants.py`
    * `real_env.py`
    * `robot_utils.py`
* `cosmos_policy/experiments/robot/`: General eval utils files
  * `cosmos_utils.py`: Cosmos Policy-specific eval utils
  * `robot_utils.py`: Other eval utils

Note: Unlike the LIBERO and RoboCasa evaluation setups, we use a server-client interface here. This is particularly useful if the user's machine which commands the robot does not have access to a local GPU with sufficient specs to run Cosmos Policy.

Training
* `cosmos_policy/experiments/robot/aloha/`: ALOHA training and eval files
  * `preprocess_split_aloha_data.py`: ALOHA data preprocessing script
  * `preprocess_split_aloha_data.py`: ALOHA data preprocessing script
* `cosmos_policy/scripts/train.py`: Cosmos Policy training script (NOTE: Do NOT use other train.py scripts that may exist elsewhere in the codebase for base video models)
* `cosmos_policy/datasets/save_aloha_t5_text_embeddings.py`: Script for precomputing T5 embeddings of task labels

## Setup

Follow the instructions here: [SETUP.md](SETUP.md). Start and enter the Docker container.

Then, inside the Docker container, run the command below to install dependencies needed for ALOHA:
```bash
uv sync --extra cu128 --group aloha --python 3.10
```

## Training on ALOHA Robot Data

You can either train on our released and preprocessed ALOHA dataset ([nvidia/ALOHA-Cosmos-Policy](https://huggingface.co/datasets/nvidia/ALOHA-Cosmos-Policy)) or collect your own ALOHA robot demonstrations. We will describe the latter process to show how data preprocessing is done, followed by training.

Let's say that you have collected a set of expert demonstrations on the ALOHA robot using the [ALOHA repo](https://github.com/tonyzhaozh/aloha).

Then, you can use our `preprocess_split_aloha_data.py` script to preprocess the raw ALOHA dataset: downsize images from 480x640 to 256x256 and split into training and validation sets. Below are examples for a few tasks in our paper:

```bash
python experiments/robot/aloha/preprocess_split_aloha_data.py \
    --dataset_path /scr/user/data/aloha2/fold_shirt_1500_steps/ \
    --out_base_dir /scr/user/data/aloha2_preprocessed/ \
    --percent_val 0.01
python experiments/robot/aloha/preprocess_split_aloha_data.py \
    --dataset_path /scr/user/data/aloha2/put_candies_in_bowl_1000_steps/ \
    --out_base_dir /scr/user/data/aloha2_preprocessed/ \
    --percent_val 0.01
python experiments/robot/aloha/preprocess_split_aloha_data.py \
    --dataset_path /scr/user/data/aloha2/put_candy_in_bag_1000_steps/ \
    --out_base_dir /scr/user/data/aloha2_preprocessed/ \
    --percent_val 0.01
python experiments/robot/aloha/preprocess_split_aloha_data.py \
    --dataset_path /scr/user/data/aloha2/put_purple_eggplant_on_plate_250_steps/ \
    --out_base_dir /scr/user/data/aloha2_preprocessed/ \
    --percent_val 0.01
python experiments/robot/aloha/preprocess_split_aloha_data.py \
    --dataset_path /scr/user/data/aloha2/put_brown_chicken_wing_on_plate_250_steps/ \
    --out_base_dir /scr/user/data/aloha2_preprocessed/ \
    --percent_val 0.01
```

Afterwards, you will have several preprocessed datasets in a directory (e.g., `/scr/user/data/aloha2_preprocessed/` in the above example). This preprocessed dataset directory is what you will pass into `ALOHADataset`, e.g., `data_dir=/scr/user/data/aloha2_preprocessed/5_tasks_mixture/`. See the config for the experiment `cosmos2_v2v_2b_480p__aloha__mixture_20250905_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80_185_demos__dataset_train__aws` in `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py` for an example.

Before you begin training, you must precompute T5 embeddings for the task descriptions that will be cross-attended with the policy's diffusion transformer backbone so that the policy can pay attention to language inputs. See `cosmos_policy/datasets/save_aloha_t5_text_embeddings.py` for instructions on how to do this. Add the path to the T5 embeddings to the experiment config as `t5_text_embeddings_path`.

Note that we release the preprocessed ALOHA dataset used in the Cosmos Policy paper as an example: [nvidia/ALOHA-Cosmos-Policy](https://huggingface.co/datasets/nvidia/ALOHA-Cosmos-Policy). To download this, run:
```bash
# Downloads to a local directory called `ALOHA-Cosmos-Policy`
hf download nvidia/ALOHA-Cosmos-Policy --repo-type dataset --local-dir ALOHA-Cosmos-Policy
# Set the current base datasets directory as BASE_DATASETS_DIR (needed for training)
export BASE_DATASETS_DIR=$(pwd)
```

Afterwards, you can begin training! The command below will launch training using our released ALOHA-Cosmos-Policy dataset (set `--nproc_per_node` to the number of GPUs available):

```bash
# Set BASE_DATASETS_DIR to the directory containing the ALOHA datasets
export BASE_DATASETS_DIR=/PATH/TO/BASE/DATASETS/DIRECTORY  # E.g., `/home/user/data/` if `ALOHA-Cosmos-Policy` is in this directory

uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80"
```

The above training command should reproduce our Cosmos Policy results if the 50K step checkpoint is evaluated. It will train Cosmos Policy using 3 input images (1 third-person image + 2 wrist camera images). Note that we use a sharp learning rate decay after a certain point (20K steps in the command above) since doing so speeds up training convergence (train L1 loss spikes down from our experience). Unlike LIBERO and RoboCasa, we found 1 node of 8 80GB GPUs to be sufficient since our real-world dataset is small.

Best practices for fine-tuning:
* In general, we recommend fine-tuning until training L1 loss reaches 0.01.
* Depending on your dataset size, you may need to adjust some hyperparameters. For example, if you use a large dataset with over 300 demos, you may need to decay the learning rate later and train for longer for best performance. Decaying too earlier can lead to a suboptimal policy.
* Please be sure to test your policy with the same device/GPU used to train it! Otherwise, performance may drop substantially.

If you run into any issues, please open a new GitHub issue.

## Launching ALOHA Robot Evaluations

(More detailed instructions coming soon!)

For now, please see `deploy.py` (remote policy server script) and `run_aloha_eval.py` (client script) as mentioned at the top of this guide. The workflow is similar to the workflow in the OpenVLA-OFT repo (see [here](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md#launching-aloha-robot-evaluations) for details). We include usage examples at the top of these two scripts.
