# Cosmos Policy in the LIBERO Simulation Benchmark

(Instructions here are adapted from [OpenVLA-OFT](https://github.com/moojink/openvla-oft).)

## Relevant Files

Evaluation
* `cosmos_policy/experiments/robot/libero/`: LIBERO eval files
  * `run_libero_eval.py`: LIBERO eval script
  * `libero_utils.py`: LIBERO eval utils
* `cosmos_policy/experiments/robot/`: General eval utils files
  * `cosmos_utils.py`: Cosmos Policy-specific eval utils
  * `robot_utils.py`: Other eval utils

Training
* `cosmos_policy/scripts/train.py`: Cosmos Policy training script (NOTE: Do NOT use other train.py scripts that may exist elsewhere in the codebase for base video models)


## Setup

First, follow the instructions here: [SETUP.md](SETUP.md). Start and enter the Docker container.

Then, inside the Docker container, run the command below to install dependencies needed for LIBERO:
```bash
uv sync --extra cu128 --group libero --python 3.10
```

### Downloading Modified LIBERO Datasets
(Optional, if you plan to launch training) To download the modified LIBERO datasets that we used in our experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 datasets altogether in one superset. You can use these to train Cosmos Policy or train other models. This step is optional since we provide pretrained Cosmos Policy checkpoints.
```bash
# Downloads to a local directory called `LIBERO-Cosmos-Policy`
hf download nvidia/LIBERO-Cosmos-Policy --repo-type dataset --local-dir LIBERO-Cosmos-Policy
# Set the current base datasets directory as BASE_DATASETS_DIR (needed for training)
export BASE_DATASETS_DIR=$(pwd)
```

## Launching LIBERO Evaluations

We trained Cosmos Policy on four LIBERO task suites altogether in one run: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long). Below is the pretrained checkpoint:
* [nvidia/Cosmos-Policy-LIBERO-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)

To start evaluations with this checkpoint, run the command below, where `task_suite_name` is one of the following: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`. Each will automatically download the checkpoint above. You can set the `TRANSFORMERS_CACHE` and `HF_HOME` environment variable to change where the checkpoint files get cached.

```bash
uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 16 \
    --num_open_loop_steps 16 \
    --task_suite_name libero_10 \
    --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
    --randomize_seed False \
    --data_collection False \
    --available_gpus "0,1,2,3,4,5,6,7" \
    --seed 195 \
    --use_variance_scale False \
    --deterministic True \
    --run_id_note chkpt45000--5stepAct--seed195--deterministic \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1
```

Notes:
* The evaluation script will run 500 trials by default (10 tasks x 50 episodes each). You can modify the number of trials per task by setting `--num_trials_per_task`. Note that the `--seed` and `--deterministic` arguments are important if you want to exactly reproduce the results in the Cosmos Policy paper. We used seeds {195, 196, 197} and `--deterministic True`. You can change these, but the results may vary slightly (and change every time you run the evaluation).
* The evaluation script logs results locally. You can also log results in Weights & Biases by setting `--use_wandb True` and specifying `--wandb_entity <ENTITY>` and `--wandb_project <PROJECT>`.
* The results reported in our paper were obtained using **Python 3.12.3 (and 3.10.18) and PyTorch 2.7.0** on an **NVIDIA H100 GPU**, averaged over three random seeds. Note that results may vary slightly if you use a different PyTorch version or different hardware.

## Training on LIBERO Datasets

First, download the LIBERO datasets following the instructions above.

Then, launch the training script below, setting `--nproc_per_node` to the number of GPUs available:

```bash
# Set BASE_DATASETS_DIR to the directory containing the LIBERO-Cosmos-Policy dataset
export BASE_DATASETS_DIR=/PATH/TO/BASE/DATASETS/DIRECTORY  # E.g., `/home/user/data/` if `LIBERO-Cosmos-Policy` is in this directory

uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_libero" \
  trainer.grad_accum_iter=8
```

This command will train with effective batch size = (local batch size) * (# GPUs) * (gradient accumulation factor) = 30 * 8 * 8 = 1920, which matches the effective batch size we used for LIBERO in the Cosmos Policy paper. The command assumes access to 1 node of 8 80GB GPUs (e.g. H100s). The config above uses sharp learning rate decay after 30K steps. For faster iteration, our original experiment used 8 nodes (64 H100s total) with no gradient accumulation and trained for 40K gradient steps (~48 hours total). Note that using fewer nodes and trying to reproduce the full run will take significantly longer.

You can modify various experiment config variables in `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`. Alternatively, you can change variables on the command line. For example, appending `dataloader_train.batch_size=4` to the command above sets the local batch size (per GPU) to 4.

In general, we recommend training until action L1 loss reaches around ~0.010. (We observed ~0.012 L1 loss after 40K gradient steps using 8 nodes (64 H100s) and no gradient accumulation -- i.e., same command as above except after removing `trainer.grad_accum_iter=8`.) Please be sure to test your policy with the same device/GPU used to train it! Otherwise, performance may drop substantially.
