# Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

<p align="center">
  <a href="https://arxiv.org/abs/2601.16163">Paper</a>&nbsp | <a href="https://research.nvidia.com/labs/dir/cosmos-policy/">Project Website</a>&nbsp | 🤗 <a href="https://huggingface.co/collections/nvidia/cosmos-policy">Models & Training Data</a>&nbsp | <a href="https://youtu.be/V2qdFD9n5BM">Summary Video</a>
</p>

## System Requirements

Inference with base Cosmos Policy only (i.e., no model-based planning):
* 1 GPU with 6.8 GB VRAM for LIBERO sim benchmark tasks
* 1 GPU with 8.9 GB VRAM for RoboCasa sim benchmark tasks
* 1 GPU with 6.0 GB VRAM for ALOHA robot tasks

Inference with Cosmos Policy + model-based planning (best-of-N search) on ALOHA robot tasks:
* Minimum (serial inference): 1 GPU with 10.0 GB VRAM
* Recommended (parallel inference): N GPUs with 10.0 GB VRAM each

Training:
* Generally, it is recommended to have at least 1 node of 8 80GB GPUs. For the experiments in the Cosmos Policy paper, we used 8 80GB GPUs (H100s) for 48 hours for small-scale ALOHA robot data fine-tuning (<200 demos), 32 80GB GPUs (H100s) for 48 hours for RoboCasa training (1200 demos), and 64 80GB GPUs (H100s) for 48 hours for LIBERO training (2000 demos). If you have fewer GPUs, you can use gradient accumulation to increase total batch size, which we found leads to faster convergence than taking more gradient steps with a smaller batch size.

## Quick Start

First, set up a Docker container following the instructions in [SETUP.md](SETUP.md).

Then, inside the Docker container, enter a Python shell via: `uv run --extra cu128 --group libero --python 3.10 python`.

Then, run the Python code below to generate (1) robot actions, (2) predicted future state (represented by robot proprioception and future image observations), and (3) future state value (expected cumulative rewards):

```python
import pickle
import torch
from PIL import Image
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
    get_t5_embedding_from_cache,
)

# Instantiate config (see PolicyEvalConfig in cosmos_policy/experiments/robot/libero/run_libero_eval.py for definitions)
cfg = PolicyEvalConfig(
    config="cosmos_predict2_2b_480p_libero__inference_only",
    ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    config_file="cosmos_policy/config/config.py",
    dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
    t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
    use_wrist_image=True,
    use_proprio=True,
    normalize_proprio=True,
    unnormalize_actions=True,
    chunk_size=16,
    num_open_loop_steps=16,
    trained_with_image_aug=True,
    use_jpeg_compression=True,
    flip_images=True,  # Only for LIBERO; images render upside-down
    num_denoising_steps_action=5,
    num_denoising_steps_future_state=1,
    num_denoising_steps_value=1,
)
# Load dataset stats for action/proprio scaling
dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
# Initialize T5 text embeddings cache
init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
# Load model
model, cosmos_config = get_model(cfg)
# Load sample observation:
#   observation (dict): {
#     "primary_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "proprio": robot proprioceptive state,
#   }
with open("cosmos_policy/experiments/robot/libero/sample_libero_10_observation.pkl", "rb") as file:
    observation = pickle.load(file)
    task_description = "put both the alphabet soup and the tomato sauce in the basket"
# Generate robot actions, future state (proprio + images), and value
action_return_dict = get_action(
    cfg,
    model,
    dataset_stats,
    observation,
    task_description,
    num_denoising_steps_action=cfg.num_denoising_steps_action,
    generate_future_state_and_value_in_parallel=True,
)
# Print actions
print(f"Generated action chunk: {action_return_dict['actions']}")
# Save future image predictions (third-person image and wrist image)
img_path1, img_path2 = "future_image.png", "future_wrist_image.png"
Image.fromarray(action_return_dict['future_image_predictions']['future_image']).save(img_path1)
Image.fromarray(action_return_dict['future_image_predictions']['future_wrist_image']).save(img_path2)
print(f"Saved future image predictions to:\n\t{img_path1}\n\t{img_path2}")
# Print value
print(f"Generated value: {action_return_dict['value_prediction']}")
```

If you run into runtime errors, you may need to enter the Python shell via `uv run   --extra cu128   --group libero   --python 3.10   python` before running the code above.

## Installation

See [SETUP.md](SETUP.md) for instructions on setting up the environment.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for fine-tuning/evaluating on LIBERO simulation benchmark task suites.

See [ROBOCASA.md](ROBOCASA.md) for fine-tuning/evaluating on RoboCasa simulation benchmark tasks.

See [ALOHA.md](ALOHA.md) for fine-tuning/evaluating on real-world ALOHA robot tasks.

## Support

If you run into any issues, please open a new GitHub issue. For critical blocking issues, please email Moo Jin Kim (moojink@cs.stanford.edu) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2601.16163):

```bibtex
@article{kim2026cosmos,
  title={Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning},
  author={Kim, Moo Jin and Gao, Yihuai and Lin, Tsung-Yi and Lin, Yen-Chen and Ge, Yunhao and Lam, Grace and Liang, Percy and Song, Shuran and Liu, Ming-Yu and Finn, Chelsea and Gu, Jinwei},
  journal={arXiv preprint arXiv:2601.16163},
  year={2026}
}
```
