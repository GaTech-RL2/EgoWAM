#!/bin/bash

# Set BASE_DATASETS_DIR to the directory containing the ALOHA-Cosmos-Policy dataset
export BASE_DATASETS_DIR="/coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy"  # E.g., `/home/user/data/` if `ALOHA-Cosmos-Policy` is in this directory
# Use local filesystem for TMPDIR to avoid NFS cleanup issues
# /dev/shm is shared memory (fast, local) - fallback to /tmp if not available
export TMPDIR="${TMPDIR:-/dev/shm}"
if [ ! -d "$TMPDIR" ] || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="/tmp"
fi
mkdir -p "$TMPDIR/cosmos_policy_$$"
export TMPDIR="$TMPDIR/cosmos_policy_$$"
export IMAGINAIRE_OUTPUT_ROOT="/coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/logs"

# uv run --extra cu128 --group aloha --python 3.10 \
python -m cosmos_policy.experiments.robot.aloha.run_aloha_eval_visualize \
  --config cosmos_predict2_2b_480p_aloha_one_demo_one_episode \
  --ckpt_path /coc/flash7/mlin365/cosmos-policy/logs/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_aloha_one_demo_one_episode/checkpoints/iter_000050000 \
  --config_file cosmos_policy/config/config.py \
  --use_wrist_image True \
  --use_proprio True \
  --normalize_proprio True \
  --unnormalize_actions True \
  --trained_with_image_aug True \
  --chunk_size 50 \
  --local_log_dir cosmos_policy/experiments/robot/aloha/logs/ \
  --seed 7 \
  --deterministic True \
  --run_id_note aloha_50K \
  --use_jpeg_compression True \
  --flip_images False \
  --num_denoising_steps_action 5 \
  --num_denoising_steps_future_state 1 \
  --num_denoising_steps_value 1 \
  --episode_indices 0 \
  --n 100 \
  --save_video_path ./visualizations

