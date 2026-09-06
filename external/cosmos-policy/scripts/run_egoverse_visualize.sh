#!/bin/bash

# Use local filesystem for TMPDIR to avoid NFS cleanup issues
export TMPDIR="${TMPDIR:-/dev/shm}"
if [ ! -d "$TMPDIR" ] || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="/tmp"
fi
mkdir -p "$TMPDIR/cosmos_policy_$$"
export TMPDIR="$TMPDIR/cosmos_policy_$$"
export IMAGINAIRE_OUTPUT_ROOT="/coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/logs"

source /coc/flash7/bli678/Projects/EgoVerse/emimic/bin/activate

python -m cosmos_policy.experiments.robot.egoverse.run_egoverse_eval_visualize \
  --config cosmos_predict2_2b_480p_egoverse \
  --ckpt_path /coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/logs/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_eva/checkpoints/iter_000060000 \
  --config_file cosmos_policy/config/config.py \
  --chunk_size 25 \
  --num_denoising_steps 35 \
  --ar_future_prediction True \
  --num_denoising_steps_future_state 35 \
  --seed 195 \
  --deterministic True \
  --run_id_note visualize \
  --episode_indices 0 \
  --max_episodes 1 \
  --n 10 \
  --save_video_path ./visualizations \
  --local_log_dir cosmos_policy/experiments/robot/egoverse/logs/ \
  --video_fps 10 \
