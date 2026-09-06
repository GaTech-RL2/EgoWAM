# Set BASE_DATASETS_DIR to the directory containing the LIBERO-Cosmos-Policy dataset
export BASE_DATASETS_DIR="/coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy"  # E.g., `/home/user/data/` if `LIBERO-Cosmos-Policy` is in this directory
# Use local filesystem for TMPDIR to avoid NFS cleanup issues
# /dev/shm is shared memory (fast, local) - fallback to /tmp if not available
export TMPDIR="${TMPDIR:-/dev/shm}"
if [ ! -d "$TMPDIR" ] || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="/tmp"
fi
mkdir -p "$TMPDIR/cosmos_policy_$$"
export TMPDIR="$TMPDIR/cosmos_policy_$$"
export IMAGINAIRE_OUTPUT_ROOT="/coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/logs"

uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval_overfit \
    --config cosmos_predict2_2b_480p_libero_one_demo_one_episode \
    --ckpt_path /coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/logs/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_libero_one_demo_one_episode/checkpoints/iter_000100000 \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --trained_with_image_aug True \
    --chunk_size 16 \
    --num_open_loop_steps 16 \
    --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
    --randomize_seed False \
    --data_collection False \
    --available_gpus "0" \
    --seed 195 \
    --use_variance_scale False \
    --deterministic True \
    --run_id_note overfit--ckpt100K--5stepAct--seed195--deterministic \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --num_trials_per_episode 1
