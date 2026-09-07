#!/bin/bash
# Pixel-PT world model: the Pixel target, but the future-image head starts from real
# Wan2.1-T2V-1.3B video weights instead of from scratch.
#
#   sbatch sbatch/hpt_wm_pixelpt.sh <task> [regime] [hydra.overrides=...]
#
# task:   cup_saucer | fold_clothes | bag_grocery
# regime: eva (robot only) | indomain (1:1 human) | everseA (~10:1, the default)
#SBATCH --job-name=hpt_wm_pixelpt
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# TWO GPUs, unlike the other variants. The Wan head is 1.42B parameters, so fp32 weights
# plus gradients plus Adam moments are ~21 GiB before a single activation -- one 44-48 GiB
# card cannot also hold the batch. The per-GPU batch is halved below so the GLOBAL batch
# stays 32 and this remains comparable to the one-GPU variants.
#SBATCH --gpus-per-node="l40s:2"
#SBATCH --qos="short"

TASK=${1:?task: cup_saucer|fold_clothes|bag_grocery}
REGIME=${2:-everseA}
EXTRA=("${@:3}")   # any further args are passed to Hydra verbatim
set -uo pipefail
ROOT=${EGOWAM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$ROOT"
mkdir -p sbatch_logs
source "${EGOWAM_VENV:-$ROOT/emimic}/bin/activate"

# Keep multiprocessing scratch off /tmp: when a node's /tmp is full, tempfile falls back to
# the CWD and litters the repo with pymp-* dirs (or dies with ENOSPC mid-run).
export TMPDIR="${TMPDIR_OVERRIDE:-${SLURM_TMPDIR:-/tmp}}"
mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Wan2.1 is a PUBLIC repo, so unlike hpt_wm_vae.sh there is no gated-asset preflight. The
# weights are pulled on first use; see docs/install.md for prefetching them, which matters
# on clusters whose login nodes kill long downloads.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Force Lightning's subprocess DDP launcher rather than SLURMEnvironment, so one task
# spawns both ranks and --ntasks-per-node stays 1.
export SLURM_JOB_NAME=bash

case "$TASK" in
  cup_saucer|fold_clothes|bag_grocery) ;;
  *) echo "task must be cup_saucer|fold_clothes|bag_grocery (got: $TASK)"; exit 2;;
esac
case "$REGIME" in
  eva|indomain|everseA) ;;
  *) echo "regime must be eva|indomain|everseA (got: $REGIME)"; exit 2;;
esac

NUM_GPUS=$(echo "${SLURM_GPUS_PER_NODE:-2}" | cut -d: -f2)
BATCH=$((32 / NUM_GPUS))   # global batch stays 32

# Robot-only runs use the eva model and declare one domain; co-train runs add aria.
if [ "$REGIME" = eva ]; then
  MODEL=hpt_wm_eva_pixelpt
  AUX=( 'model.robomimic_model.auxiliary_ac_keys.eva_bimanual=[future_image_latent]' )
  BS=( data.train_dataloader_params.eva_bimanual.batch_size=$BATCH
       data.valid_dataloader_params.eva_bimanual.batch_size=$BATCH )
else
  MODEL=hpt_wm_cotrain_pixelpt
  AUX=( 'model.robomimic_model.auxiliary_ac_keys.eva_bimanual=[future_image_latent]'
        'model.robomimic_model.auxiliary_ac_keys.aria_bimanual=[future_image_latent]' )
  AUX+=( 'model.robomimic_model.shared_auxiliary_ac_keys=[future_image_latent]'
         model.robomimic_model.use_shared_vae_stem=false )
  BS=( data.train_dataloader_params.eva_bimanual.batch_size=$BATCH
       data.valid_dataloader_params.eva_bimanual.batch_size=$BATCH
       data.train_dataloader_params.aria_bimanual.batch_size=$BATCH
       data.valid_dataloader_params.aria_bimanual.batch_size=$BATCH )
fi

DESC=hpt_wm_pixelpt_${REGIME}_${TASK}
echo "[$(date)] host=$(hostname) task=$TASK regime=$REGIME model=$MODEL gpus=$NUM_GPUS per-gpu-batch=$BATCH"
python egowam/trainHydra.py \
    --config-name=train_zarr_wm.yaml \
    trainer=ddp_long \
    data=${TASK}/${REGIME}_wm \
    model=$MODEL \
    launch_params.gpus_per_node=$NUM_GPUS \
    "${BS[@]}" \
    model.robomimic_model.wm_prediction_mode=joint \
    "${AUX[@]}" \
    model.robomimic_model.auxiliary_loss_weights.future_ee_pose=0.0 \
    model.robomimic_model.auxiliary_loss_weights.future_image_latent=1.0 \
    model.robomimic_model.auxiliary_loss_weights.future_flow=0.0 \
    model.robomimic_model.training_stage=3 \
    logger.wandb.project=${WANDB_PROJECT:-egowam} \
    name=${TASK} \
    description=${DESC} \
    "${EXTRA[@]}"
rc=$?
echo "[$(date)] EXIT=$rc"
exit $rc
