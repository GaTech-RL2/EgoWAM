#!/bin/bash
# DINO world model: predict the future frame's latent alongside the action.
#
#   sbatch sbatch/hpt_wm_rae.sh <task> [regime] [hydra.overrides=...]
#
# task:   cup_saucer | fold_clothes | bag_grocery
# regime: eva (robot only) | indomain (1:1 human) | everseA (~10:1, the default)
#SBATCH --job-name=hpt_wm_rae
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="l40s:1"
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
# Node-local, never NFS: multiprocessing's temp dir must not live on /coc/flash7/scratch.
# On NFS a deleted-but-open file is silly-renamed to .nfsXXXX and rmtree then fails with
# EBUSY at shutdown, burying the real error under pages of traceback.
export TMPDIR="${TMPDIR_OVERRIDE:-${SLURM_TMPDIR:-/tmp}}"
mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Force Lightning's subprocess DDP launcher rather than SLURMEnvironment.
export SLURM_JOB_NAME=bash

case "$TASK" in
  cup_saucer|fold_clothes|bag_grocery) ;;
  *) echo "task must be cup_saucer|fold_clothes|bag_grocery (got: $TASK)"; exit 2;;
esac
case "$REGIME" in
  eva|indomain|everseA) ;;
  *) echo "regime must be eva|indomain|everseA (got: $REGIME)"; exit 2;;
esac

# Robot-only runs use the eva model and declare one domain; co-train runs add aria.
if [ "$REGIME" = eva ]; then
  MODEL=hpt_wm_eva_rae
  AUX=( 'model.robomimic_model.auxiliary_ac_keys.eva_bimanual=[future_image_latent]' )
else
  MODEL=hpt_wm_cotrain_rae
  AUX=( 'model.robomimic_model.auxiliary_ac_keys.eva_bimanual=[future_image_latent]'
        'model.robomimic_model.auxiliary_ac_keys.aria_bimanual=[future_image_latent]' )
fi

DESC=hpt_wm_rae_${REGIME}_${TASK}
echo "[$(date)] host=$(hostname) task=$TASK regime=$REGIME model=$MODEL"
python egowam/trainHydra.py \
    --config-name=train_zarr_wm.yaml \
    data=${TASK}/${REGIME}_wm \
    model=$MODEL \
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
