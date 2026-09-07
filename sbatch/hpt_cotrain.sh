#!/bin/bash
# BC (action supervision only) co-trained on human data.
#
#   sbatch sbatch/hpt_cotrain.sh <task> [regime] [hydra.overrides=...]
#
# task:   cup_saucer | fold_clothes | bag_grocery
# regime: indomain (1:1 human) | everseA (~10:1 EgoVerse, the default)
#SBATCH --job-name=hpt_cotrain
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
  indomain|everseA) ;;
  *) echo "regime must be indomain|everseA (got: $REGIME)"; exit 2;;
esac

DESC=hpt_cotrain_${REGIME}_${TASK}
echo "[$(date)] host=$(hostname) task=$TASK regime=$REGIME data=${TASK}/${REGIME}"
python egowam/trainHydra.py \
    --config-name=train_zarr.yaml \
    data=${TASK}/${REGIME} \
    model=hpt_bc_cotrain \
    trainer.strategy=ddp_find_unused_parameters_true \
    logger.wandb.project=${WANDB_PROJECT:-egowam} \
    name=${TASK} \
    description=${DESC} \
    "${EXTRA[@]}"
rc=$?
echo "[$(date)] EXIT=$rc"
exit $rc
