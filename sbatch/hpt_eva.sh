#!/bin/bash
# BC (action supervision only), robot data only.
#
#   sbatch sbatch/hpt_eva.sh <task> [hydra.overrides=...]
#
# task: cup_saucer | fold_clothes | bag_grocery
#SBATCH --job-name=hpt_eva
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
EXTRA=("${@:2}")   # any further args are passed to Hydra verbatim
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

DESC=hpt_eva_${TASK}
echo "[$(date)] host=$(hostname) task=$TASK model=hpt_bc_eva data=${TASK}/eva"
python egowam/trainHydra.py \
    --config-name=train_zarr.yaml \
    data=${TASK}/eva \
    model=hpt_bc_eva \
    logger.wandb.project=${WANDB_PROJECT:-egowam} \
    name=${TASK} \
    description=${DESC} \
    "${EXTRA[@]}"
rc=$?
echo "[$(date)] EXIT=$rc"
exit $rc
