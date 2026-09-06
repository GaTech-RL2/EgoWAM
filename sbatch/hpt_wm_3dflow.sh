#!/bin/bash
# 3D-Flow world model: predict camera-stabilized dense 3D point motion alongside the action.
#
#   sbatch sbatch/hpt_wm_3dflow.sh <task> [regime] [hydra.overrides=...]
#
# task:   cup_saucer | fold_clothes | bag_grocery
# regime: eva (robot only) | indomain (1:1 human) | everseA (~10:1, the default)
#
# Needs Track4World flow sidecars for the episodes selected. flow3d=point3d adds the
# point-flow keys to the schematic; without it the loader never publishes them.
#SBATCH --job-name=hpt_wm_3dflow
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

if [ "$REGIME" = eva ]; then MODEL=hpt_wm_eva_3dflow; else MODEL=hpt_wm_cotrain_3dflow; fi

DESC=hpt_wm_3dflow_${REGIME}_${TASK}
# Isolate the flow-stats cache per run (single-user contract; recompute on data change).
export FLOW_STATS_CACHE_DIR="${FLOW_STATS_CACHE_DIR:-$ROOT/.cache/flow_stats/${DESC}}"

echo "[$(date)] host=$(hostname) task=$TASK regime=$REGIME model=$MODEL"
python egowam/trainHydra.py \
    --config-name=train_zarr_wm.yaml \
    flow3d=point3d \
    data=${TASK}/${REGIME}_wm3dflow \
    model=$MODEL \
    logger.wandb.project=${WANDB_PROJECT:-egowam} \
    name=${TASK} \
    description=${DESC} \
    "${EXTRA[@]}"
rc=$?
echo "[$(date)] EXIT=$rc"
exit $rc
