#!/bin/bash
# RoboTwin 3D-point-flow WM training (single-aloha or cross), 1x l40s on overcap.
# Usage: sbatch sbatch/robotwin/train_rt_3dflow.sh <setting> <task> [epochs]
#   setting = single | cross ; task = pick | stack | hang
# Auto-resumes from <rundir>/checkpoints/last.ckpt on requeue. wandb project=egowam_robotwin.
#SBATCH --job-name=rt3dflow
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --partition="overcap"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --qos="short"
#SBATCH --requeue
#SBATCH --open-mode=append

set -uo pipefail
SETTING=${1:?setting: single|cross}
TASK=${2:?task: pick|stack|hang}
EPOCHS=${3:-2000}

# sbatch copies this file to a spool path, so ${BASH_SOURCE[0]} does NOT point at the repo.
# Anchor on SLURM_SUBMIT_DIR (the directory you ran sbatch from) instead.
ROOT=${EGOWAM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
ZARR=${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}/zarrDataset
FLOWBASE=${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}_3d_flow_track4world
mkdir -p ${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}/_runs/_logs
cd "$ROOT"
source emimic2/bin/activate
export PYTHONIOENCODING=utf-8 HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$TASK" in
  pick)  TDIR=pick_diverse_bottles; SET=aloha_arx_piper_ur5;;
  *) echo "only the pick task ships in this release (got: $TASK)"; exit 1;;
esac

DESC=3dflow_${SETTING}_${TASK}
RUNDIR=${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}/_runs/${DESC}
mkdir -p "$RUNDIR/checkpoints"
# Isolate the flow-stats cache per run (single-user contract; recompute on data change).
export FLOW_STATS_CACHE_DIR="$RUNDIR/flow_stats"

COMMON=(
  trainer.max_epochs="${EPOCHS}"
  trainer.check_val_every_n_epoch=100
  # ckpt every 50 at train-epoch-end (cheap progress banking for preemptible overcap).
  callbacks.model_checkpoint.every_n_epochs=50
  +callbacks.model_checkpoint.save_on_train_epoch_end=true
  logger.wandb.project=egowam_robotwin
  name="rt_${DESC}"
  description="${DESC}"
  hydra.run.dir="${RUNDIR}"
)
CKPT="${RUNDIR}/checkpoints/last.ckpt"
if [ -f "$CKPT" ]; then COMMON+=("ckpt_path=${CKPT}"); echo "[resume] from ${CKPT}"; fi

echo "[$(date)] host=$(hostname) setting=$SETTING task=$TASK 3dflow epochs=$EPOCHS desc=$DESC"
if [ "$SETTING" = single ]; then
  FOLDER=${ZARR}/${TDIR}/aloha-agilex_clean_50
  FLOWROOT=${FLOWBASE}/${TDIR}/aloha-agilex_clean_50
  python egowam/trainHydra.py --config-name=train_zarr_wm_robotwin \
    flow3d=point3d_rt \
    model=robotwin/hpt_wm_rt_aloha_3dflow data=robotwin/aloha_3dflow \
    data.train_datasets.robotwin_aloha_bimanual.resolver.folder_path="${FOLDER}" \
    data.valid_datasets.robotwin_aloha_bimanual.resolver.folder_path="${FOLDER}" \
    data.train_datasets.robotwin_aloha_bimanual.resolver.key_map.point_flow_root="${FLOWROOT}" \
    data.valid_datasets.robotwin_aloha_bimanual.resolver.key_map.point_flow_root="${FLOWROOT}" \
    "${COMMON[@]}"
else
  python egowam/trainHydra.py --config-name=train_zarr_wm_robotwin \
    flow3d=point3d_rt \
    model=robotwin/hpt_wm_rt_cross_${SET}_3dflow data=robotwin/${TASK}_cross_3dflow \
    "${COMMON[@]}"
fi
rc=$?
echo "[$(date)] EXIT=$rc"
exit $rc   # propagate so SLURM shows FAILED on a crash (not a false COMPLETED)
