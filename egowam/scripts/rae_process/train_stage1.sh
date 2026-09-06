#!/bin/bash
# RAE stage-1: train the decoder that reconstructs images from frozen DINOv2 features.
# Its checkpoint is what the DINO variant loads via EGOWAM_RAE_CKPT_ROOT.
#
#   sbatch egowam/scripts/rae_process/train_stage1.slurm [results-tag]
#
# Set WANDB_API_KEY in your environment (or ~/.netrc) to log; unset means offline.
#SBATCH --job-name=rae_stage1
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# l40s is scarce on rl2-lab (4 nodes); a40 has the same 48 GB and far more capacity.
# Override at submit time if you want a specific type: sbatch --gpus-per-node=l40s:1 ...
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos=short

set -uo pipefail
ROOT=${EGOWAM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$ROOT"
mkdir -p sbatch_logs
source "${EGOWAM_VENV:-$ROOT/.venv}/bin/activate"

# Keep multiprocessing scratch off /tmp (see sbatch/release_check/smoke_train.sh).
# Node-local, never NFS: multiprocessing's temp dir must not live on /coc/flash7/scratch.
# On NFS a deleted-but-open file is silly-renamed to .nfsXXXX and rmtree then fails with
# EBUSY at shutdown, burying the real error under pages of traceback.
export TMPDIR="${TMPDIR_OVERRIDE:-${SLURM_TMPDIR:-/tmp}}"
mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_JOB_NAME=bash
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_MODE=offline

RAE_ROOT=${EGOWAM_RAE_ROOT:-$ROOT/external/RAE}
TAG=${1:-$(date +%Y%m%d_%H%M%S)}
DATA_CFG=${RAE_STAGE1_DATA:-$ROOT/egowam/scripts/rae_process/data/eva_aria_cotrain.yaml}

# RAE's resume_utils.configure_experiment_dirs() asserts on EXPERIMENT_NAME and, when --wandb is
# passed, reads ENTITY and PROJECT. It writes to <results-dir>/<EXPERIMENT_NAME>, so results-dir is
# the parent and the tag names the run.
RESULTS=${EGOWAM_RAE_CKPT_ROOT:-$ROOT/logs/rae}
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-stage1_${TAG}}"
export ENTITY="${ENTITY:-${WANDB_ENTITY:-egowam}}"
export PROJECT="${PROJECT:-${WANDB_PROJECT:-egowam_rae}}"

echo "[$(date)] host=$(hostname) rae_root=$RAE_ROOT results=$RESULTS/$EXPERIMENT_NAME"
echo "[$(date)] data_config=$DATA_CFG"

# Trained from scratch. --finetune would start from the pretrained DINOv2-B decoder at
# models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt, which is not vendored (weights never
# are); add it back only if you have that checkpoint locally.
torchrun --standalone --nproc_per_node="${NPROC:-1}" \
  egowam/scripts/rae_process/train_stage1.py -- \
    --config configs/stage1/training/DINOv2-B_decXL.yaml \
    --data-config "$DATA_CFG" \
    --results-dir "$RESULTS" \
    --image-size "${IMAGE_SIZE:-256}" \
    --precision "${PRECISION:-bf16}" \
    ${WANDB_API_KEY:+--wandb}
rc=$?
echo "[$(date)] EXIT=$rc"
exit $rc
