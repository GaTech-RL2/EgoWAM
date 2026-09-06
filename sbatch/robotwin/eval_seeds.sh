#!/bin/bash
# Held-out 50-seed success-rate eval for one (variant,setting,task) config, on a40 overcap.
# Resumable (per-seed JSON via --results-path) and requeue-safe: overcap preemption OR a bad-renderer
# node both trigger a fresh attempt that resumes from the JSON. A node that fails the SAPIEN render
# check exits 0 inside RoboTwin, so we detect "Render Error" and exit non-zero -> --requeue reschedules.
#SBATCH --job-name=eval50
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --partition=overcap
#SBATCH --account=rl2-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a40:1
#SBATCH --qos=short
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=spd-13,kitt

set -uo pipefail
TASK=${1:?task_name}
CKPT=${2:?ckpt}
RESULTS=${3:?results_path}
TAG=${4:?variant_tag}
NIS=${5:-10}
NTARGET=${6:-100}

# sbatch copies this file to a spool path, so ${BASH_SOURCE[0]} does NOT point at the repo.
# Anchor on SLURM_SUBMIT_DIR (the directory you ran sbatch from) instead.
ROOT=${EGOWAM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$ROOT/external/RoboTwin"
source "${EGOWAM_SIM_VENV:-$ROOT/.venv-sim}/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8 HYDRA_FULL_ERROR=1 DISPLAY="" PYOPENGL_PLATFORM=egl

echo "[$(date)] host=$(hostname) tag=$TAG task=$TASK nis=$NIS results=$RESULTS (attempt on $SLURM_JOB_ID)"
RUNLOG=$(dirname "$RESULTS")/_logs/${TAG}_run_${SLURM_JOB_ID}.log
mkdir -p "$(dirname "$RUNLOG")"
"${EGOWAM_SIM_VENV:-$ROOT/.venv-sim}/bin/python" -m egowam.scripts.robotwin_process.sim_smoke \
  --task "$TASK" --task-config demo_clean --robot aloha-agilex --seeds 1 --mode eval_seeds \
  --ckpt "$CKPT" --results-path "$RESULTS" --variant-tag "$TAG" \
  --eval-seed-file "${EVAL_SEED_FILE:-$ROOT/egowam/scripts/robotwin_process/seed.txt}" \
  --n-target "$NTARGET" --seed-base 100000 --num-inference-steps "$NIS" \
  --ik mink --max-joint-vel 1.0 --exec-horizon 30 --steps-per-wp 15 2>&1 | tee "$RUNLOG"
code=${PIPESTATUS[0]}
if grep -q "Render Error" "$RUNLOG"; then
  echo "[$(date)] RENDER-FAIL on $(hostname) -> exit 1 to requeue"
  exit 1
fi
echo "[$(date)] EXIT=$code"
exit "$code"
