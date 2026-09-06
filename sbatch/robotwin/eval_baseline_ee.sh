#!/bin/bash
# Held-out success-rate eval for an EE-Cartesian baseline (ACT / DP) on a40 overcap.
# Mirrors eval_seeds.sh (EgoWAM template): resumable per-seed JSON + requeue-safe. A node that fails
# the SAPIEN render check exits 0 inside RoboTwin, so we detect "Render Error" -> exit 1 to requeue.
#
# Usage:  sbatch sbatch/robotwin/eval_baseline_ee.sh <policy> <task> <ckpt> <results_json> [ntarget]
#   ACT: <ckpt> = ckpt DIR  (policy_last.ckpt + dataset_stats.pkl)
#   DP : <ckpt> = ckpt FILE (e.g. policy/DP/checkpoints/hanging_mug-demo_clean-50-0/50.ckpt)
# e.g. sbatch sbatch/robotwin/eval_baseline_ee.sh dp hanging_mug \
#        policy/DP/checkpoints/hanging_mug-demo_clean-50-0/50.ckpt \
#        ${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}/_baselines_ee/_eval/dp_hanging_mug.json 100
#SBATCH --job-name=evalee
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
POLICY=${1:?policy act|dp}
TASK=${2:?task_name}
CKPT=${3:?ckpt (ACT: dir | DP: .ckpt file)}
RESULTS=${4:?results_path}
NTARGET=${5:-100}

# sbatch copies this file to a spool path, so ${BASH_SOURCE[0]} does NOT point at the repo.
# Anchor on SLURM_SUBMIT_DIR (the directory you ran sbatch from) instead.
ROOT=${EGOWAM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$ROOT/external/RoboTwin"
source "${EGOWAM_SIM_VENV:-$ROOT/.venv-sim}/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8 HYDRA_FULL_ERROR=1 DISPLAY="" PYOPENGL_PLATFORM=egl
export TMPDIR=${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}/_sim_rollout/_eval50/_logs

echo "[$(date)] host=$(hostname) policy=$POLICY task=$TASK ntarget=$NTARGET results=$RESULTS (attempt $SLURM_JOB_ID)"
RUNLOG=$(dirname "$RESULTS")/_logs/${POLICY}_${TASK}_run_${SLURM_JOB_ID}.log
mkdir -p "$(dirname "$RUNLOG")"
"${EGOWAM_SIM_VENV:-$ROOT/.venv-sim}/bin/python" -m egowam.scripts.robotwin_process.eval_baseline_ee \
  --policy "$POLICY" --task "$TASK" --task-config demo_clean --robot aloha-agilex \
  --ckpt-dir "$CKPT" --results-path "$RESULTS" \
  --n-target "$NTARGET" --seed-base 100000 \
  --exec-horizon 30 --steps-per-wp 15 --max-joint-vel 1.0 2>&1 | tee "$RUNLOG"
code=${PIPESTATUS[0]}
if grep -q "Render Error" "$RUNLOG"; then
  echo "[$(date)] RENDER-FAIL on $(hostname) -> exit 1 to requeue"
  exit 1
fi
echo "[$(date)] EXIT=$code"
exit "$code"
