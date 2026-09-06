#!/bin/bash
# Real RoboTwin WM training run (single-aloha or cross), 1x a40, rl2-lab qos=long (7d).
# Usage: sbatch sbatch/robotwin/train_rt.sh <setting> <task> <variant> [epochs]
#   setting = single | cross ; task = pick | stack | hang ; variant = hpt | vae | rae
# Auto-resumes from <rundir>/checkpoints/last.ckpt on requeue. wandb project=egowam_robotwin.
#SBATCH --job-name=rttrain
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --requeue
#SBATCH --open-mode=append

set -uo pipefail
SETTING=${1:?setting: single|cross}
TASK=${2:?task: pick|stack|hang}
VARIANT=${3:-hpt}
EPOCHS=${4:-2000}

# sbatch copies this file to a spool path, so ${BASH_SOURCE[0]} does NOT point at the repo.
# Anchor on SLURM_SUBMIT_DIR (the directory you ran sbatch from) instead.
ROOT=${EGOWAM_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
mkdir -p ${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}/_runs/_logs
cd "$ROOT"
source emimic2/bin/activate
export PYTHONIOENCODING=utf-8 HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VAE loads the gated Cosmos tokenizer; use mlin365's cached copy offline (no gated HF download).
# Scoped to vae only so HPT/RAE keep a writable default HF cache.
if [ "$VARIANT" = vae ]; then
  export HF_HOME=${HF_HOME:-/coc/flash7/mlin365/hf_cache}
  export HF_HUB_OFFLINE=1
fi

case "$TASK" in
  pick)  TDIR=pick_diverse_bottles; SET=aloha_arx_piper_ur5;;
  *) echo "only the pick task ships in this release (got: $TASK)"; exit 1;;
esac

DESC=${VARIANT}_${SETTING}_${TASK}                 # modality_setting_task (per existing sbatch style)
RUNDIR=${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}/_runs/${DESC}
mkdir -p "$RUNDIR/checkpoints"

COMMON=(
  trainer.max_epochs="${EPOCHS}"
  trainer.check_val_every_n_epoch=100
  # ckpt every 50 at train-epoch-end (decoupled from the 100-epoch val) so
  # preemptible overcap slots bank progress before they die; val cost unchanged.
  callbacks.model_checkpoint.every_n_epochs=50
  +callbacks.model_checkpoint.save_on_train_epoch_end=true
  logger.wandb.project=egowam_robotwin
  name="rt_${SETTING}_${TASK}"
  description="${DESC}"
  hydra.run.dir="${RUNDIR}"
)
CKPT="${RUNDIR}/checkpoints/last.ckpt"
if [ -f "$CKPT" ]; then COMMON+=("ckpt_path=${CKPT}"); echo "[resume] from ${CKPT}"; fi

# Only the `vae` variant's world-model head is the gated Wan/Cosmos tokenizer; the rae, hpt and
# 3dflow RoboTwin configs build their heads from local assets. Preflight it before the data pass.
if [ "$VARIANT" = vae ]; then
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  python egowam/scripts/check_gated_assets.py || exit 1
fi

echo "[$(date)] host=$(hostname) setting=$SETTING task=$TASK variant=$VARIANT epochs=$EPOCHS desc=$DESC"
if [ "$SETTING" = single ]; then
  FOLDER=${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}/zarrDataset/${TDIR}/aloha-agilex_clean_50
  python egowam/trainHydra.py --config-name=train_zarr_wm_robotwin \
    model=robotwin/hpt_wm_rt_aloha_${VARIANT} data=robotwin/aloha_single \
    data.train_datasets.robotwin_aloha_bimanual.resolver.folder_path="${FOLDER}" \
    data.valid_datasets.robotwin_aloha_bimanual.resolver.folder_path="${FOLDER}" \
    "${COMMON[@]}"
else
  python egowam/trainHydra.py --config-name=train_zarr_wm_robotwin \
    model=robotwin/hpt_wm_rt_cross_${SET}_${VARIANT} data=robotwin/${TASK}_cross \
    "${COMMON[@]}"
fi
rc=$?   # exit code of the python train (last cmd of the taken branch)
echo "[$(date)] EXIT=$rc"
exit $rc   # propagate so SLURM shows FAILED on a crash (not a false COMPLETED)
