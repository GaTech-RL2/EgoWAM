#!/bin/bash
# DINO world model with a DINOv2 vision stem instead of ResNet-18.
#
#   sbatch sbatch/hpt_wm_rae_dinov2.sh <task> [regime] [hydra.overrides=...]
#
# task:   cup_saucer | fold_clothes | bag_grocery
# regime: eva (robot only) | indomain (1:1 human) | everseA (~10:1, the default)
#
# Same world-model target as hpt_wm_rae.sh; only the image encoder differs. DINOv2's pretrained
# features generalize across scenes, lighting and backgrounds far better than a ResNet-18 trained
# from demonstrations alone, so prefer this when deploying outside the training scenes.
#
# NOTE: the paper's numbers are all ResNet-18 (hpt_wm_rae.sh). Runs from this launcher are not
# comparable to them.
#SBATCH --job-name=hpt_wm_rae_dinov2
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

# DINOv2 emits 768-dim features where ResNet-18 emits 256, so every stem that consumes an
# encoder has to widen with it, and the trunk needs a cross-attention bridge to accept 768-dim
# context. Encoders: front_img_1 + both wrist cameras. Consumers: the shared front stem and the
# two eva wrist stems.
STEM=()
for CAM in front_img_1 left_wrist_img right_wrist_img; do
  STEM+=( "model.robomimic_model.encoder_specs.${CAM}._target_=egowam.models.hpt_nets.DINOv2Stem"
          "model.robomimic_model.encoder_specs.${CAM}.output_dim=768"
          "+model.robomimic_model.encoder_specs.${CAM}.input_size=224" )
done
STEM+=( 'model.robomimic_model.shared_stem_specs.front_img_1.input_dim=768'
        'model.robomimic_model.shared_stem_specs.front_img_1.output_dim=768'
        'model.robomimic_model.shared_stem_specs.front_img_1.widths=[768]'
        '+model.robomimic_model.shared_stem_specs.front_img_1.specs.cross_attn.crossattn_context_dim=768' )
for CAM in left_wrist_img right_wrist_img; do
  STEM+=( "model.robomimic_model.stem_specs.eva_bimanual.${CAM}.input_dim=768"
          "model.robomimic_model.stem_specs.eva_bimanual.${CAM}.output_dim=768"
          "model.robomimic_model.stem_specs.eva_bimanual.${CAM}.widths=[768]"
          "+model.robomimic_model.stem_specs.eva_bimanual.${CAM}.specs.cross_attn.crossattn_context_dim=768" )
done

DESC=hpt_wm_rae_dinov2_${REGIME}_${TASK}
echo "[$(date)] host=$(hostname) task=$TASK regime=$REGIME model=$MODEL stem=DINOv2"
python egowam/trainHydra.py \
    --config-name=train_zarr_wm.yaml \
    data=${TASK}/${REGIME}_wm \
    model=$MODEL \
    model.robomimic_model.wm_prediction_mode=joint \
    "${AUX[@]}" \
    "${STEM[@]}" \
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
