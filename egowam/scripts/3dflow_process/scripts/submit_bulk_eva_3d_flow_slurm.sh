#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Pipeline lives in egowm-hpt/egowam/scripts/3dflow_process/scripts; the track4world library,
# checkpoints, and config come from the external/Track4World submodule (run
# 3dflow_process/sync_overlay.sh once to wire the egoverse overlay into it).
EGOWM_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
T4W_ROOT="${EGOWM_ROOT}/external/Track4World"
REPO_ROOT="${T4W_ROOT}"
CONDA_BASE="$(conda info --base)"
CONDA_ENV="track4world"

DATASET_ROOT="${EGOWAM_FLOW_INPUT:-/coc/flash7/scratch/egowm/egoverseS3ZarrDataset/eva_cup_saucer_1}"
OUTPUT_ROOT="${EGOWAM_FLOW_OUTPUT:-/coc/flash7/scratch/egowm/egoverseS3ZarrDataset/eva_cup_saucer_1_3d_flow_track4world}"
DATASET_NAME="eva_cup_saucer_1"
CONSTANT_HEAD_POSE_CSV="${EGOWAM_HEAD_POSE_CSV:-/coc/flash7/xyin85/Depth-Anything-3/head_poses.csv}"
CONSTANT_HEAD_POSE_ROW=0
MAX_FRAMES=""
MIN_VALID_FRAMES=30
ARRAY_PARALLELISM=32
JOB_NAME="eva-3d-flow"
DRY_RUN=0
FORCE=0

PARTITION="overcap"
ACCOUNT="overcap"
QOS=""
CONSTRAINT=""
EXCLUDE="qt-1"
GPUS=1
GPU_TYPE="a40"
CPUS=8
MEM="96G"
TIME_LIMIT="02:00:00"

CKPT_INIT="checkpoints/track4world_da3.pth"
CONFIG_PATH="track4world/config/eval/v1.json"
COORDINATE="world_depthanythingv3"
IMAGE_SIZE=640
INFERENCE_ITERS=4
RATE=16
FRAME_SKIP=45

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --constant-head-pose-csv) CONSTANT_HEAD_POSE_CSV="$2"; shift 2 ;;
    --constant-head-pose-row) CONSTANT_HEAD_POSE_ROW="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --min-valid-frames) MIN_VALID_FRAMES="$2"; shift 2 ;;
    --array-parallelism) ARRAY_PARALLELISM="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --constraint) CONSTRAINT="$2"; shift 2 ;;
    --exclude) EXCLUDE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --ckpt-init) CKPT_INIT="$2"; shift 2 ;;
    --config-path) CONFIG_PATH="$2"; shift 2 ;;
    --coordinate) COORDINATE="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --inference-iters) INFERENCE_ITERS="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --frame-skip) FRAME_SKIP="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --help)
      cat <<'EOF'
Usage:
  bash scripts/submit_bulk_eva_3d_flow_slurm.sh \
    --dataset-root /coc/flash7/scratch/egowm/egoverseS3ZarrDataset/eva_cup_saucer_1 \
    --output-root /coc/flash7/scratch/egowm/egoverseS3ZarrDataset/eva_cup_saucer_1_3d_flow_track4world \
    --constant-head-pose-csv /coc/flash7/xyin85/Depth-Anything-3/head_poses.csv \
    [--dry-run]

Builds a manifest of EVA episodes (one row per timestamp zarr) and submits
them as a Slurm array invoking scripts/compute_3d_flow_fixed_camera.py.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CONSTANT_HEAD_POSE_CSV}" ]]; then
  echo "--constant-head-pose-csv is required" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/manifests" "${OUTPUT_ROOT}/slurm"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"  # submodule root: track4world + utils3d

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_PATH="${OUTPUT_ROOT}/manifests/${JOB_NAME}_${STAMP}.jsonl"
SUMMARY_PATH="${OUTPUT_ROOT}/manifests/${JOB_NAME}_${STAMP}_summary.json"

export DATASET_ROOT OUTPUT_ROOT DATASET_NAME MIN_VALID_FRAMES MANIFEST_PATH SUMMARY_PATH
export FORCE_REPROCESS="${FORCE}"
python - <<'PY'
import json
import os
from pathlib import Path

import zarr

dataset_root = Path(os.environ["DATASET_ROOT"])
output_root = Path(os.environ["OUTPUT_ROOT"])
dataset_name = os.environ["DATASET_NAME"]
min_valid_frames = int(os.environ["MIN_VALID_FRAMES"])
manifest_path = Path(os.environ["MANIFEST_PATH"])
summary_path = Path(os.environ["SUMMARY_PATH"])
force_reprocess = os.environ.get("FORCE_REPROCESS", "0") == "1"


def is_completed(output_dir: Path, row: dict) -> bool:
    success = output_dir / "_SUCCESS"
    seg_meta = output_dir / "segment_metadata.json"
    required = [
        output_dir / "sampled_uv.npy",
        output_dir / "sampled_points_t.npy",
        output_dir / "sampled_points_tk.npy",
        output_dir / "sampled_proj_flow_2d.npy",
        output_dir / "metadata.json",
    ]
    if not success.exists() or not seg_meta.exists():
        return False
    if not all(p.exists() for p in required):
        return False
    try:
        with open(seg_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return (
        meta.get("source_zarr_path") == row["source_zarr_path"]
        and meta.get("valid_frame_count") == row["valid_frame_count"]
        and isinstance(meta.get("raw_flow_artifacts"), dict)
    )


rows: list[dict] = []
total_stores = 0
short_skipped = 0
completed_skipped = 0
empty_skipped = 0
stores = sorted(path for path in dataset_root.iterdir() if path.is_dir())

for zarr_path in stores:
    total_stores += 1
    try:
        root = zarr.open(zarr_path, mode="r")
        attrs = dict(root.attrs)
    except Exception:
        empty_skipped += 1
        continue
    total_frames = int(attrs.get("total_frames", 0))
    if total_frames <= 0:
        try:
            total_frames = int(root["images.front_1"].shape[0])
        except Exception:
            empty_skipped += 1
            continue
    if total_frames < min_valid_frames:
        short_skipped += 1
        continue

    output_dir = output_root / f"egoverseS3ZarrDataset__{dataset_name}__{zarr_path.name}"
    row = {
        "source_zarr_path": str(zarr_path),
        "valid_frame_count": int(total_frames),
        "output_dir": str(output_dir),
    }
    if not force_reprocess and is_completed(output_dir, row):
        completed_skipped += 1
        continue
    rows.append(row)

with open(manifest_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")

summary = {
    "dataset_root": str(dataset_root),
    "output_root": str(output_root),
    "dataset_name": dataset_name,
    "stores": total_stores,
    "empty_stores_skipped": empty_skipped,
    "short_stores_skipped": short_skipped,
    "completed_stores_skipped": completed_skipped,
    "rows_to_submit": len(rows),
    "min_valid_frames": min_valid_frames,
    "manifest_path": str(manifest_path),
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(json.dumps(summary, sort_keys=True))
PY

TASK_COUNT="$(wc -l < "${MANIFEST_PATH}")"
echo "Manifest: ${MANIFEST_PATH}"
echo "Summary: ${SUMMARY_PATH}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

if [[ "${TASK_COUNT}" -eq 0 ]]; then
  echo "No pending EVA 3D-flow jobs to submit."
  exit 0
fi

SBATCH_ARGS=(
  --partition="${PARTITION}"
  --job-name="${JOB_NAME}"
  --nodes=1
  --ntasks-per-node=1
  --gpus-per-node="${GPU_TYPE}:${GPUS}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
  --time="${TIME_LIMIT}"
  --array="0-$((TASK_COUNT - 1))%${ARRAY_PARALLELISM}"
  --output="${OUTPUT_ROOT}/slurm/%x-%A_%a.out"
  --error="${OUTPUT_ROOT}/slurm/%x-%A_%a.err"
  --chdir="${REPO_ROOT}"
)

if [[ -n "${ACCOUNT}" ]]; then SBATCH_ARGS+=(--account="${ACCOUNT}"); fi
if [[ -n "${QOS}" ]]; then SBATCH_ARGS+=(--qos="${QOS}"); fi
if [[ -n "${CONSTRAINT}" ]]; then SBATCH_ARGS+=(--constraint="${CONSTRAINT}"); fi
if [[ -n "${EXCLUDE}" ]]; then SBATCH_ARGS+=(--exclude="${EXCLUDE}"); fi

sbatch "${SBATCH_ARGS[@]}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"  # submodule root: track4world + utils3d

echo "hostname: \$(hostname)"
echo "SLURM_JOB_ID=\${SLURM_JOB_ID:-unset}"
echo "SLURM_ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

manifest_path = Path("${MANIFEST_PATH}")
task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
with open(manifest_path, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]
row = rows[task_index]

cmd = [
    "python", "scripts/compute_3d_flow_fixed_camera.py",
    "--zarr-path", row["source_zarr_path"],
    "--output-dir", row["output_dir"],
    "--constant-head-pose-csv", "${CONSTANT_HEAD_POSE_CSV}",
    "--constant-head-pose-row", "${CONSTANT_HEAD_POSE_ROW}",
    "--ckpt-init", "${CKPT_INIT}",
    "--config-path", "${CONFIG_PATH}",
    "--coordinate", "${COORDINATE}",
    "--image-size", "${IMAGE_SIZE}",
    "--inference-iters", "${INFERENCE_ITERS}",
    "--rate", "${RATE}",
    "--frame-skip", "${FRAME_SKIP}",
]
if "${MAX_FRAMES}":
    cmd.extend(["--max-frames", "${MAX_FRAMES}"])

subprocess.check_call(cmd)
PY
EOF
