#!/bin/bash
# CPU-only RoboTwin HDF5 -> EgoVerse zarr conversion for ONE (task, robot, variant) dataset.
# Usage: sbatch --job-name=<name> sbatch/robotwin/convert_dataset.sh <task> <robot> <variant>
#   e.g. sbatch sbatch/robotwin/convert_dataset.sh stack_bowls_three arx-x5 randomized_500
#SBATCH --job-name=rtconv
#SBATCH --output=sbatch_logs/%x_%j.out
#SBATCH --error=sbatch_logs/%x_%j.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos="short"
#SBATCH --exclude="dynamics"

set -uo pipefail
TASK=$1
ROBOT=$2
VAR=$3

ROOT=${EGOWAM_ROBOTWIN_ROOT:-/coc/flash7/scratch/egowm/robotwin}
DS=$ROOT/dataset
ZD=$ROOT/zarrDataset
ZIP=$DS/$TASK/${ROBOT}_${VAR}.zip
DIR=$DS/$TASK/${ROBOT}_${VAR}

cd ${EGOWAM_ROOT:-$(git rev-parse --show-toplevel)}
source emimic2/bin/activate

echo "[$(date)] host=$(hostname) task=$TASK robot=$ROBOT var=$VAR"
if [ ! -f "$ZIP" ]; then echo "MISSING zip: $ZIP"; exit 1; fi

# Unzip in place if not already extracted (idempotent; per-robot subdir -> no cross-job clash).
if [ ! -d "$DIR/data" ]; then
  echo "[$(date)] unzip $ZIP"
  ( cd "$DS/$TASK" && unzip -o -q "${ROBOT}_${VAR}.zip" )
fi

# ZarrWriter opens each episode with mode="w" -> re-running overwrites cleanly (safe to resubmit).
python -m egowam.scripts.robotwin_process.robotwin_to_zarr \
  --robot-dir "$DIR" --robot "$ROBOT" --task "$TASK" --output-root "$ZD"

N=$(ls "$ZD/$TASK/${ROBOT}_${VAR}" 2>/dev/null | grep -c '\.zarr$')
echo "[$(date)] DONE task=$TASK robot=$ROBOT var=$VAR -> $N zarr episodes"
