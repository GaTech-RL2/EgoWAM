#!/usr/bin/env bash
# Compose the minimal egoverse flow-extraction pipeline with the upstream
# vendored external/Track4World tree.
#
# Upstream external/Track4World provides the MODEL (track4world.nets / .utils /
# config). This repo's 3dflow_process/overlay/ provides the egoverse-specific
# glue that upstream lacks (track4world.data.*, track4world.pipeline.efep_export,
# track4world.utils.flow_vis) plus the two compute entry points. We symlink the
# overlay into the submodule's package tree so `import track4world.data...`
# resolves, keeping the overlay tracked HERE (single source of truth) while
# external/Track4World stays the upstream tree.
#
# Third-party deps (utils3d, the pi3 backbone) are NOT vendored — they are part
# of the upstream Track4World install (README "Install Third-Party Modules") and
# are cloned into the Track4World working tree (untracked by this repo).
#
# Run once after cloning upstream Track4World into external/Track4World (and
# again if the overlay set changes). Idempotent.
#
# NOTE: scripts import via `python scripts/compute_*.py`, which does NOT put the
# Track4World root on sys.path. The submitters export PYTHONPATH="<that root>"
# so `track4world` / `utils3d` resolve regardless of CWD.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGOWM_ROOT="$(cd "${HERE}/../../.." && pwd)"
T4W="${EGOWM_ROOT}/external/Track4World"
OVR="${HERE}/overlay/track4world"

[[ -d "${T4W}/track4world" ]] || { echo "submodule not initialised: ${T4W}" >&2; exit 1; }

# --- egoverse overlay (tracked here) -> submodule package -------------------
ln -sfnT "${OVR}/data"     "${T4W}/track4world/data"
ln -sf   "${OVR}/utils/flow_vis.py" "${T4W}/track4world/utils/flow_vis.py"

# entry compute scripts resolve as scripts/compute_*.py from the submodule CWD
ln -sf "${HERE}/scripts/compute_3d_flow_fixed_camera.py" "${T4W}/scripts/compute_3d_flow_fixed_camera.py"
ln -sf "${HERE}/scripts/compute_3d_flow_compensated.py"  "${T4W}/scripts/compute_3d_flow_compensated.py"

# --- upstream third-party install (utils3d, pi3) ----------------------------
# Per Track4World README. Not vendored; cloned into the submodule. Idempotent.
[[ -L "${T4W}/utils3d" ]] && rm -f "${T4W}/utils3d"   # drop any legacy vendored symlink
if [[ ! -e "${T4W}/utils3d/__init__.py" ]]; then
  rm -rf "${T4W}/utils3d"
  git clone --depth 1 https://github.com/jiah-cloud/utils3d.git "${T4W}/utils3d"
fi
PI3="${T4W}/track4world/nets/external/pi3"
[[ -L "${PI3}" ]] && rm -f "${PI3}"                   # drop any legacy vendored symlink
if [[ ! -e "${PI3}/models/pi3.py" ]]; then
  rm -rf "${PI3}" "${T4W}/track4world/nets/external/pi3_repo"
  git clone --no-checkout --depth 1 https://github.com/yyfz/Pi3.git \
    "${T4W}/track4world/nets/external/pi3_repo"
  ( cd "${T4W}/track4world/nets/external/pi3_repo" && git sparse-checkout set pi3 && git checkout )
  mv "${T4W}/track4world/nets/external/pi3_repo/pi3" "${PI3}"
fi

# --- model checkpoint (weights not in git) ----------------------------------
# Set TRACK4WORLD_CKPT to an existing track4world_da3.pth to symlink it in; otherwise
# place the file at ${T4W}/checkpoints/track4world_da3.pth yourself.
LOCAL_CKPT="${TRACK4WORLD_CKPT:-}"
if [[ -f "${LOCAL_CKPT}" && ! -e "${T4W}/checkpoints/track4world_da3.pth" ]]; then
  mkdir -p "${T4W}/checkpoints"
  ln -sf "${LOCAL_CKPT}" "${T4W}/checkpoints/track4world_da3.pth"
fi

echo "overlay + third-party install synced into ${T4W}"
