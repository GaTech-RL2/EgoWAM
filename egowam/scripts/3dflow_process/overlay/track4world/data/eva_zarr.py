"""EVA zarr loader for fixed-camera datasets.

EVA episodes store RGB under ``images.front_1`` but do not record per-frame
head pose, eye gaze, or RGB timestamps. The scene camera is fixed, so we
broadcast a single constant head pose (conventionally row 0 of an Aria MPS
trajectory CSV) across every frame; this yields an identity relative transform
between any two frames, making downstream camera-compensation code run
unchanged.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from track4world.data.aria_zarr import decode_jpeg_item


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


@dataclass(frozen=True)
class EvaEpisode:
    zarr_path: Path
    fps: int
    total_frames: int
    valid_indices: np.ndarray
    head_pose: np.ndarray
    image_hw: tuple[int, int]


def load_eva_head_pose_csv(csv_path: str | Path, row: int = 0) -> np.ndarray:
    """Read one row of an Aria MPS trajectory CSV and return it in wxyz quat order.

    The CSV header is ``tracking_timestamp_us, utc_timestamp_ns, tx, ty, tz,
    qx, qy, qz, qw``. ``track4world.data.aria_pose.build_head_only_c2w`` expects
    quaternion entries in ``[qw, qx, qy, qz]`` order, matching Aria's
    ``obs_head_pose`` layout. Returns shape ``(7,)`` float64 as
    ``[tx, ty, tz, qw, qx, qy, qz]``.
    """
    csv_path = _to_path(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if row < 0 or row >= len(rows):
        raise IndexError(f"row {row} out of range for {csv_path} ({len(rows)} rows)")
    cols = {name: idx for idx, name in enumerate(header)}
    required = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
    missing = [name for name in required if name not in cols]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    r = rows[row]
    tx = float(r[cols["tx"]])
    ty = float(r[cols["ty"]])
    tz = float(r[cols["tz"]])
    qx = float(r[cols["qx"]])
    qy = float(r[cols["qy"]])
    qz = float(r[cols["qz"]])
    qw = float(r[cols["qw"]])
    pose = np.array([tx, ty, tz, qw, qx, qy, qz], dtype=np.float64)
    quat_norm = float(np.linalg.norm(pose[3:7]))
    if not np.all(np.isfinite(pose)) or quat_norm < 1e-6:
        raise ValueError(f"degenerate head pose in row {row} of {csv_path}")
    return pose


def open_eva_episode(
    zarr_path: str | Path,
    constant_head_pose: np.ndarray,
) -> EvaEpisode:
    """Open an EVA zarr and broadcast ``constant_head_pose`` across all frames."""
    zarr_path = _to_path(zarr_path)
    constant_head_pose = np.asarray(constant_head_pose, dtype=np.float64)
    if constant_head_pose.shape != (7,):
        raise ValueError(
            f"constant_head_pose must have shape (7,), got {constant_head_pose.shape}"
        )

    root = zarr.open(zarr_path, mode="r")
    attrs = dict(root.attrs)
    total_frames = int(attrs.get("total_frames", 0))
    if total_frames <= 0:
        image_array = root["images.front_1"]
        total_frames = int(image_array.shape[0])
    if total_frames <= 0:
        raise ValueError(f"{zarr_path} reports zero frames")
    fps = int(attrs.get("fps", 30))

    features = attrs.get("features", {})
    image_meta = features.get("images.front_1", {})
    shape = image_meta.get("shape")
    if isinstance(shape, list) and len(shape) >= 2:
        image_hw = (int(shape[0]), int(shape[1]))
    else:
        first = decode_jpeg_item(root["images.front_1"][0])
        image_hw = (int(first.shape[0]), int(first.shape[1]))

    head_pose = np.broadcast_to(
        constant_head_pose, (total_frames, 7)
    ).astype(np.float64, copy=True)
    valid_indices = np.arange(total_frames, dtype=np.int64)

    return EvaEpisode(
        zarr_path=zarr_path,
        fps=fps,
        total_frames=total_frames,
        valid_indices=valid_indices,
        head_pose=head_pose,
        image_hw=image_hw,
    )
