from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import io

import numpy as np
from PIL import Image
import zarr

ARIA_EPISODE_GAP_THRESHOLD_NS = 1_000_000_000


@dataclass(frozen=True)
class AriaSegment:
    segment_index: int
    start_index: int
    end_index: int
    valid_indices: np.ndarray


@dataclass(frozen=True)
class AriaEpisode:
    zarr_path: Path
    fps: int
    segment_index: int | None
    segment_start: int | None
    segment_end: int | None
    valid_indices: np.ndarray
    timestamps_ns: np.ndarray
    head_pose: np.ndarray
    eye_gaze: np.ndarray
    image_hw: tuple[int, int]


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _all_zero_rows(arr: np.ndarray) -> np.ndarray:
    return np.all(arr == 0, axis=1)


def _unwrap_bytes(item: Any) -> bytes:
    value = item
    for _ in range(16):
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
                continue
            if value.dtype.kind in {"S", "V"} and value.size == 1:
                value = value.reshape(-1)[0]
                continue
        break

    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "V"}:
        return bytes(value.reshape(-1)[0])
    if isinstance(value, np.bytes_):
        return bytes(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"Unsupported JPEG payload type: {type(value)!r}")


def decode_jpeg_item(item: Any) -> np.ndarray:
    payload = _unwrap_bytes(item)
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    return np.asarray(image)


def _load_episode_arrays(zarr_path: Path) -> tuple[zarr.Group, np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(zarr_path, mode="r")
    timestamps_ns = np.asarray(root["obs_rgb_timestamps_ns"])
    head_pose = np.asarray(root["obs_head_pose"])
    eye_gaze = np.asarray(root["obs_eye_gaze"])
    if head_pose.ndim != 2 or head_pose.shape[1] != 7:
        raise ValueError(f"Expected obs_head_pose shape (T, 7), got {head_pose.shape}")
    if eye_gaze.ndim != 2 or eye_gaze.shape[1] != 3:
        raise ValueError(f"Expected obs_eye_gaze shape (T, 3), got {eye_gaze.shape}")
    if timestamps_ns.shape[0] != head_pose.shape[0] or eye_gaze.shape[0] != head_pose.shape[0]:
        raise ValueError("Timestamp, head pose, and eye gaze arrays must have identical frame counts")
    return root, timestamps_ns, head_pose, eye_gaze


def _build_valid_mask(timestamps_ns: np.ndarray, head_pose: np.ndarray) -> np.ndarray:
    return (timestamps_ns != 0) & (~_all_zero_rows(head_pose))


def _infer_usable_stop(valid_mask: np.ndarray, zarr_path: Path) -> int:
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        raise ValueError(f"No valid frames found in {zarr_path}")
    return int(valid_indices[-1]) + 1


def _detect_segment_bounds(
    timestamps_ns: np.ndarray,
    usable_stop: int,
    gap_threshold_ns: int,
) -> list[tuple[int, int]]:
    usable_timestamps = timestamps_ns[:usable_stop]
    if usable_timestamps.shape[0] <= 0:
        return []
    if usable_timestamps.shape[0] == 1:
        return [(0, 0)]

    cut_indices = np.flatnonzero(np.diff(usable_timestamps) > gap_threshold_ns).astype(np.int64)
    starts = [0, *[int(idx) + 1 for idx in cut_indices.tolist()]]
    ends = [*[int(idx) for idx in cut_indices.tolist()], usable_stop - 1]
    return list(zip(starts, ends, strict=True))


def _build_segments(
    timestamps_ns: np.ndarray,
    valid_mask: np.ndarray,
    usable_stop: int,
    gap_threshold_ns: int,
) -> list[AriaSegment]:
    segments: list[AriaSegment] = []
    for segment_index, (start_index, end_index) in enumerate(
        _detect_segment_bounds(timestamps_ns, usable_stop, gap_threshold_ns)
    ):
        segment_valid_indices = np.flatnonzero(valid_mask[start_index : end_index + 1]) + start_index
        segments.append(
            AriaSegment(
                segment_index=segment_index,
                start_index=start_index,
                end_index=end_index,
                valid_indices=segment_valid_indices.astype(np.int64, copy=False),
            )
        )
    return segments


def list_episode_segments(
    zarr_path: str | Path,
    min_valid_frames: int = 0,
    gap_threshold_ns: int = ARIA_EPISODE_GAP_THRESHOLD_NS,
) -> list[AriaSegment]:
    zarr_path = _to_path(zarr_path)
    _, timestamps_ns, head_pose, _ = _load_episode_arrays(zarr_path)
    valid_mask = _build_valid_mask(timestamps_ns, head_pose)
    usable_stop = _infer_usable_stop(valid_mask, zarr_path)
    segments = _build_segments(timestamps_ns, valid_mask, usable_stop, gap_threshold_ns)
    return [segment for segment in segments if segment.valid_indices.size >= int(min_valid_frames)]


def open_episode(
    zarr_path: str | Path,
    segment_index: int | None = None,
    gap_threshold_ns: int = ARIA_EPISODE_GAP_THRESHOLD_NS,
) -> AriaEpisode:
    zarr_path = _to_path(zarr_path)
    root, timestamps_ns, head_pose, eye_gaze = _load_episode_arrays(zarr_path)

    valid_mask = _build_valid_mask(timestamps_ns, head_pose)
    usable_stop = _infer_usable_stop(valid_mask, zarr_path)
    valid_indices = np.flatnonzero(valid_mask[:usable_stop]).astype(np.int64, copy=False)
    selected_segment: AriaSegment | None = None
    if segment_index is not None:
        segments = _build_segments(timestamps_ns, valid_mask, usable_stop, gap_threshold_ns)
        segment_map = {segment.segment_index: segment for segment in segments}
        if segment_index not in segment_map:
            raise ValueError(f"Segment index {segment_index} not found in {zarr_path}")
        selected_segment = segment_map[segment_index]
        valid_indices = selected_segment.valid_indices
        if valid_indices.size == 0:
            raise ValueError(
                f"Segment index {segment_index} in {zarr_path} does not contain any valid frames"
            )

    image_hw: tuple[int, int]
    features = dict(root.attrs).get("features", {})
    image_meta = features.get("images.front_1", {})
    shape = image_meta.get("shape")
    if isinstance(shape, list) and len(shape) >= 2:
        image_hw = (int(shape[0]), int(shape[1]))
    else:
        first_image = decode_jpeg_item(root["images.front_1"][int(valid_indices[0])])
        image_hw = (int(first_image.shape[0]), int(first_image.shape[1]))

    fps = int(dict(root.attrs).get("fps", 30))
    return AriaEpisode(
        zarr_path=zarr_path,
        fps=fps,
        segment_index=None if selected_segment is None else selected_segment.segment_index,
        segment_start=None if selected_segment is None else selected_segment.start_index,
        segment_end=None if selected_segment is None else selected_segment.end_index,
        valid_indices=valid_indices.astype(np.int64, copy=False),
        timestamps_ns=timestamps_ns[valid_indices].astype(np.int64, copy=False),
        head_pose=head_pose[valid_indices].astype(np.float64, copy=False),
        eye_gaze=eye_gaze[valid_indices].astype(np.float64, copy=False),
        image_hw=image_hw,
    )


def export_frames(
    zarr_path: str | Path,
    out_dir: str | Path,
    valid_indices: np.ndarray,
    max_frames: int | None = None,
) -> list[Path]:
    zarr_path = _to_path(zarr_path)
    out_dir = _to_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open(zarr_path, mode="r")
    image_array = root["images.front_1"]
    indices = np.asarray(valid_indices, dtype=np.int64)
    if max_frames is not None:
        indices = indices[: int(max_frames)]

    frame_paths: list[Path] = []
    for frame_idx, src_idx in enumerate(indices.tolist()):
        image = decode_jpeg_item(image_array[src_idx])
        frame_path = out_dir / f"{frame_idx:05d}.jpg"
        Image.fromarray(image).save(frame_path, format="JPEG", quality=95)
        frame_paths.append(frame_path)

    return frame_paths
