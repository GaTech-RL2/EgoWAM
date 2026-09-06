"""Unit tests for point_flow_loader (per-episode artifact indexing + trajectory build)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from egowam.rldb.zarr.point_flow_loader import (
    PointFlowIndex,
    build_2d_endpoint_trajectory,
    build_3d_endpoint_trajectory,
    build_threshold_mask,
)


def _write_episode(
    root: Path,
    basename: str,
    *,
    T: int,
    N: int,
    frame_skip: int,
    include_2d: bool = True,
    nan_frac: float = 0.0,
    image_shape: tuple[int, int] = (16, 20),
) -> Path:
    # Point-flow roots store episodes by basename without the .zarr suffix.
    on_disk = basename[: -len(".zarr")] if basename.endswith(".zarr") else basename
    ep = root / on_disk
    ep.mkdir(parents=True, exist_ok=True)

    H, W = image_shape
    rng = np.random.default_rng(0)
    # Random pixel anchors within bounds.
    uv_x = rng.integers(0, W, size=(N,), dtype=np.int32)
    uv_y = rng.integers(0, H, size=(N,), dtype=np.int32)
    sampled_uv = np.stack([uv_x, uv_y], axis=-1).astype(np.int32)
    np.save(ep / "sampled_uv.npy", sampled_uv)

    # 3D positions at t and t+k.
    points_t = rng.normal(size=(T, N, 3)).astype(np.float32)
    points_tk = points_t + rng.normal(scale=0.1, size=(T, N, 3)).astype(np.float32)

    if nan_frac > 0:
        nan_mask = rng.random(size=(T, N)) < nan_frac
        points_t[nan_mask] = np.nan
        points_tk[nan_mask] = np.nan

    np.save(ep / "sampled_points_t.npy", points_t)
    np.save(ep / "sampled_points_tk.npy", points_tk)

    if include_2d:
        # Compact per-anchor 2D flow (T, N, 2).
        flow2d = rng.normal(size=(T, N, 2)).astype(np.float32)
        np.save(ep / "sampled_proj_flow_2d.npy", flow2d)

    with open(ep / "metadata.json", "w") as f:
        json.dump({"frame_skip": frame_skip}, f)
    return ep


@pytest.fixture(autouse=True)
def _clear_index_cache():
    PointFlowIndex._CACHE.clear()
    yield
    PointFlowIndex._CACHE.clear()


def test_load_episode_3d_only(tmp_path: Path):
    _write_episode(tmp_path, "ep1.zarr", T=5, N=16, frame_skip=30, include_2d=False)
    ep = PointFlowIndex.load_episode(tmp_path, "ep1.zarr", load_2d=False)
    assert ep.sampled_uv.shape == (16, 2)
    assert ep.sampled_points_t.shape == (5, 16, 3)
    assert ep.sampled_points_tk.shape == (5, 16, 3)
    assert ep.sampled_flow_2d is None
    assert ep.frame_skip == 30
    assert ep.num_frames == 5


def test_load_episode_2d_requires_dense_flow(tmp_path: Path):
    _write_episode(tmp_path, "ep1.zarr", T=5, N=16, frame_skip=30, include_2d=False)
    with pytest.raises(FileNotFoundError, match="sampled_proj_flow_2d"):
        PointFlowIndex.load_episode(tmp_path, "ep1.zarr", load_2d=True)


def test_load_episode_caches_memmaps(tmp_path: Path):
    _write_episode(tmp_path, "ep1.zarr", T=5, N=16, frame_skip=30, include_2d=True)
    a = PointFlowIndex.load_episode(tmp_path, "ep1.zarr", load_2d=True)
    b = PointFlowIndex.load_episode(tmp_path, "ep1.zarr", load_2d=True)
    assert a is b


def test_build_3d_trajectory_shape_and_endpoints(tmp_path: Path):
    _write_episode(tmp_path, "ep.zarr", T=3, N=5, frame_skip=30, include_2d=False)
    ep = PointFlowIndex.load_episode(tmp_path, "ep.zarr", load_2d=False)
    traj = build_3d_endpoint_trajectory(ep, frame_idx=1)
    assert traj.shape == (30, 5, 3)
    np.testing.assert_allclose(
        traj[0], np.asarray(ep.sampled_points_t[1], dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        traj[-1], np.asarray(ep.sampled_points_tk[1], dtype=np.float32), atol=1e-6
    )


def test_build_2d_trajectory_uses_baked_flow_at_uv(tmp_path: Path):
    _write_episode(
        tmp_path,
        "ep.zarr",
        T=4,
        N=8,
        frame_skip=45,
        include_2d=True,
        image_shape=(12, 14),
    )
    ep = PointFlowIndex.load_episode(tmp_path, "ep.zarr", load_2d=True)
    frame = 2
    traj = build_2d_endpoint_trajectory(ep, frame_idx=frame)
    assert traj.shape == (45, 8, 2)
    # First step must equal the UV anchors.
    expected_start = ep.sampled_uv.astype(np.float32)
    np.testing.assert_allclose(traj[0], expected_start, atol=1e-6)
    # Last step must equal UV + baked flow.
    flow_at_uv = np.asarray(ep.sampled_flow_2d[frame], dtype=np.float32)
    expected_end = expected_start + flow_at_uv
    np.testing.assert_allclose(traj[-1], expected_end, atol=1e-6)


def test_threshold_mask_rejects_nan_and_keeps_finite(tmp_path: Path):
    _write_episode(
        tmp_path, "ep.zarr", T=3, N=100, frame_skip=30, include_2d=False, nan_frac=0.2
    )
    ep = PointFlowIndex.load_episode(tmp_path, "ep.zarr", load_2d=False)
    # threshold=0 -> only non-finite points fail the mask.
    mask = build_threshold_mask(ep, frame_idx=0, dim=3, threshold=0.0)
    assert mask.shape == (100,)
    assert mask.dtype == bool
    # Some invalid points are present due to nan_frac=0.2.
    assert mask.sum() < 100


def test_threshold_mask_rejects_stationary_points(tmp_path: Path):
    # Build an episode where all point displacements are O(0.1). A threshold
    # well above that must reject every point; a threshold of 0 must keep them.
    _write_episode(tmp_path, "ep.zarr", T=3, N=32, frame_skip=30, include_2d=False)
    ep = PointFlowIndex.load_episode(tmp_path, "ep.zarr", load_2d=False)
    kept = build_threshold_mask(ep, frame_idx=0, dim=3, threshold=0.0)
    assert kept.sum() == 32
    suppressed = build_threshold_mask(ep, frame_idx=0, dim=3, threshold=10.0)
    assert suppressed.sum() == 0


def test_threshold_mask_2d_uses_baked_flow(tmp_path: Path):
    _write_episode(
        tmp_path,
        "ep.zarr",
        T=4,
        N=16,
        frame_skip=30,
        include_2d=True,
        image_shape=(12, 14),
    )
    ep = PointFlowIndex.load_episode(tmp_path, "ep.zarr", load_2d=True)
    mask = build_threshold_mask(ep, frame_idx=0, dim=2, threshold=0.0)
    assert mask.shape == (16,)
    assert mask.dtype == bool


def _write_segmented_episode(
    root: Path,
    *,
    canonical_zarr: str,
    segment_idx: int,
    segment_start: int,
    segment_end: int,  # inclusive on disk; segments_for returns end+1 exclusive
    T: int,
    N: int = 4,
    frame_skip: int = 30,
) -> Path:
    ep = root / f"seg__{segment_idx}"
    ep.mkdir(parents=True, exist_ok=True)
    np.save(ep / "sampled_uv.npy", np.zeros((N, 2), dtype=np.int32))
    np.save(ep / "sampled_points_t.npy", np.zeros((T, N, 3), dtype=np.float32))
    np.save(ep / "sampled_points_tk.npy", np.zeros((T, N, 3), dtype=np.float32))
    with open(ep / "metadata.json", "w") as f:
        json.dump({"frame_skip": frame_skip, "zarr_path": canonical_zarr}, f)
    with open(ep / "segment_metadata.json", "w") as f:
        json.dump({"segment_start": segment_start, "segment_end": segment_end}, f)
    return ep


def test_segments_for_returns_multi_segment_bounds(tmp_path: Path):
    """Aria zarrs map to multiple point-flow segments with gaps between them.
    segments_for must expose all of them so callers can build the covered
    frame set."""
    zarr_name = "episode_xyz.zarr"
    _write_segmented_episode(
        tmp_path,
        canonical_zarr=zarr_name,
        segment_idx=0,
        segment_start=0,
        segment_end=151,
        T=152,
    )
    _write_segmented_episode(
        tmp_path,
        canonical_zarr=zarr_name,
        segment_idx=1,
        segment_start=163,
        segment_end=456,
        T=294,
    )
    PointFlowIndex._INDEX.clear()

    segs = PointFlowIndex.segments_for(tmp_path, zarr_name)
    spans = sorted((s, e) for s, e, _ in segs)
    assert spans == [(0, 152), (163, 457)]


def test_segments_for_missing_zarr_returns_empty(tmp_path: Path):
    PointFlowIndex._INDEX.clear()
    assert PointFlowIndex.segments_for(tmp_path, "nonexistent.zarr") == []
