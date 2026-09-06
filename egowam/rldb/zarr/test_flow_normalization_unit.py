"""Unit tests for flow normalization utilities and InterpolatePointTrajectory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from egowam.rldb.zarr.action_chunk_transforms import InterpolatePointTrajectory
from egowam.rldb.zarr.flow_normalization import (
    FlowStatsStaleError,
    build_manifest,
    compute_flow_stats,
    denormalize_flow,
    ensure_flow_stats,
    episode_ids_digest,
    load_flow_stats_cache,
    normalize_flow,
    save_flow_stats_cache_atomic,
)

# ---------------------------------------------------------------------------
# Normalize / denormalize round-trip
# ---------------------------------------------------------------------------


def _make_stats(d: int) -> dict:
    rng = np.random.default_rng(0)
    return {
        "mean": rng.normal(size=(d,)).astype(np.float64),
        "std": np.abs(rng.normal(size=(d,))).astype(np.float64) + 0.1,
    }


def test_flow_normalize_denormalize_round_trip_2d():
    stats = _make_stats(2)
    delta = np.random.default_rng(1).normal(size=(3, 100, 7, 2)).astype(np.float32)
    normed = normalize_flow(delta, stats)
    recovered = denormalize_flow(normed, stats)
    np.testing.assert_allclose(recovered, delta, rtol=1e-5, atol=1e-5)


def test_flow_normalize_denormalize_round_trip_3d_torch():
    stats = _make_stats(3)
    delta = torch.randn(2, 100, 5, 3, dtype=torch.float32)
    normed = normalize_flow(delta, stats)
    recovered = denormalize_flow(normed, stats)
    torch.testing.assert_close(recovered, delta, rtol=1e-5, atol=1e-5)


def test_compute_flow_stats_ignores_nan():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=[0.5, -1.0, 2.0], scale=[0.3, 0.1, 1.5], size=(1000, 3))
    # Inject 9% NaN over channel 0 to mirror the occlusion pattern.
    nan_mask = rng.random(1000) < 0.09
    contaminated = data.copy()
    contaminated[nan_mask, 0] = np.nan
    stats = compute_flow_stats([contaminated])
    # Compare against manually-masked ground truth on channel 0.
    mean0 = np.nanmean(contaminated[:, 0])
    std0 = np.nanstd(contaminated[:, 0])
    np.testing.assert_allclose(stats["mean"][0], mean0, atol=1e-6)
    np.testing.assert_allclose(stats["std"][0], max(std0, 1e-6), atol=1e-6)


def test_compute_flow_stats_streams_multiple_chunks():
    rng = np.random.default_rng(0)
    full = rng.normal(size=(500, 2))
    chunks = [full[:200], full[200:350], full[350:]]
    stats = compute_flow_stats(chunks)
    ref_mean = full.mean(axis=0)
    ref_std = full.std(axis=0)
    np.testing.assert_allclose(stats["mean"], ref_mean, atol=1e-10)
    np.testing.assert_allclose(stats["std"], np.maximum(ref_std, 1e-6), atol=1e-10)


def test_compute_flow_stats_all_nan_channel_raises():
    data = np.full((10, 2), np.nan)
    with pytest.raises(ValueError):
        compute_flow_stats([data])


# ---------------------------------------------------------------------------
# Cache I/O + manifest freshness
# ---------------------------------------------------------------------------


def _build_sample_manifest(**overrides):
    base = dict(
        flow_chunk_length=100,
        point_flow_roots={"eva_bimanual": "/data/eva_point_flow"},
        train_episode_ids=["e1", "e2", "e3"],
    )
    base.update(overrides)
    return build_manifest(**base)


def test_save_and_load_cache_round_trip(tmp_path: Path):
    manifest = _build_sample_manifest()
    stats = {
        "eva_bimanual.future_flow_3d_points": {
            "mean": np.array([0.1, -0.2, 0.3], dtype=np.float64),
            "std": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        }
    }
    cache = tmp_path / "flow_stats.json"
    save_flow_stats_cache_atomic(cache, manifest=manifest, stats=stats)
    loaded = load_flow_stats_cache(cache)
    assert loaded["manifest"] == manifest
    np.testing.assert_array_equal(
        loaded["stats"]["eva_bimanual.future_flow_3d_points"]["mean"],
        stats["eva_bimanual.future_flow_3d_points"]["mean"],
    )


def test_ensure_flow_stats_fresh_cache_computes_and_writes(tmp_path: Path):
    manifest = _build_sample_manifest()
    computed = {
        "eva_bimanual.future_flow_3d_points": {
            "mean": np.zeros(3),
            "std": np.ones(3),
        }
    }
    cache = tmp_path / "flow_stats.json"
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return computed

    stats = ensure_flow_stats(
        cache, current_manifest=manifest, compute_stats_fn=compute
    )
    assert calls["n"] == 1
    assert "eva_bimanual.future_flow_3d_points" in stats
    assert cache.exists()

    # Second call reuses the cache without invoking compute.
    stats2 = ensure_flow_stats(
        cache, current_manifest=manifest, compute_stats_fn=compute
    )
    assert calls["n"] == 1
    np.testing.assert_array_equal(
        stats2["eva_bimanual.future_flow_3d_points"]["std"],
        computed["eva_bimanual.future_flow_3d_points"]["std"],
    )


def test_ensure_flow_stats_manifest_mismatch_raises(tmp_path: Path):
    cache = tmp_path / "flow_stats.json"
    stats = {"k": {"mean": np.zeros(2), "std": np.ones(2)}}
    save_flow_stats_cache_atomic(cache, manifest=_build_sample_manifest(), stats=stats)
    other = _build_sample_manifest(flow_chunk_length=200)
    with pytest.raises(FlowStatsStaleError, match="FLOW_CHUNK_LENGTH"):
        ensure_flow_stats(
            cache,
            current_manifest=other,
            compute_stats_fn=lambda: stats,
        )


def test_episode_ids_digest_order_invariant():
    a = episode_ids_digest(["e1", "e3", "e2"])
    b = episode_ids_digest(["e3", "e2", "e1"])
    assert a == b


# ---------------------------------------------------------------------------
# InterpolatePointTrajectory
# ---------------------------------------------------------------------------


def _linear_raw_chunk(P_t: np.ndarray, P_tk: np.ndarray, k: int) -> np.ndarray:
    """Build (k, N, D) linear-interp endpoints trajectory."""
    steps = np.linspace(0.0, 1.0, k)[:, None, None]
    return P_t[None] + steps * (P_tk - P_t)[None]


def test_interpolate_point_trajectory_shape_eva():
    N, D = 512, 3
    rng = np.random.default_rng(0)
    P_t = rng.normal(size=(N, D))
    P_tk = rng.normal(size=(N, D))
    raw = _linear_raw_chunk(P_t, P_tk, k=30)
    t = InterpolatePointTrajectory(
        new_chunk_length=100, action_key="x", output_action_key="x", stride=1
    )
    out = t.transform({"x": raw})["x"]
    assert out.shape == (100, N, D)


def test_interpolate_point_trajectory_aria_endpoint_preserved():
    """Round-13 regression: ensure P_tk survives the k=45 -> 100 resample with stride=1."""
    N, D = 8, 3
    rng = np.random.default_rng(1)
    P_t = rng.normal(size=(N, D))
    P_tk = rng.normal(size=(N, D))
    raw = _linear_raw_chunk(P_t, P_tk, k=45)
    t = InterpolatePointTrajectory(
        new_chunk_length=100, action_key="x", output_action_key="x", stride=1
    )
    out = t.transform({"x": raw})["x"]
    # The last output step must equal P_tk exactly (interp endpoint, stride=1).
    np.testing.assert_allclose(out[-1], P_tk, atol=1e-10)
    np.testing.assert_allclose(out[0], P_t, atol=1e-10)


def test_interpolate_point_trajectory_stride_drops_endpoint_if_gt1():
    """Document that stride>1 on k=45 would drop P_tk. Flow MUST use stride=1."""
    N, D = 4, 3
    rng = np.random.default_rng(2)
    P_t = rng.normal(size=(N, D))
    P_tk = rng.normal(size=(N, D))
    raw = _linear_raw_chunk(P_t, P_tk, k=45)
    t = InterpolatePointTrajectory(
        new_chunk_length=100, action_key="x", output_action_key="x", stride=3
    )
    out = t.transform({"x": raw})["x"]
    # With stride=3 the subsequence is indices 0,3,...,42 -> last is index 42,
    # NOT the P_tk at index 44. The output last step therefore differs from P_tk.
    assert not np.allclose(out[-1], P_tk, atol=1e-4)


def test_interpolate_point_trajectory_rejects_rank_2():
    t = InterpolatePointTrajectory(
        new_chunk_length=100, action_key="x", output_action_key="x", stride=1
    )
    with pytest.raises(ValueError, match=r"T, N, D"):
        t.transform({"x": np.zeros((30, 3))})
