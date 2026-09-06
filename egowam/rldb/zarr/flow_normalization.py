"""
Flow normalization utilities for the new point-conditioned 2D and 3D flow heads.

Contract: flow deltas are normalized exactly once, in the dataset loader, using
cached per-embodiment mean/std stats in flow_stats.json. Consumers at eval/viz
time explicitly call denormalize_flow() to recover pixel or world-frame units.
Flow keys are excluded from DataSchematic.normalize_data via the is_flow_key
passthrough so generic quantile normalization never touches them.

See plans/3d_point_flow_supervision.md "Flow normalization contract" and
"Flow-stats lifecycle" for the full design.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import filelock
import numpy as np
import torch

FLOW_STATS_SCHEMA_VERSION = 1


class FlowStatsStaleError(RuntimeError):
    """Raised when flow_stats.json manifest does not match the current dataset."""


# Module-level registry: preflight writes here once; transforms/heads read from here.
# Keyed by flow tensor key (e.g. "future.observations.flow_3d_points") or a
# domain-qualified alias. Single ownership; val datasets inject train stats into
# this same registry so train/val share normalization.
_FLOW_STATS_REGISTRY: dict[str, dict[str, np.ndarray]] = {}


def register_flow_stats(flow_key: str, stats: dict[str, np.ndarray]) -> None:
    _FLOW_STATS_REGISTRY[flow_key] = {
        "mean": np.asarray(stats["mean"], dtype=np.float64),
        "std": np.asarray(stats["std"], dtype=np.float64),
    }


def get_flow_stats(flow_key: str) -> dict[str, np.ndarray]:
    if flow_key not in _FLOW_STATS_REGISTRY:
        raise KeyError(
            f"flow stats for '{flow_key}' not registered; call ensure_flow_stats "
            f"preflight in trainHydra.py before dataset[0]"
        )
    return _FLOW_STATS_REGISTRY[flow_key]


def has_flow_stats(flow_key: str) -> bool:
    return flow_key in _FLOW_STATS_REGISTRY


def clear_flow_stats_registry() -> None:
    _FLOW_STATS_REGISTRY.clear()


def compute_flow_stats(
    delta_iter: Iterable[np.ndarray],
) -> dict[str, np.ndarray]:
    """Streaming NaN-aware per-channel mean/std over (any leading dims, C) arrays.

    Each yielded array is flattened over all leading dims; NaN entries are
    ignored. Returns {"mean": (C,), "std": (C,)} float64. std is floored at 1e-6
    to avoid divide-by-zero for degenerate channels.
    """
    # Two-pass Welford-like streaming: we accumulate sum, sum_sq, count per channel.
    total_sum: np.ndarray | None = None
    total_sqsum: np.ndarray | None = None
    total_count: np.ndarray | None = None

    for delta in delta_iter:
        arr = np.asarray(delta, dtype=np.float64)
        if arr.ndim == 0:
            continue
        C = arr.shape[-1]
        flat = arr.reshape(-1, C)
        mask = ~np.isnan(flat)
        valid = np.where(mask, flat, 0.0)
        count = mask.sum(axis=0).astype(np.float64)  # (C,)
        s = valid.sum(axis=0)  # (C,)
        sq = (valid * valid).sum(axis=0)  # (C,)

        if total_sum is None:
            total_sum = s
            total_sqsum = sq
            total_count = count
        else:
            total_sum += s
            total_sqsum += sq
            total_count += count

    if total_count is None or np.any(total_count == 0):
        raise ValueError(
            "compute_flow_stats got no valid (non-NaN) samples in at least one channel"
        )

    mean = total_sum / total_count
    var = total_sqsum / total_count - mean * mean
    var = np.clip(var, 0.0, None)  # numerical floor
    std = np.sqrt(var)
    std = np.maximum(std, 1e-6)

    return {"mean": mean.astype(np.float64), "std": std.astype(np.float64)}


def _to_numpy(stats: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "mean": np.asarray(stats["mean"], dtype=np.float64),
        "std": np.asarray(stats["std"], dtype=np.float64),
    }


def normalize_flow(delta, stats: dict[str, Any]):
    """(delta - mean) / std. Broadcasts over all leading dims. NumPy or torch."""
    if isinstance(delta, torch.Tensor):
        mean = torch.as_tensor(stats["mean"], dtype=delta.dtype, device=delta.device)
        std = torch.as_tensor(stats["std"], dtype=delta.dtype, device=delta.device)
        return (delta - mean) / std
    arr = np.asarray(delta)
    s = _to_numpy(stats)
    return (arr - s["mean"]) / s["std"]


def denormalize_flow(normed, stats: dict[str, Any]):
    """normed * std + mean. Inverse of normalize_flow. NumPy or torch."""
    if isinstance(normed, torch.Tensor):
        mean = torch.as_tensor(stats["mean"], dtype=normed.dtype, device=normed.device)
        std = torch.as_tensor(stats["std"], dtype=normed.dtype, device=normed.device)
        return normed * std + mean
    arr = np.asarray(normed)
    s = _to_numpy(stats)
    return arr * s["std"] + s["mean"]


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def episode_ids_digest(episode_ids: Iterable[str]) -> str:
    """SHA-1 over the sorted episode ID list. Used in the tiny manifest."""
    sorted_ids = sorted(str(e) for e in episode_ids)
    h = hashlib.sha1()
    for eid in sorted_ids:
        h.update(eid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_manifest(
    *,
    flow_chunk_length: int,
    point_flow_roots: dict[str, str],
    train_episode_ids: Iterable[str],
    movement_threshold: float = 0.0,
) -> dict[str, Any]:
    """Build the tiny manifest stored alongside flow stats.

    movement_threshold is part of the identity because below-threshold points
    are excluded from stats; changing it changes the computed mean/std.
    """
    return {
        "schema_version": FLOW_STATS_SCHEMA_VERSION,
        "FLOW_CHUNK_LENGTH": int(flow_chunk_length),
        "point_flow_roots": {
            str(k): str(v) for k, v in sorted(point_flow_roots.items())
        },
        "episode_ids_digest": episode_ids_digest(train_episode_ids),
        "movement_threshold": float(movement_threshold),
    }


def _manifest_matches(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str | None]:
    """Return (matches, first_differing_field_or_None)."""
    for key in (
        "schema_version",
        "FLOW_CHUNK_LENGTH",
        "point_flow_roots",
        "episode_ids_digest",
        "movement_threshold",
    ):
        if a.get(key) != b.get(key):
            return False, key
    return True, None


def load_flow_stats_cache(path: str | os.PathLike) -> dict[str, Any]:
    """Read flow_stats.json. Returns {'manifest': {...}, 'stats': {flow_key: {mean, std}, ...}}."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "manifest" not in data or "stats" not in data:
        raise RuntimeError(
            f"flow_stats cache at {path} is malformed: missing 'manifest' or 'stats'"
        )
    # Convert stats lists back to numpy arrays.
    for flow_key, s in data["stats"].items():
        s["mean"] = np.asarray(s["mean"], dtype=np.float64)
        s["std"] = np.asarray(s["std"], dtype=np.float64)
    return data


def save_flow_stats_cache_atomic(
    path: str | os.PathLike,
    *,
    manifest: dict[str, Any],
    stats: dict[str, dict[str, np.ndarray]],
) -> None:
    """Write flow_stats.json atomically: tmp + fsync + rename, then fsync parent dir."""
    path = Path(path)
    payload = {
        "manifest": manifest,
        "stats": {
            k: {
                "mean": np.asarray(v["mean"]).tolist(),
                "std": np.asarray(v["std"]).tolist(),
            }
            for k, v in stats.items()
        },
    }
    # Unique tmp name per-process so concurrent writers don't stomp on each other's tmp.
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    # Fsync the parent dir to persist the rename across NFS commit boundaries.
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def ensure_flow_stats(
    cache_path: str | os.PathLike,
    *,
    current_manifest: dict[str, Any],
    compute_stats_fn,
    lock_timeout_seconds: float = 7200.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Load flow stats if cache is present and manifest matches; otherwise compute+write.

    Synchronization across DDP / submitit ranks: filelock.FileLock on a sibling
    .lock file serializes the compute+write region. Any rank entering the lock
    second sees the finalized file and returns.

    compute_stats_fn: zero-arg callable returning stats dict {flow_key: {"mean", "std"}}.
    It must reproduce the loader's per-key target pipeline (native trajectory ->
    InterpolatePointTrajectory -> step-wise delta -> mask/subsample) so the cached
    stats match what __getitem__ actually feeds the head.

    Returns the stats dict (mean/std numpy arrays per flow key). Raises
    FlowStatsStaleError on manifest mismatch with a pre-existing cache.
    """
    cache_path = Path(cache_path)
    lock_path = str(cache_path) + ".lock"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: valid cache already present.
    if cache_path.exists():
        cached = load_flow_stats_cache(cache_path)
        ok, bad = _manifest_matches(cached["manifest"], current_manifest)
        if ok:
            return cached["stats"]
        raise FlowStatsStaleError(
            f"flow_stats cache at {cache_path} is stale: field {bad!r} differs "
            f"(cache={cached['manifest'].get(bad)!r}, current={current_manifest.get(bad)!r}). "
            f"Delete {cache_path} and rerun."
        )

    # Slow path: acquire lock and either load (race winner) or compute.
    lock = filelock.FileLock(lock_path, timeout=lock_timeout_seconds)
    with lock:
        if cache_path.exists():
            cached = load_flow_stats_cache(cache_path)
            ok, bad = _manifest_matches(cached["manifest"], current_manifest)
            if ok:
                return cached["stats"]
            raise FlowStatsStaleError(
                f"flow_stats cache at {cache_path} is stale: field {bad!r} differs. "
                f"Delete {cache_path} and rerun."
            )
        stats = compute_stats_fn()
        save_flow_stats_cache_atomic(cache_path, manifest=current_manifest, stats=stats)
        return stats
