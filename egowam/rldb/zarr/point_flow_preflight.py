"""
Preflight helper: compute or load point-flow normalization stats BEFORE
trainHydra calls dataset[0] for shape inference.

Design (plan "Flow-stats lifecycle"):
  1. For each train dataset that has any point_flow keys in its key_map, build
     a manifest (FLOW_CHUNK_LENGTH, point_flow_roots, episode_ids_digest).
  2. Call ensure_flow_stats(cache_path, manifest, compute_fn). The compute_fn
     reproduces the loader's per-key pipeline (native-k endpoints -> interp to
     FLOW_CHUNK_LENGTH -> delta relative to step 0) and returns per-channel
     mean/std across sampled points/steps/episodes.
  3. Register the resulting stats in flow_normalization's module-level registry
     under per-embodiment stats_keys so NormalizeFlowDelta transforms can read
     them from either train or val datasets without re-scanning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from egowam.rldb.zarr.flow_normalization import (
    build_manifest,
    compute_flow_stats,
    ensure_flow_stats,
    register_flow_stats,
)
from egowam.rldb.zarr.point_flow_loader import (
    POINT_FLOW_SOURCE,
    PointFlowIndex,
    build_2d_endpoint_trajectory,
    build_3d_endpoint_trajectory,
    build_threshold_mask,
)
from egowam.utils.pose_utils import _interpolate_linear

logger = logging.getLogger(__name__)

FLOW_CHUNK_LENGTH = 100  # mirrors egowam.rldb.embodiment.eva.FLOW_CHUNK_LENGTH


def _resample_native_to_chunk(
    traj_native: np.ndarray, *, new_chunk_length: int
) -> np.ndarray:
    """Resample (k, N, D) linearly to (new_chunk_length, N, D). Matches
    InterpolatePointTrajectory (stride=1 implied)."""
    k, N, D = traj_native.shape
    flat = traj_native.reshape(k, N * D)
    out_flat = _interpolate_linear(flat, new_chunk_length)
    return out_flat.reshape(new_chunk_length, N, D)


def _collect_point_flow_specs(key_map: dict) -> list[tuple[str, dict]]:
    """Return [(flow_key, spec), ...] for point-flow keys in a dataset's key_map."""
    out = []
    for k, spec in key_map.items():
        if isinstance(spec, dict) and spec.get("source") == POINT_FLOW_SOURCE:
            out.append((k, spec))
    return out


def _episode_iter(dataset) -> Iterable[Path]:
    """Yield episode paths for a ZarrDataset (single-episode) or a MultiDataset."""
    # Single-episode ZarrDataset exposes .episode_path.
    if hasattr(dataset, "episode_path"):
        yield Path(dataset.episode_path)
        return
    # MultiDataset exposes .datasets (dict: episode_id -> ZarrDataset).
    if hasattr(dataset, "datasets"):
        ds_collection = dataset.datasets
        members = (
            ds_collection.values()
            if hasattr(ds_collection, "values")
            else ds_collection
        )
        for ds in members:
            if hasattr(ds, "episode_path"):
                yield Path(ds.episode_path)
        return
    raise TypeError(
        f"dataset type {type(dataset)} does not expose episode_path or datasets"
    )


def _build_compute_stats_fn(
    *,
    stats_registry_key: str,
    episode_paths: list[Path],
    point_flow_root: str,
    point_dim: int,
    movement_threshold: float,
    flow_chunk_length: int,
    sample_frac: float = 1.0,
    rng_seed: int = 0,
):
    """Return a callable that, when invoked, reproduces the loader pipeline on
    each episode and returns a {flow_key: {mean, std}} dict for the cache."""

    def compute():
        rng = np.random.default_rng(rng_seed)

        # Enumerate all segment dirs under the root once. Each training-set
        # episode's zarr_path may map to many segments (Aria) or one (EVA).
        # We load/scan every segment so stats are computed on the entire
        # training distribution of point-flow data under this root.
        PointFlowIndex._build_index(Path(point_flow_root).expanduser().resolve())
        idx = PointFlowIndex._INDEX[Path(point_flow_root).expanduser().resolve()]
        # Dedupe segment dirs (index stores entries under multiple keys).
        segment_dirs: list[Path] = []
        seen_dirs: set[Path] = set()
        for segs in idx.values():
            for _start, _end, ep_dir in segs:
                if ep_dir not in seen_dirs:
                    seen_dirs.add(ep_dir)
                    segment_dirs.append(ep_dir)

        def delta_iter():
            for ep_dir in segment_dirs:
                try:
                    episode = PointFlowIndex._open_episode(
                        ep_dir, load_2d=(point_dim == 2)
                    )
                except (FileNotFoundError, ValueError) as e:
                    logger.warning(f"[flow preflight] skipping {ep_dir}: {e}")
                    continue
                T = episode.num_frames  # local frame count within this segment
                n_frames = T if sample_frac >= 1.0 else max(1, int(T * sample_frac))
                local_idxs = (
                    np.arange(T)
                    if sample_frac >= 1.0
                    else rng.choice(T, size=n_frames, replace=False)
                )
                # Builders take zarr-absolute frame_idx and translate to
                # segment-local internally via _local_idx. episode.start_frame
                # is absolute; add the local offset.
                for li in local_idxs:
                    zarr_idx = int(li) + episode.start_frame
                    validity = build_threshold_mask(
                        episode,
                        zarr_idx,
                        dim=point_dim,
                        threshold=movement_threshold,
                    )
                    if not validity.any():
                        continue
                    if point_dim == 3:
                        traj = build_3d_endpoint_trajectory(episode, zarr_idx)
                    else:
                        traj = build_2d_endpoint_trajectory(episode, zarr_idx)
                    # Match loader: keep every anchor; apply validity so
                    # occluded points don't pollute per-channel stats.
                    traj = traj.astype(np.float32, copy=False)
                    traj = traj[:, validity, :]
                    resampled = _resample_native_to_chunk(
                        traj, new_chunk_length=flow_chunk_length
                    )
                    delta = resampled - resampled[0:1]  # (H, N_valid, D)
                    yield delta

        mean_std = compute_flow_stats(delta_iter())
        return {stats_registry_key: mean_std}

    return compute


def run_flow_stats_preflight(
    *,
    train_datasets: dict,
    cache_dir: str | Path,
    sample_frac: float = 1.0,
    flow_chunk_length: int = FLOW_CHUNK_LENGTH,
) -> None:
    """For each train dataset containing point-flow keys, ensure stats exist and
    register them in the flow_normalization module-level registry.

    Must be called BEFORE dataset[0] shape inference so InterpolatePointTrajectory
    / NormalizeFlowDelta can run without hitting a 'stats not registered' error.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, ds in train_datasets.items():
        # Reach a representative episode's key_map to discover point-flow specs.
        # MultiDataset.datasets is a dict keyed by episode id; ZarrDataset has
        # key_map directly.
        key_map = getattr(ds, "key_map", None)
        if key_map is None and hasattr(ds, "datasets") and ds.datasets:
            coll = ds.datasets
            first = next(iter(coll.values())) if hasattr(coll, "values") else coll[0]
            key_map = getattr(first, "key_map", None)
        if key_map is None:
            logger.warning(
                f"[flow preflight] {dataset_name}: no key_map exposed; skipping"
            )
            continue

        specs = _collect_point_flow_specs(key_map)
        if not specs:
            continue

        episode_paths = list(_episode_iter(ds))
        if not episode_paths:
            logger.warning(
                f"[flow preflight] {dataset_name}: no episodes found; skipping"
            )
            continue

        for flow_key, spec in specs:
            point_flow_root = spec["point_flow_root"]
            point_dim = int(spec["point_dim"])
            movement_threshold = float(spec.get("movement_threshold", 0.0))
            stats_registry_key = f"{dataset_name}.{flow_key.split('.')[-1]}"
            # e.g. "eva_bimanual.future_flow_3d_points"

            manifest = build_manifest(
                flow_chunk_length=flow_chunk_length,
                point_flow_roots={dataset_name: point_flow_root},
                train_episode_ids=[str(p) for p in episode_paths],
                movement_threshold=movement_threshold,
            )
            cache_path = (
                cache_dir / f"flow_stats__{stats_registry_key.replace('.', '__')}.json"
            )

            compute_fn = _build_compute_stats_fn(
                stats_registry_key=stats_registry_key,
                episode_paths=episode_paths,
                point_flow_root=point_flow_root,
                point_dim=point_dim,
                movement_threshold=movement_threshold,
                flow_chunk_length=flow_chunk_length,
                sample_frac=sample_frac,
            )
            logger.info(
                f"[flow preflight] computing/loading stats for "
                f"{stats_registry_key} (cache={cache_path})"
            )
            stats = ensure_flow_stats(
                cache_path,
                current_manifest=manifest,
                compute_stats_fn=compute_fn,
            )
            # ensure_flow_stats returns {flow_key: {mean, std}, ...} consistently.
            mean_std = stats.get(stats_registry_key) or next(iter(stats.values()))
            register_flow_stats(stats_registry_key, mean_std)
            logger.info(
                f"[flow preflight] registered {stats_registry_key}: "
                f"mean={np.asarray(mean_std['mean']).tolist()[:3]}... "
                f"std={np.asarray(mean_std['std']).tolist()[:3]}..."
            )


def inject_train_stats_into_val(
    *,
    train_datasets: dict,
    valid_datasets: dict,
) -> None:
    """No-op by design. Stats live in a process-wide registry; val datasets hit
    the same module-level _FLOW_STATS_REGISTRY via NormalizeFlowDelta at batch
    time. This function exists as a documented hook if future per-dataset
    injection semantics are needed.
    """
    return None
