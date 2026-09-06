"""
ZarrDataset implementation for EgoVerse.

Mirrors the LeRobotDataset API while reading data from Zarr arrays
instead of parquet/HF datasets.

Directory structure (per-episode metadata):
    dataset_root/
    └── episode_{ep_idx}.zarr/
        ├── observations.images.{cam}  (JPEG compressed)
        ├── observations.state
        ├── actions_joints
        └── ...

Each episode is self-contained with its own metadata, enabling:
- Independent episode uploads to S3
- Parallel processing without global coordination
- Easy episode-level data management
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import simplejpeg
import torch
import zarr

# from action_chunk_transforms import Transform
from egowam.rldb.embodiment.embodiment import EMBODIMENT
from egowam.rldb.zarr.point_flow_loader import (
    POINT_FLOW_SOURCE,
    PointFlowIndex,
    build_2d_endpoint_trajectory,
    build_3d_endpoint_trajectory,
    build_threshold_mask,
)
from egowam.utils.aws.aws_data_utils import load_env
from egowam.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
)
from egowam.utils.egowamUtils import INTRINSICS
from egowam.utils.pose_utils import _xyzwxyz_to_matrix

logger = logging.getLogger(__name__)

SEED = 42
ARIA_TIMESTAMP_KEY = "obs_rgb_timestamps_ns"
ARIA_EPISODE_GAP_THRESHOLD_NS = 1_000_000_000
ARIA_EXTERNAL_FLOW_SOURCE = "aria_external_flow"
EVA_EXTERNAL_FLOW_SOURCE = "eva_external_flow"
EXTERNAL_FLOW_SOURCES = frozenset({ARIA_EXTERNAL_FLOW_SOURCE, EVA_EXTERNAL_FLOW_SOURCE})

# Roots for the new point-conditioned flow datasets (Aria 3D flow tracks, EVA ditto).
# Default dataset root; override with EGOWAM_DATA_ROOT (mirrors paths.data_root in the
# Hydra configs) so a new machine is configured by environment rather than by editing code.
_DATA_ROOT = os.environ.get(
    "EGOWAM_DATA_ROOT", "/coc/flash7/scratch/egowm/egoverseS3ZarrDataset"
)
ARIA_POINT_FLOW_SOURCE = f"{_DATA_ROOT}/aria_cup_saucer_3d_flow_track4world"
EVA_POINT_FLOW_SOURCE = f"{_DATA_ROOT}/eva_cup_saucer_1_3d_flow_track4world"

# Key prefixes used by the new point-flow heads. Kept as package-level constants so
# embodiment DataSchematic entries, the loader, and hpt_wm.py stay in sync.
POINT_FLOW_2D_KEY_SUFFIX = "future_flow_2d_points"
POINT_FLOW_3D_KEY_SUFFIX = "future_flow_3d_points"
ARIA_FLOW_INTRINSICS_KEY = "base"
ARIA_FLOW_IMAGE_SHAPE = (448, 640)
ARIA_BASE_IMAGE_SHAPE = (480, 640)
ARIA_FLOW_DOWNSAMPLE_STRIDE = 16
FLOW_QUERY_GRID_TOLERANCE_PX = 1.0

# EVA episodes longer than this are excluded (avoids huge trajectories).
EVA_MAX_FRAMES_PER_EPISODE = 1000000
EVA_EMBODIMENT_IDS = frozenset(
    (
        EMBODIMENT.EVA_RIGHT_ARM.value,
        EMBODIMENT.EVA_LEFT_ARM.value,
        EMBODIMENT.EVA_BIMANUAL.value,
    )
)


def _is_eva_embodiment(embodiment) -> bool:
    if isinstance(embodiment, (int, np.integer)):
        return int(embodiment) in EVA_EMBODIMENT_IDS
    return str(embodiment).lower().startswith("eva")


def _episode_basename(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(".zarr"):
        return name[: -len(".zarr")]
    return name


def _resolve_metadata_artifact_path(
    metadata_path: Path,
    artifact_path: str | Path | None,
    *,
    default: Path,
) -> Path:
    if artifact_path is None:
        return default
    resolved = Path(artifact_path)
    if not resolved.is_absolute():
        resolved = metadata_path.parent / resolved
    return resolved


def _scaled_intrinsics_for_image(
    intrinsics_key: str,
    target_shape: tuple[int, int],
    base_shape: tuple[int, int] = ARIA_BASE_IMAGE_SHAPE,
) -> np.ndarray:
    height, width = target_shape
    base_height, base_width = base_shape
    scale_x = float(width) / float(base_width)
    scale_y = float(height) / float(base_height)

    k = np.asarray(INTRINSICS[intrinsics_key], dtype=np.float64)[:, :3].copy()
    k[0, 0] *= scale_x
    k[0, 2] *= scale_x
    k[1, 1] *= scale_y
    k[1, 2] *= scale_y
    return k


def _camera_rotation_flow_from_head_poses(
    current_head_pose: np.ndarray,
    future_head_pose: np.ndarray,
    *,
    image_shape: tuple[int, int] = ARIA_FLOW_IMAGE_SHAPE,
    intrinsics_key: str = ARIA_FLOW_INTRINSICS_KEY,
) -> np.ndarray:
    """Estimate image flow caused by camera rotation only.

    This intentionally ignores camera translation. Residual flow over static
    regions is therefore expected whenever head translation/parallax is
    non-negligible.
    """
    head_poses = np.asarray([current_head_pose, future_head_pose], dtype=np.float64)
    head_pose_mats = _xyzwxyz_to_matrix(head_poses)
    current_world_from_cam = head_pose_mats[0, :3, :3]
    future_world_from_cam = head_pose_mats[1, :3, :3]
    future_from_current = future_world_from_cam.T @ current_world_from_cam

    height, width = image_shape
    k = _scaled_intrinsics_for_image(intrinsics_key, image_shape)
    k_inv = np.linalg.inv(k)
    image_transform = k @ future_from_current @ k_inv

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    homog = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=0).reshape(3, -1)
    warped = image_transform @ homog
    warped_xy = warped[:2] / np.clip(warped[2:3], a_min=1e-8, a_max=None)
    flow = warped_xy.reshape(2, height, width)
    flow[0] -= grid_x
    flow[1] -= grid_y
    return flow.astype(np.float32, copy=False)


def _canonical_flow_query_grid(
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    return np.stack([grid_x, grid_y], axis=0)


def _validate_flow_query_grid(
    anchor_points: np.ndarray,
    *,
    trajectory_path: Path,
    tolerance_px: float = FLOW_QUERY_GRID_TOLERANCE_PX,
) -> None:
    if anchor_points.ndim != 3 or anchor_points.shape[0] != 2:
        raise ValueError(
            f"Expected anchor query points with shape (2, H, W), got {anchor_points.shape}"
        )
    if tolerance_px < 0:
        raise ValueError(f"tolerance_px must be non-negative, got {tolerance_px}")

    canonical = _canonical_flow_query_grid(tuple(anchor_points.shape[1:]))
    max_abs_error = float(np.max(np.abs(anchor_points - canonical)))
    if max_abs_error > tolerance_px:
        raise ValueError(
            "External flow query grid does not match the canonical pixel lattice for "
            f"{trajectory_path}. max_abs_error={max_abs_error:.4f}px "
            f"(tolerance={tolerance_px:.4f}px)"
        )


def _resolve_external_flow_future_step(
    valid_frame_indices: Iterable[int],
    *,
    anchor_frame_idx: int,
    future_offset: int,
    segment_start: int,
    segment_end: int,
) -> tuple[int, int]:
    if future_offset < 0:
        raise ValueError(f"future_offset must be non-negative, got {future_offset}")

    valid = np.asarray(tuple(int(idx) for idx in valid_frame_indices), dtype=np.int64)
    if valid.size == 0:
        raise IndexError("External flow segment has no valid frame indices.")

    target_frame_idx = min(
        max(anchor_frame_idx + future_offset, segment_start), segment_end
    )
    candidate_positions = np.flatnonzero(
        (valid >= int(anchor_frame_idx)) & (valid <= int(target_frame_idx))
    )
    if candidate_positions.size == 0:
        raise IndexError(
            "External flow segment does not contain any valid frame between "
            f"anchor={anchor_frame_idx} and clamped_target={target_frame_idx}."
        )

    future_step = int(candidate_positions[-1])
    future_frame_idx = int(valid[future_step])
    return future_step, future_frame_idx


def _uniform_subsample_flow(
    flow: np.ndarray,
    *,
    stride: int = ARIA_FLOW_DOWNSAMPLE_STRIDE,
) -> np.ndarray:
    if stride <= 0:
        raise ValueError(f"downsample_stride must be positive, got {stride}")
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"Expected flow with shape (2, H, W), got {flow.shape}")
    return np.ascontiguousarray(flow[:, ::stride, ::stride])


def _normalize_flow_by_image_resolution(
    flow: np.ndarray,
    *,
    temporal_span: int = 1,
) -> np.ndarray:
    """Convert pixel flow to per-axis normalized flow per actual raw-frame step."""
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"Expected flow with shape (2, H, W), got {flow.shape}")
    if temporal_span <= 0:
        raise ValueError(f"temporal_span must be positive, got {temporal_span}")

    _, height, width = flow.shape
    normalized = np.asarray(flow, dtype=np.float32).copy()
    scale = float(temporal_span)
    normalized[0] /= float(width) * scale
    normalized[1] /= float(height) * scale
    return normalized


@dataclass(frozen=True)
class AriaFlowSegment:
    source_episode: str
    segment_start: int
    segment_end: int
    valid_frame_indices: tuple[int, ...]
    trajectory_path: Path

    @property
    def anchor_frame_index(self) -> int:
        if len(self.valid_frame_indices) == 0:
            return int(self.segment_start)
        return int(self.valid_frame_indices[0])

    @property
    def valid_frame_count(self) -> int:
        return len(self.valid_frame_indices)


class AriaFlowArtifactIndex:
    _CACHE: dict[Path, "AriaFlowArtifactIndex"] = {}

    def __init__(self, flow_root: Path):
        self.flow_root = Path(flow_root)
        self._segments_by_source: dict[str, dict[int, AriaFlowSegment]] = {}
        self._load()

    @classmethod
    def from_root(cls, flow_root: str | Path) -> "AriaFlowArtifactIndex":
        root = Path(flow_root).expanduser().resolve()
        cached = cls._CACHE.get(root)
        if cached is None:
            cached = cls(root)
            cls._CACHE[root] = cached
        return cached

    def segments_for_source(self, source_episode: str) -> dict[int, AriaFlowSegment]:
        return self._segments_by_source.get(source_episode, {})

    def _load(self) -> None:
        if not self.flow_root.is_dir():
            raise FileNotFoundError(
                f"Aria flow root does not exist or is not a directory: {self.flow_root}"
            )

        for meta_path in sorted(self.flow_root.glob("*/segment_metadata.json")):
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            raw_artifacts = metadata.get("raw_flow_artifacts", {})
            trajectory_path = _resolve_metadata_artifact_path(
                meta_path,
                raw_artifacts.get("trajectory_path"),
                default=meta_path.parent / "2d_output" / "trajectory_2d_query0.npy",
            )
            if not trajectory_path.is_file():
                logger.warning(
                    "Skipping flow segment %s because trajectory file is missing.",
                    meta_path.parent,
                )
                continue

            valid_frame_indices = metadata.get("valid_frame_indices")
            if valid_frame_indices is None:
                valid_frame_indices = list(
                    range(
                        int(metadata["segment_start"]),
                        int(metadata["segment_end"]) + 1,
                    )
                )

            segment = AriaFlowSegment(
                source_episode=_episode_basename(metadata["source_zarr_path"]),
                segment_start=int(metadata["segment_start"]),
                segment_end=int(metadata["segment_end"]),
                valid_frame_indices=tuple(int(idx) for idx in valid_frame_indices),
                trajectory_path=trajectory_path,
            )
            anchors = segment.valid_frame_indices
            if not anchors:
                anchors = (segment.anchor_frame_index,)
            source_segments = self._segments_by_source.setdefault(
                segment.source_episode, {}
            )
            for anchor_idx in anchors:
                source_segments[int(anchor_idx)] = segment


def split_dataset_names(dataset_names, valid_ratio=0.2, seed=SEED):
    """
    Split a list of dataset names into train/valid sets.
    Args:
        dataset_names (Iterable[str])
        valid_ratio (float): fraction of datasets to put in valid.
        seed (int): for deterministic shuffling.


    Returns:
        train_set (set[str]), valid_set (set[str])
    """
    names = sorted(dataset_names)
    if not names:
        return set(), set()

    rng = random.Random(seed)
    rng.shuffle(names)

    if not (0.0 <= valid_ratio <= 1.0):
        raise ValueError(f"valid_ratio must be in [0,1], got {valid_ratio}")

    n_valid = int(len(names) * valid_ratio)
    if valid_ratio > 0.0:
        n_valid = max(1, n_valid)

    valid = set(names[:n_valid])
    train = set(names[n_valid:])
    return train, valid


# Legacy `robot_name` filter values -> live `app.episodes` columns. The table dropped `robot_name`
# (along with processed_path / processing_error / mp4_path) and gained `rig_name`, so data configs
# written against the old schema must be translated before the select.
#
# `aria_bimanual` is NOT an embodiment value: Aria episodes are `human_bimanual` recorded on the
# `aria_gen1` rig. The rig_name clause is load-bearing -- without it, task=cup_on_saucer +
# embodiment=human_bimanual matches 5963 episodes (5409 of them Mecka rigs) instead of 552.
ROBOT_NAME_TO_COLUMNS = {
    "aria_bimanual": {"embodiment": "human_bimanual", "rig_name": "aria_gen1"},
}


def remap_legacy_filters(filters: dict, columns) -> dict:
    """Translate pre-schema-change filter keys onto the live episode-table columns.

    A no-op when the table still has a `robot_name` column, so this stays correct if the schema is
    ever rolled back. Robot embodiments (e.g. ``eva_bimanual``) kept their name as an ``embodiment``
    value and need only the key renamed.
    """
    filters = dict(filters)
    if "robot_name" not in filters or "robot_name" in columns:
        return filters

    value = filters.pop("robot_name")
    mapped = ROBOT_NAME_TO_COLUMNS.get(value, {"embodiment": value})
    for key, mapped_value in mapped.items():
        if key not in columns:
            raise ValueError(
                f"Cannot map robot_name={value!r}: the episode table has no {key!r} column. "
                f"Available columns: {sorted(columns)}"
            )
        filters.setdefault(key, mapped_value)
    return filters


class EpisodeResolver:
    """
    Base class for episode resolution utilities.
    Provides shared static/class helpers; subclasses implement resolve().
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        dataset_options: dict | None = None,
    ):
        self.folder_path = Path(folder_path)
        self.key_map = key_map
        self.transform_list = transform_list
        self.dataset_options = (
            dict(dataset_options) if dataset_options is not None else {}
        )

    def _load_zarr_datasets(self, search_path: Path, valid_folder_names: set[str]):
        """
        Loads multiple Zarr datasets from the specified folder path, filtering only those whose hashes
        are present in the valid_folder_names set.

        Args:
            search_path (Path): The root directory to search for Zarr datasets.
            valid_folder_names (set[str]): A set of valid folder names (episode hashes without ".zarr") to filter datasets.
        Returns:
            dict[str, ZarrDataset]: a dictionary mapping string keys to constructed zarr datasets from valid filters.
        """
        all_paths = sorted(search_path.iterdir())
        datasets: dict[str, ZarrDataset] = {}
        skipped: list[str] = []
        for p in all_paths:
            if not p.is_dir():
                logger.info(f"{p} is not a valid directory")
                skipped.append(p.name)
                continue
            name = p.name
            if name.endswith(".zarr"):
                name = name[: -len(".zarr")]
            if name not in valid_folder_names:
                skipped.append(p.name)
                continue
            try:
                ds_obj = ZarrDataset(
                    p,
                    key_map=self.key_map,
                    transform_list=self.transform_list,
                    **self.dataset_options,
                )
                # Skip EVA episodes longer than 1100 frames
                if _is_eva_embodiment(ds_obj.embodiment) and (
                    ds_obj.total_frames > EVA_MAX_FRAMES_PER_EPISODE
                ):
                    logger.info(
                        "Skipping EVA episode %s: %d frames > %d",
                        name,
                        ds_obj.total_frames,
                        EVA_MAX_FRAMES_PER_EPISODE,
                    )
                    skipped.append(p.name)
                    continue
                datasets[name] = ds_obj
            except Exception as e:
                logger.error(f"Failed to load dataset at {p}: {e}")
                skipped.append(p.name)

        return datasets

    @classmethod
    def _episode_already_present(cls, local_dir: Path, episode_hash: str) -> bool:
        direct = local_dir / episode_hash
        if direct.is_dir():
            return True


class S3EpisodeResolver(EpisodeResolver):
    """
    Resolves episodes via SQL table and optionally syncs from S3.
    """

    def __init__(
        self,
        folder_path: Path,
        bucket_name: str = "rldb",
        main_prefix: str = "processed_v3",
        key_map: dict | None = None,
        transform_list: list | None = None,
        dataset_options: dict | None = None,
        debug: bool = False,
        flow_root: str | Path | None = None,
    ):
        self.bucket_name = bucket_name
        self.main_prefix = main_prefix
        self.debug = debug
        # When set, 3D point-flow sidecars are synced here from app.episodes.flow_3d_path,
        # making the SQL catalogue the source of truth instead of a hand-maintained
        # directory. Point the key_map's point_flow_root at the same path.
        self.flow_root = Path(flow_root).expanduser() if flow_root else None
        super().__init__(
            folder_path,
            key_map=key_map,
            transform_list=transform_list,
            dataset_options=dataset_options,
        )

    def resolve(
        self,
        filters: dict | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters.
        Syncs S3 paths to local_root before indexing.
        """
        filters = dict(filters) if filters is not None else {}
        filters["is_deleted"] = False

        if self.folder_path.is_dir():
            logger.info(f"Using existing directory: {self.folder_path}")
        if not self.folder_path.is_dir():
            self.folder_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Filters: {filters}")

        filtered_paths = self.sync_from_filters(
            bucket_name=self.bucket_name,
            filters=filters,
            local_dir=self.folder_path,
            debug=self.debug,
        )

        valid_hashes = {hashes for _, hashes in filtered_paths}
        if not valid_hashes:
            raise ValueError(
                "No valid collection names from _get_filtered_paths: "
                "filters matched no episodes in the SQL table."
            )

        if self.flow_root is not None:
            self._sync_flow_from_filters(
                filters=filters,
                flow_root=self.flow_root,
                valid_hashes=valid_hashes,
                debug=self.debug,
            )

        datasets = self._load_zarr_datasets(
            search_path=self.folder_path,
            valid_folder_names=valid_hashes,
        )

        return datasets

    @staticmethod
    def _get_flow_paths(filters: dict | None = None) -> dict[str, str]:
        """Map episode_hash -> flow_3d_path for episodes matching ``filters``.

        Episodes whose flow_3d_path is empty are omitted: the column defaults to ''
        and is only back-filled for episodes that have been through the Track4World
        pipeline, so absence is normal rather than an error.
        """
        filters = dict(filters) if filters is not None else {}
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        if "flow_3d_path" not in df.columns:
            raise KeyError(
                "app.episodes has no 'flow_3d_path' column; this resolver needs it to "
                "locate 3D point-flow sidecars. Drop flow_root to fall back on a "
                "pre-populated point_flow_root directory."
            )
        filters = remap_legacy_filters(filters, df.columns)
        series = pd.Series(filters)
        out = df.loc[
            (df[list(filters)] == series).all(axis=1),
            ["episode_hash", "flow_3d_path"],
        ]
        out = out[out["flow_3d_path"].fillna("").astype(str).str.strip() != ""]
        return dict(out.itertuples(index=False, name=None))

    @classmethod
    def _sync_flow_from_filters(
        cls,
        *,
        filters: dict,
        flow_root: Path,
        valid_hashes: set[str],
        numworkers: int = 10,
        debug: bool = False,
    ) -> None:
        """Sync each episode's flow_3d_path prefix into ``flow_root``.

        Each prefix holds segment directories already in the layout PointFlowIndex
        expects (``<root>/<segment_dir>/segment_metadata.json``), so they are flattened
        directly into the root rather than nested per episode. That keeps the loader's
        root-scanning API unchanged.
        """
        flow_paths = cls._get_flow_paths(filters)
        wanted = {h: p for h, p in flow_paths.items() if h in valid_hashes}
        missing = sorted(valid_hashes - set(wanted))
        if missing:
            logger.warning(
                "%d of %d episodes have no flow_3d_path in app.episodes; their flow "
                "targets will be absent (first few: %s)",
                len(missing),
                len(valid_hashes),
                missing[:5],
            )
        if not wanted:
            logger.warning("No flow_3d_path values to sync for these filters.")
            return

        flow_root.mkdir(parents=True, exist_ok=True)
        lines = [
            f'sync "{p.rstrip("/")}/*" "{str(flow_root)}/"'
            for p in sorted(wanted.values())
        ]
        cls._run_s5cmd_batch(lines, numworkers=numworkers)
        logger.info(
            "Synced flow sidecars for %d episodes into %s", len(wanted), flow_root
        )

    @staticmethod
    def _get_filtered_paths(
        filters: dict | None = None, debug: bool = False
    ) -> list[tuple[str, str]]:
        """
        Filters episodes from the SQL episode table according to the criteria specified in `filters`
        and returns a list of (zarr_processed_path, episode_hash) tuples for episodes that match and
        have a non-null zarr_processed_path.

        Args:
            filters (dict): Dictionary of filter key-value pairs to apply on the episode table.

        Returns:
            list[tuple[str, str]]: List of tuples, each containing (zarr_processed_path, episode_hash)
                                   for episodes passing the filter criteria.
        """
        filters = dict(filters) if filters is not None else {}
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        filters = remap_legacy_filters(filters, df.columns)
        series = pd.Series(filters)

        output = df.loc[
            (df[list(filters)] == series).all(axis=1),
            ["zarr_processed_path", "episode_hash"],
        ]
        before_len = len(output)

        if debug:
            logger.info("Debug mode: limiting to 10 datasets.")
            output = output.head(10)

        output = output[
            output["zarr_processed_path"].fillna("").astype(str).str.strip() != ""
        ]
        logger.info(
            f"Skipped {before_len - len(output)} episodes with null zarr_processed_path: {output}"
        )

        paths = list(output.itertuples(index=False, name=None))
        logger.info(f"Paths: {paths}")
        return paths

    @classmethod
    def _sync_s3_to_local(
        cls,
        bucket_name: str,
        s3_paths: list[tuple[str, str]],
        local_dir: Path,
        numworkers: int = 10,
    ):
        if not s3_paths:
            return

        # 0) Skip episodes already present locally
        to_sync = []
        already = []
        for processed_path, episode_hash in s3_paths:
            if cls._episode_already_present(local_dir, episode_hash):
                already.append(episode_hash)
            else:
                to_sync.append((processed_path, episode_hash))

        if already:
            logger.info("Skipping %d episodes already present locally.", len(already))

        if not to_sync:
            logger.info("Nothing to sync from S3 (all episodes already present).")
            return

        # 1) Build s5cmd batch script (one line per episode)
        local_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="_s5cmd_sync_",
            suffix=".txt",
            delete=False,
        ) as tmp_file:
            batch_path = Path(tmp_file.name)

        lines = []
        for processed_path, episode_hash in to_sync:
            # processed_path like: s3://rldb/processed_v2/eva/<hash>/
            if processed_path.startswith("s3://"):
                src_prefix = processed_path.rstrip("/") + "/*"
            else:
                src_prefix = (
                    f"s3://{bucket_name}/{processed_path.lstrip('/').rstrip('/')}"
                    + "/*"
                )

            # Destination is the root local_dir; s5cmd will preserve <hash>/... under it
            dst = local_dir / episode_hash
            lines.append(f'sync "{src_prefix}" "{str(dst)}/"')

        cls._run_s5cmd_batch(lines, numworkers=numworkers, batch_path=batch_path)

    @staticmethod
    def _run_s5cmd_batch(
        lines: list[str],
        numworkers: int = 10,
        batch_path: Path | None = None,
    ) -> None:
        """Run a batch of s5cmd commands against R2. Shared by the zarr and flow syncs."""
        if not lines:
            return
        if batch_path is None:
            with tempfile.NamedTemporaryFile(
                prefix="_s5cmd_sync_", suffix=".txt", delete=False
            ) as tmp_file:
                batch_path = Path(tmp_file.name)
        try:
            batch_path.write_text("\n".join(lines) + "\n")

            load_env()
            rl2_endpoint_url = os.environ.get("R2_ENDPOINT_URL")
            access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
            secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
            if not all([rl2_endpoint_url, access_key_id, secret_access_key]):
                raise ValueError(
                    "R2 credentials missing. Ensure ~/.egoverse_env has "
                    "R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY."
                )
            s5cmd_env = os.environ.copy()
            s5cmd_env["AWS_ACCESS_KEY_ID"] = access_key_id
            s5cmd_env["AWS_SECRET_ACCESS_KEY"] = secret_access_key
            s5cmd_env["AWS_DEFAULT_REGION"] = "auto"
            s5cmd_env["AWS_REGION"] = "auto"
            cmd = [
                "s5cmd",
                "--endpoint-url",
                rl2_endpoint_url,
                "--numworkers",
                str(numworkers),
                "run",
                str(batch_path),
            ]
            logger.info("Running s5cmd batch (%d lines): %s", len(lines), " ".join(cmd))
            subprocess.run(cmd, check=True, env=s5cmd_env)

        finally:
            try:
                batch_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to delete batch file %s: %s", batch_path, e)

    @classmethod
    def sync_from_filters(
        cls,
        *,
        bucket_name: str,
        filters: dict,
        local_dir: Path,
        numworkers: int = 10,
        debug: bool = False,
    ):
        """
        Public API:
        - resolves episodes from DB using filters
        - runs a single aws s3 sync with includes
        - downloads into local_dir

        Args:
            numworkers: Number of parallel workers for s5cmd.

        Returns:
            List[(processed_path, episode_hash)]
        """

        # 1) Resolve episodes from DB
        filtered_paths = cls._get_filtered_paths(filters, debug=debug)
        if not filtered_paths:
            logger.warning("No episodes matched filters.")
            return []

        # 2) Logging
        logger.info(
            f"Syncing S3 datasets with filters {filters} to local directory {local_dir}..."
        )

        # 3) Sync
        cls._sync_s3_to_local(
            bucket_name=bucket_name,
            s3_paths=filtered_paths,
            local_dir=local_dir,
            numworkers=numworkers,
        )

        return filtered_paths


class LocalEpisodeResolver(EpisodeResolver):
    """
    Resolves episodes from local Zarr stores, filtering via local metadata.
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        dataset_options: dict | None = None,
        debug=False,
    ):
        super().__init__(folder_path, key_map, transform_list, dataset_options)
        self.debug = debug

    @staticmethod
    def _local_filters_match(metadata: dict, episode_hash: str, filters: dict) -> bool:
        for key, value in filters.items():
            if key == "episode_hash":
                if episode_hash != value:
                    return False
                continue

            if key == "robot_name":
                meta_value = (
                    metadata.get("robot_name")
                    or metadata.get("robot_type")
                    or metadata.get("embodiment")
                )
            elif key == "is_deleted":
                meta_value = metadata.get("is_deleted", False)
            else:
                meta_value = metadata.get(key)

            if meta_value is None:
                return False
            if meta_value != value:
                return False

        return True

    @classmethod
    def _get_local_filtered_paths(
        cls, search_path: Path, filters: dict, debug: bool = False
    ):
        if not search_path.is_dir():
            logger.warning("Local path does not exist: %s", search_path)
            return []

        filtered = []
        for p in sorted(search_path.iterdir()):
            if not p.is_dir():
                continue

            episode_hash = p.name[:-5] if p.name.endswith(".zarr") else p.name

            try:
                store = zarr.open_group(str(p), mode="r")
                metadata = dict(store.attrs)
            except Exception as e:
                logger.warning("Failed to read metadata for %s: %s", p, e)
                continue

            if cls._local_filters_match(metadata, episode_hash, filters):
                filtered.append((str(p), episode_hash))

        if debug:
            logger.info("Debug mode: limiting to 10 datasets.")
            filtered = filtered[:10]

        logger.info("Local filtered paths: %s", filtered)
        return filtered

    def resolve(
        self,
        sync_from_s3=False,
        filters: dict | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters from local data.
        """
        if sync_from_s3:
            logger.warning(
                "LocalEpisodeResolver does not sync from S3; ignoring sync_from_s3=True."
            )

        filters = dict(filters) if filters is not None else {}
        filters.setdefault("is_deleted", False)

        filtered_paths = self._get_local_filtered_paths(
            self.folder_path, filters, debug=self.debug
        )

        valid_folder_names = {folder_name for _, folder_name in filtered_paths}
        logger.info(f"Valid folder names: {valid_folder_names}")
        if not valid_folder_names:
            raise ValueError(
                "No valid collection names from local filtering: "
                "filters matched no episodes in the local directory."
            )

        datasets = self._load_zarr_datasets(
            search_path=self.folder_path, valid_folder_names=valid_folder_names
        )

        return datasets


class MultiDataset(torch.utils.data.Dataset):
    """
    Self wrapping MultiDataset, can wrap zarr or multi dataset.

    """

    def __init__(
        self,
        datasets,
        mode="train",
        percent=0.1,
        valid_ratio=0.2,
        **kwargs,
    ):
        """
        Args:
            datasets (dict): Dictionary mapping unique dataset hashes (str) to dataset objects. Datasets can be individual Zarr datasets or other multi-datasets; mixing different types is supported.
            mode (str, optional): Split mode to use (e.g., "train", "valid"). Defaults to "train".
            percent (float, optional): Fraction of the dataset to use from each underlying dataset. Defaults to 0.1.
            valid_ratio (float, optional): Validation split ratio for datasets that support a train/valid split.
            **kwargs: Additional keyword arguments passed to underlying dataset constructors if needed.
        """
        self.train_collections, self.valid_collections = split_dataset_names(
            datasets.keys(), valid_ratio=valid_ratio, seed=SEED
        )

        if mode == "train":
            chosen = self.train_collections
        elif mode == "valid":
            chosen = self.valid_collections
        elif mode == "total":
            chosen = set(datasets.keys())
        elif mode == "percent":
            all_names = sorted(datasets.keys())
            rng = random.Random(SEED)
            rng.shuffle(all_names)

            n_keep = int(len(all_names) * percent)
            if percent > 0.0:
                n_keep = max(1, n_keep)
            chosen = set(all_names[:n_keep])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.datasets = {rid: ds for rid, ds in datasets.items() if rid in chosen}
        assert self.datasets, "No datasets left after applying mode split."

        self.index_map = []
        for dataset_name, dataset in self.datasets.items():
            for local_idx in range(len(dataset)):
                self.index_map.append((dataset_name, local_idx))

        super().__init__()

    @property
    def embodiment(self):
        """Resolve embodiment by delegating to the first child dataset."""
        first_ds = next(iter(self.datasets.values()))
        return first_ds.embodiment

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_name, local_idx = self.index_map[idx]
        data = self.datasets[dataset_name][local_idx]

        robot_name = self.datasets[dataset_name].embodiment
        data["metadata.robot_name"] = robot_name
        data["embodiment"] = robot_name

        return data

    @classmethod
    def _from_resolver(cls, resolver: EpisodeResolver, **kwargs):
        """
        create a MultiDataset from an EpisodeResolver.

        Args:
            resolver (EpisodeResolver): The resolver instance to use for loading datasets.
            embodiment: The embodiment identifier to use for resolving datasets.
            **kwargs: Keyword args forwarded to resolver (e.g., filters,
                sync_from_s3) and MultiDataset constructor (e.g., mode, percent,
                key_map, valid_ratio).
        Returns:
            MultiDataset: The constructed multi-dataset.
        """
        # TODO add key_map and transform pass to children

        sync_from_s3 = kwargs.pop("sync_from_s3", False)
        filters = kwargs.pop("filters", {}) or {}

        if isinstance(resolver, LocalEpisodeResolver):
            resolved = resolver.resolve(
                sync_from_s3=sync_from_s3,
                filters=filters,
            )
        else:
            resolved = resolver.resolve(filters=filters)

        return cls(datasets=resolved, **kwargs)


class ZarrDataset(torch.utils.data.Dataset):
    """
    Base Zarr Dataset object, Just intializes as pass through to read from zarr episode
    """

    def __init__(
        self,
        Episode_path: Path,
        key_map: dict,
        transform_list: list | None = None,
        prefix_pose_repair: dict | None = None,
        flow_root: str | Path | None = None,
        flow_filter_to_anchors: bool = True,
    ):
        """
        Args:
            episode_path: just a path to the designated zarr episode
            key_map: dict mapping from dataset keys to zarr keys and horizon info, e.g. {"obs/image/front": {"zarr_key": "observations.images.front", "horizon": 4}, ...}
            transform_list: list of Transform objects to apply to the data after loading, e.g. for action chunk transformations. Should be in order of application.
        """
        self.episode_path = Episode_path
        self.key_map = key_map
        self.transform = transform_list
        self.metadata = None
        self._image_keys = None  # Lazy-loaded set of JPEG-encoded keys
        self._json_keys = None  # Lazy-loaded set of JSON-encoded keys
        self._annotations = None
        self.prefix_pose_repair = self._normalize_prefix_pose_repair(prefix_pose_repair)
        self._prefix_pose_repair_cache: dict[str, np.ndarray] = {}
        self.episode_starts: list[int] = []
        self.episode_ends: list[int] = []
        self._segment_start_by_frame = np.empty((0,), dtype=np.int64)
        self._segment_end_by_frame = np.empty((0,), dtype=np.int64)
        self._flow_root = (
            Path(flow_root).expanduser().resolve() if flow_root is not None else None
        )
        self._flow_filter_to_anchors = bool(flow_filter_to_anchors)
        self._external_flow_specs: dict[str, dict] = {}
        self._flow_segments_by_anchor: dict[int, AriaFlowSegment] = {}
        self._filtered_indices: np.ndarray | None = None
        self.init_episode()
        self._init_external_flow()
        self._init_point_flow_filter()
        super().__init__()

    def init_episode(self):
        """
        inits the zarr episode and all the metadata associated, as well as total_frames for len
        """
        self.episode_reader = ZarrEpisode(self.episode_path)
        self.metadata = self.episode_reader.metadata
        self.total_frames = self.metadata["total_frames"]
        self.embodiment = self.metadata["embodiment"]
        self.keys_dict = {k: (0, None) for k in self.episode_reader._collect_keys()}

        # Detect JPEG-encoded image keys from metadata
        self._image_keys = self._detect_image_keys()
        self._json_keys = self._detect_json_keys()
        self._init_internal_segments()
        self._prefix_pose_repair_cache = self._build_prefix_pose_repair_cache()

    def _init_external_flow(self) -> None:
        self._external_flow_specs = {
            key: spec
            for key, spec in self.key_map.items()
            if spec.get("source") in EXTERNAL_FLOW_SOURCES
        }
        if not self._external_flow_specs:
            return

        flow_sources = {
            spec.get("source") for spec in self._external_flow_specs.values()
        }
        if flow_sources == {ARIA_EXTERNAL_FLOW_SOURCE}:
            if not str(self.embodiment).startswith("aria"):
                raise ValueError(
                    "External Aria flow supervision is only supported for Aria embodiments."
                )
        elif flow_sources == {EVA_EXTERNAL_FLOW_SOURCE}:
            if not _is_eva_embodiment(self.embodiment):
                raise ValueError(
                    "External EVA flow supervision is only supported for EVA embodiments."
                )
        else:
            raise ValueError(
                "Mixed external flow sources are not supported in one dataset: "
                f"{sorted(str(source) for source in flow_sources)}"
            )

        flow_roots = {
            Path(spec["flow_root"]).expanduser().resolve()
            for spec in self._external_flow_specs.values()
            if spec.get("flow_root")
        }
        if self._flow_root is not None:
            flow_roots.add(self._flow_root)
        if len(flow_roots) != 1:
            raise ValueError(
                "External flow supervision requires exactly one flow root. "
                f"Resolved roots: {sorted(str(root) for root in flow_roots)}"
            )

        flow_root = next(iter(flow_roots))
        flow_index = AriaFlowArtifactIndex.from_root(flow_root)
        episode_name = _episode_basename(self.episode_path)
        self._flow_root = flow_root
        self._flow_segments_by_anchor = flow_index.segments_for_source(episode_name)

        if not self._flow_segments_by_anchor:
            logger.warning(
                "No external flow segments found for %s under %s.",
                episode_name,
                flow_root,
            )

        if self._flow_filter_to_anchors:
            self._filtered_indices = self._build_flow_anchor_indices()

    def _init_point_flow_filter(self) -> None:
        """Restrict __getitem__ to zarr frames covered by point-flow segments.

        Point-flow artifacts cover discontinuous zarr frame ranges (Aria
        splits one recording into multiple segments with gaps between). If we
        don't filter, __getitem__ can hand an uncovered frame to
        _load_point_flow_target, which raises IndexError from PointFlowIndex.
        """
        point_flow_specs = [
            spec
            for spec in self.key_map.values()
            if spec.get("source") == POINT_FLOW_SOURCE
        ]
        if not point_flow_specs:
            return

        covered_by_root: dict[str, set[int]] = {}
        # Aria-style head/tail skip: trim the first / last N frames of each
        # point-flow segment before exposing anchors to __getitem__. Segments
        # that collapse to length <= 0 after trimming are discarded entirely.
        # Multiple specs on the same root must agree on skip amounts (take the
        # max so neither head's contract is violated).
        skip_head_by_root: dict[str, int] = {}
        skip_tail_by_root: dict[str, int] = {}
        for spec in point_flow_specs:
            root = spec.get("point_flow_root")
            if not root:
                raise KeyError("point-flow spec missing 'point_flow_root'")
            root_key = str(Path(root).expanduser().resolve())
            skip_head_by_root[root_key] = max(
                skip_head_by_root.get(root_key, 0),
                int(spec.get("head_skip_frames", 0)),
            )
            skip_tail_by_root[root_key] = max(
                skip_tail_by_root.get(root_key, 0),
                int(spec.get("tail_skip_frames", 0)),
            )

        for spec in point_flow_specs:
            root = spec.get("point_flow_root")
            root_key = str(Path(root).expanduser().resolve())
            covered = covered_by_root.setdefault(root_key, set())
            head = skip_head_by_root[root_key]
            tail = skip_tail_by_root[root_key]
            segs = PointFlowIndex.segments_for(root, self.episode_path)
            for start, end, _ in segs:
                s = max(int(start), 0) + head
                e = min(int(end), int(self.total_frames)) - tail
                if e > s:
                    covered.update(range(s, e))

        if len(covered_by_root) == 1:
            point_flow_indices = next(iter(covered_by_root.values()))
        else:
            point_flow_indices = set.intersection(*covered_by_root.values())

        if self._filtered_indices is not None:
            existing = {int(i) for i in self._filtered_indices.tolist()}
            point_flow_indices = point_flow_indices & existing

        if not point_flow_indices:
            logger.warning(
                "No zarr frames covered by point-flow segments for %s; "
                "dataset will expose 0 samples.",
                self.episode_path,
            )

        self._filtered_indices = np.asarray(sorted(point_flow_indices), dtype=np.int64)

    def _build_flow_anchor_indices(self) -> np.ndarray:
        valid_anchor_sets: list[set[int]] = []
        for spec in self._external_flow_specs.values():
            anchors = {
                anchor_idx
                for anchor_idx, segment in self._flow_segments_by_anchor.items()
                if self._external_flow_target_available(anchor_idx, segment, spec)
            }
            valid_anchor_sets.append(anchors)

        if not valid_anchor_sets:
            return np.empty((0,), dtype=np.int64)

        common = set.intersection(*valid_anchor_sets)
        if not common:
            return np.empty((0,), dtype=np.int64)
        return np.asarray(sorted(common), dtype=np.int64)

    def _external_flow_target_available(
        self,
        anchor_idx: int,
        segment: AriaFlowSegment,
        spec: dict,
    ) -> bool:
        try:
            segment_start, segment_end = self._segment_bounds_for_index(anchor_idx)
            _resolve_external_flow_future_step(
                segment.valid_frame_indices,
                anchor_frame_idx=int(anchor_idx),
                future_offset=int(spec.get("future_offset", 0)),
                segment_start=segment_start,
                segment_end=segment_end,
            )
        except (IndexError, ValueError):
            return False
        return True

    @staticmethod
    def _normalize_prefix_pose_repair(
        prefix_pose_repair: dict | None,
    ) -> dict | None:
        if prefix_pose_repair is None:
            return None

        if not isinstance(prefix_pose_repair, dict):
            raise TypeError(
                "prefix_pose_repair must be a dict or None, got "
                f"{type(prefix_pose_repair)}"
            )

        zarr_key = prefix_pose_repair.get("zarr_key")
        zarr_keys = prefix_pose_repair.get("zarr_keys")
        threshold = prefix_pose_repair.get("threshold")
        if threshold is None:
            raise ValueError(
                "prefix_pose_repair must include 'threshold' and one of "
                "'zarr_key' or 'zarr_keys'."
            )

        if zarr_key is not None and zarr_keys is not None:
            raise ValueError(
                "prefix_pose_repair accepts either 'zarr_key' or 'zarr_keys', not both."
            )

        if zarr_key is not None:
            normalized_keys = [str(zarr_key)]
        else:
            if zarr_keys is None:
                raise ValueError(
                    "prefix_pose_repair must include one of 'zarr_key' or 'zarr_keys'."
                )
            if isinstance(zarr_keys, str):
                normalized_keys = [zarr_keys]
            else:
                normalized_keys = [str(key) for key in zarr_keys]
            if not normalized_keys:
                return None

        return {
            "zarr_keys": normalized_keys,
            "threshold": float(threshold),
        }

    def _build_prefix_pose_repair_cache(self) -> dict[str, np.ndarray]:
        if self.prefix_pose_repair is None:
            return {}

        threshold = self.prefix_pose_repair["threshold"]
        repaired_cache: dict[str, np.ndarray] = {}
        best_key = None
        best_break_idx = None
        best_break_delta = None
        best_repaired = None
        debug_messages: list[str] = []
        for zarr_key in self.prefix_pose_repair["zarr_keys"]:
            if zarr_key not in self.episode_reader._store:
                logger.warning(
                    "prefix_pose_repair key %s not found in %s",
                    zarr_key,
                    self.episode_path,
                )
                continue

            raw = np.asarray(self.episode_reader._store[zarr_key][: self.total_frames])
            break_info = self._find_prefix_pose_break(raw, threshold=threshold)
            if break_info is None:
                continue
            break_idx, break_delta = break_info
            debug_messages.append(
                f"{zarr_key} first break at {break_idx} -> {break_idx + 1} "
                f"({break_delta:.6f})"
            )

            if best_break_idx is None or break_idx < best_break_idx:
                best_key = zarr_key
                best_break_idx = break_idx
                best_break_delta = break_delta
                best_repaired = self._repair_prefix_pose_array(raw, break_idx=break_idx)

        if best_key is not None and best_repaired is not None:
            repaired_cache[best_key] = best_repaired
            logger.info(
                "prefix_pose_repair "
                f"threshold > {threshold}: "
                + ", ".join(debug_messages)
                + f"; applying fix to {best_key} at {best_break_idx} -> "
                f"{best_break_idx + 1} ({best_break_delta:.6f}) for "
                f"{self.episode_path}"
            )
        else:
            logger.info(
                "prefix_pose_repair "
                f"threshold > {threshold}: no break found for "
                f"{self.prefix_pose_repair['zarr_keys']} in {self.episode_path}"
            )

        return repaired_cache

    @staticmethod
    def _pose_step_deltas(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2 or arr.shape[0] < 2:
            return np.empty((0,), dtype=np.float64)

        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape[1] != 7:
            return np.linalg.norm(np.diff(arr, axis=0), axis=1)

        trans_deltas = np.linalg.norm(np.diff(arr[:, :3], axis=0), axis=1)
        quat_prev = arr[:-1, 3:7]
        quat_next = arr[1:, 3:7]
        quat_prev_norms = np.linalg.norm(quat_prev, axis=1, keepdims=True)
        quat_next_norms = np.linalg.norm(quat_next, axis=1, keepdims=True)

        quat_deltas = np.linalg.norm(quat_next - quat_prev, axis=1)
        valid_mask = (quat_prev_norms[:, 0] > 0) & (quat_next_norms[:, 0] > 0)
        if np.any(valid_mask):
            quat_prev_unit = quat_prev[valid_mask] / quat_prev_norms[valid_mask]
            quat_next_unit = quat_next[valid_mask] / quat_next_norms[valid_mask]
            quat_deltas[valid_mask] = np.minimum(
                np.linalg.norm(quat_next_unit - quat_prev_unit, axis=1),
                np.linalg.norm(quat_next_unit + quat_prev_unit, axis=1),
            )

        return np.sqrt(trans_deltas**2 + quat_deltas**2)

    @staticmethod
    def _find_prefix_pose_break(
        arr: np.ndarray,
        *,
        threshold: float,
        min_tail_frames: int = 50,
    ) -> tuple[int, float] | None:
        if arr.ndim != 2 or arr.shape[0] < 2:
            return None

        if not np.issubdtype(arr.dtype, np.number):
            return None

        deltas = ZarrDataset._pose_step_deltas(arr)
        break_indices = np.flatnonzero(deltas > threshold)
        if break_indices.size == 0:
            return None

        break_idx = int(break_indices[0])
        if arr.shape[0] - (break_idx + 1) < min_tail_frames:
            return None
        return break_idx, float(deltas[break_idx])

    @staticmethod
    def _repair_prefix_pose_array(
        arr: np.ndarray, *, break_idx: int
    ) -> np.ndarray | None:
        if arr.ndim != 2 or arr.shape[0] < 2:
            return None

        replacement_idx = break_idx + 1
        if replacement_idx >= arr.shape[0]:
            return None

        repaired = arr.copy()
        repaired[:replacement_idx] = repaired[replacement_idx]
        return repaired

    def _set_single_segment(self) -> None:
        if self.total_frames <= 0:
            self.episode_starts = []
            self.episode_ends = []
            self._segment_start_by_frame = np.empty((0,), dtype=np.int64)
            self._segment_end_by_frame = np.empty((0,), dtype=np.int64)
            return

        self.episode_starts = [0]
        self.episode_ends = [self.total_frames - 1]
        self._segment_start_by_frame = np.zeros((self.total_frames,), dtype=np.int64)
        self._segment_end_by_frame = np.full(
            (self.total_frames,), self.total_frames - 1, dtype=np.int64
        )

    def _init_internal_segments(self) -> None:
        self._set_single_segment()

        if self.total_frames <= 1:
            return

        if not str(self.embodiment).startswith("aria"):
            return

        if ARIA_TIMESTAMP_KEY not in self.episode_reader._store:
            return

        timestamps = self._load_valid_timestamps()
        if timestamps is None:
            return

        cut_indices = self._detect_internal_episode_cuts(timestamps)
        self._build_segment_lookup(cut_indices)

    def _load_valid_timestamps(self) -> np.ndarray | None:
        try:
            timestamps = self.episode_reader._store[ARIA_TIMESTAMP_KEY][
                : self.total_frames
            ]
        except Exception as exc:
            logger.warning(
                "Failed to load %s for %s: %s",
                ARIA_TIMESTAMP_KEY,
                self.episode_path,
                exc,
            )
            return None

        if not isinstance(timestamps, np.ndarray):
            timestamps = np.asarray(timestamps)

        if timestamps.shape[0] < self.total_frames:
            logger.warning(
                "Timestamp key %s shorter than total_frames for %s: %s < %s",
                ARIA_TIMESTAMP_KEY,
                self.episode_path,
                timestamps.shape[0],
                self.total_frames,
            )
            return None

        if not np.issubdtype(timestamps.dtype, np.number):
            logger.warning(
                "Timestamp key %s is non-numeric for %s: %s",
                ARIA_TIMESTAMP_KEY,
                self.episode_path,
                timestamps.dtype,
            )
            return None

        return timestamps

    def _detect_internal_episode_cuts(self, timestamps: np.ndarray) -> np.ndarray:
        if timestamps.shape[0] <= 1:
            return np.empty((0,), dtype=np.int64)

        diffs = np.diff(timestamps)
        return np.flatnonzero(diffs > ARIA_EPISODE_GAP_THRESHOLD_NS).astype(np.int64)

    def _build_segment_lookup(self, cut_indices: np.ndarray) -> None:
        if self.total_frames <= 0:
            self._set_single_segment()
            return

        starts = [0, *[int(cut) + 1 for cut in cut_indices.tolist()]]
        ends = [*[int(cut) for cut in cut_indices.tolist()], self.total_frames - 1]

        segment_start_by_frame = np.empty((self.total_frames,), dtype=np.int64)
        segment_end_by_frame = np.empty((self.total_frames,), dtype=np.int64)

        for start, end in zip(starts, ends, strict=True):
            segment_start_by_frame[start : end + 1] = start
            segment_end_by_frame[start : end + 1] = end

        self.episode_starts = starts
        self.episode_ends = ends
        self._segment_start_by_frame = segment_start_by_frame
        self._segment_end_by_frame = segment_end_by_frame

    def _detect_image_keys(self) -> set[str]:
        """
        Detect which keys contain JPEG-encoded image data from metadata.

        Returns:
            Set of keys containing JPEG data
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "jpeg"}

    def _detect_json_keys(self) -> set[str]:
        """
        Detect keys containing JSON-encoded bytes from metadata.

        Returns:
            Set of keys containing JSON payloads.
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "json"}

    @staticmethod
    def _decode_json_entry(value):
        if isinstance(value, np.void):
            value = value.item()
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _load_annotations(self) -> list[dict]:
        """
        Load and cache decoded language annotations.

        Expected format per entry:
            {"text": str, "start_idx": int, "end_idx": int}
        """
        if self._annotations is not None:
            return self._annotations

        raw = self.episode_reader._store["annotations"][:]

        decoded = [self._decode_json_entry(x) for x in raw]
        self._annotations = [d for d in decoded if isinstance(d, dict)]
        return self._annotations

    def _annotation_text_for_frame(self, frame_idx: int) -> str:
        """
        Resolve language annotation text for a frame from span annotations.
        """
        annotations = self._load_annotations()
        for ann in annotations:
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            if start_idx <= frame_idx <= end_idx:
                return str(ann.get("text", ""))
        return ""

    def _segment_bounds_for_index(self, idx: int) -> tuple[int, int]:
        if self.total_frames <= 0:
            raise IndexError("Cannot resolve segment bounds for an empty dataset.")

        clamped_idx = min(max(idx, 0), self.total_frames - 1)
        return (
            int(self._segment_start_by_frame[clamped_idx]),
            int(self._segment_end_by_frame[clamped_idx]),
        )

    def _resolve_bounded_read_interval(
        self,
        anchor_idx: int,
        horizon: int | None,
        future_offset: int = 0,
    ) -> tuple[int, int | None]:
        segment_start, segment_end = self._segment_bounds_for_index(anchor_idx)
        base_idx = min(max(anchor_idx + future_offset, segment_start), segment_end)

        if horizon is None:
            return base_idx, None

        end_idx = min(base_idx + horizon, segment_end + 1)
        return base_idx, end_idx

    def _resolve_dataset_index(self, idx: int) -> int:
        if self._filtered_indices is None:
            return idx

        if idx < 0 or idx >= self._filtered_indices.shape[0]:
            raise IndexError(
                f"Filtered dataset index {idx} out of range for length {len(self)}."
            )
        return int(self._filtered_indices[idx])

    def _load_head_pose_for_frame(self, frame_idx: int) -> np.ndarray:
        head_pose = self._read_zarr_key_interval("obs_head_pose", (frame_idx, None))
        return np.asarray(head_pose, dtype=np.float64)

    def _load_point_flow_target(
        self, frame_idx: int, spec: dict
    ) -> dict[str, np.ndarray]:
        """Build the native-k point-flow trajectory + sidecars for one (frame, key).

        Returns a dict with the main trajectory tensor under "trajectory" plus
        the sidecars needed to assemble dict conditioning for PointFMPolicy:
          - "trajectory":     (k_raw, N, D) float32   (native-k, every anchor)
          - "current_points": (N, D) float32          (= trajectory[0])
          - "validity_mask":  (N,) bool               (True = finite P_t / P_tk)
          - "frame_skip":     int scalar (for reference)

        __getitem__ unpacks this into parallel batch keys "{k}",
        "{k}__current_points", "{k}__validity_mask". D=2 for the 2D head, D=3
        for the 3D head. Shared validity mask + subsample indices are cached on
        the ZarrDataset so a joint 2D+3D run sees the same indices in both
        heads within one __getitem__.

        InterpolatePointTrajectory (later in the embodiment's transform chain)
        resamples "trajectory" to FLOW_CHUNK_LENGTH=100. Normalization runs
        there too, using flow_stats injected via ensure_flow_stats.
        """
        root = spec.get("point_flow_root")
        if not root:
            raise KeyError("point-flow spec missing 'point_flow_root'")
        dim = int(spec.get("point_dim", 3))
        if dim not in (2, 3):
            raise ValueError(f"point_dim must be 2 or 3, got {dim}")
        movement_threshold = float(spec.get("movement_threshold", 0.0))

        episode = PointFlowIndex.load_episode(
            root, self.episode_path, load_2d=(dim == 2), frame_idx=int(frame_idx)
        )
        # Builders take the zarr-absolute frame_idx and translate to segment-
        # local internally via _local_idx. No manual translation needed here.

        validity = build_threshold_mask(
            episode,
            int(frame_idx),
            dim=dim,
            threshold=movement_threshold,
        )

        if dim == 3:
            traj = build_3d_endpoint_trajectory(episode, int(frame_idx))
        else:
            traj = build_2d_endpoint_trajectory(episode, int(frame_idx))
        traj = traj.astype(np.float32, copy=False)
        # Zero out points the mask rejected (below threshold or non-finite) so
        # no NaN / noisy-background values reach the head. Loss masking then
        # sees exactly the same validity as the decoder input.
        bad = ~validity
        if bad.any():
            traj = np.where(bad[None, :, None], np.zeros_like(traj), traj)

        # sampled_uv is the static (u, v) pixel anchor per point. Plumb it so
        # val-time viz can draw per-point motion arrows on the base frame.
        sampled_uv = np.asarray(episode.sampled_uv, dtype=np.int64)
        if sampled_uv.ndim != 2 or sampled_uv.shape[-1] != 2:
            raise ValueError(
                f"sampled_uv must be (N, 2), got {sampled_uv.shape} in {episode.episode_dir}"
            )

        return {
            "trajectory": traj,
            "current_points": traj[0].copy(),
            "validity_mask": validity,
            "sampled_uv": sampled_uv.astype(np.int64, copy=False),
            "frame_skip": int(episode.frame_skip),
        }

    @staticmethod
    def _load_raw_external_flow(
        segment: AriaFlowSegment, anchor_step: int, future_step: int
    ) -> np.ndarray:
        trajectory = np.load(segment.trajectory_path, mmap_mode="r")
        if trajectory.ndim != 4 or trajectory.shape[1] != 2:
            raise ValueError(
                f"Expected dense trajectory array of shape (T, 2, H, W), got "
                f"{trajectory.shape} from {segment.trajectory_path}"
            )
        _validate_flow_query_grid(
            np.asarray(trajectory[0], dtype=np.float32),
            trajectory_path=segment.trajectory_path,
        )

        return np.asarray(trajectory[future_step], dtype=np.float32) - np.asarray(
            trajectory[anchor_step], dtype=np.float32
        )

    def _load_external_flow_target(self, frame_idx: int, spec: dict) -> np.ndarray:
        segment = self._flow_segments_by_anchor.get(int(frame_idx))
        if segment is None:
            raise KeyError(
                f"No external flow segment found for frame {frame_idx} in "
                f"{self.episode_path}."
            )

        segment_start, segment_end = self._segment_bounds_for_index(int(frame_idx))
        future_step, future_frame_idx = _resolve_external_flow_future_step(
            segment.valid_frame_indices,
            anchor_frame_idx=int(frame_idx),
            future_offset=int(spec.get("future_offset", 0)),
            segment_start=segment_start,
            segment_end=segment_end,
        )

        valid_frame_indices = np.asarray(segment.valid_frame_indices, dtype=np.int64)
        anchor_matches = np.flatnonzero(valid_frame_indices == int(frame_idx))
        if anchor_matches.size == 0:
            raise KeyError(
                f"Anchor frame {frame_idx} not present in valid flow indices for "
                f"{self.episode_path}."
            )
        anchor_step = int(anchor_matches[0])
        current_frame_idx = int(frame_idx)

        raw_flow = self._load_raw_external_flow(segment, anchor_step, future_step)
        source = spec.get("source")
        if source == ARIA_EXTERNAL_FLOW_SOURCE:
            current_head_pose = self._load_head_pose_for_frame(current_frame_idx)
            future_head_pose = self._load_head_pose_for_frame(future_frame_idx)
            output_flow = raw_flow - _camera_rotation_flow_from_head_poses(
                current_head_pose,
                future_head_pose,
                image_shape=tuple(raw_flow.shape[1:]),
            )
        elif source == EVA_EXTERNAL_FLOW_SOURCE:
            output_flow = raw_flow
        else:
            raise ValueError(f"Unsupported external flow source: {source}")

        # Dense future_flow is returned in raw pixel-displacement units. The new
        # point-conditioned flow heads (future_flow_2d_points / future_flow_3d_points)
        # use a separate loader path and normalize via flow_stats.json. Do NOT apply
        # _normalize_flow_by_image_resolution here — downstream _build_flow_panel and
        # MSE/L1 logging in hpt_wm.py assume raw pixel units.
        return _uniform_subsample_flow(
            output_flow,
            stride=int(spec.get("downsample_stride", ARIA_FLOW_DOWNSAMPLE_STRIDE)),
        )

    def __len__(self) -> int:
        if self._filtered_indices is not None:
            return int(self._filtered_indices.shape[0])
        return self.total_frames

    def _pad_sequences(self, data, horizon: int | None) -> dict:
        if horizon is None:
            return data

        # Note that k is zarr key
        for k in data:
            if isinstance(data[k], np.ndarray):
                seq_len = data[k].shape[0]
                if seq_len < horizon:
                    # Pad by repeating the last frame
                    pad_len = horizon - seq_len
                    last_frame = data[k][-1:]  # Keep dims: (1, action_dim)
                    padding = np.repeat(last_frame, pad_len, axis=0)
                    data[k] = np.concatenate([data[k], padding], axis=0)

        return data

    def _read_zarr_key_interval(self, zarr_key: str, interval: tuple[int, int | None]):
        if zarr_key in self._prefix_pose_repair_cache:
            start, end = interval
            cached = self._prefix_pose_repair_cache[zarr_key]
            if end is None:
                return cached[start].copy()
            return cached[start:end].copy()

        raw = self.episode_reader.read({zarr_key: interval})
        return raw[zarr_key]

    def _decode_loaded_value(self, zarr_key: str, value):
        if zarr_key in self._image_keys:
            if (
                isinstance(value, np.ndarray)
                and value.dtype == object
                and value.ndim == 1
            ):
                decoded_seq = []
                for jpeg_bytes in value:
                    decoded = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
                    decoded_seq.append(np.transpose(decoded, (2, 0, 1)) / 255.0)
                return np.stack(decoded_seq, axis=0)

            decoded = simplejpeg.decode_jpeg(value, colorspace="RGB")
            return np.transpose(decoded, (2, 0, 1)) / 255.0

        if zarr_key in self._json_keys:
            if isinstance(value, np.ndarray):
                return [self._decode_json_entry(v) for v in value]
            return self._decode_json_entry(value)

        return value

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Build keys_dict with ranges based on whether action chunking is enabled
        idx = self._resolve_dataset_index(idx)
        # Reset per-sample point-flow cache so 2D and 3D heads share the same
        # valid-point subsample within this __getitem__ but not across samples.
        self._point_flow_cache = {}
        data = {}
        for k in self.key_map:
            spec = self.key_map[k]
            zarr_key = spec["zarr_key"]
            horizon = spec.get("horizon", None)

            if zarr_key == "annotations":
                data[k] = self._annotation_text_for_frame(idx)
                continue

            if spec.get("source") in EXTERNAL_FLOW_SOURCES:
                data[k] = self._load_external_flow_target(idx, spec)
                continue

            if spec.get("source") == POINT_FLOW_SOURCE:
                pf = self._load_point_flow_target(idx, spec)
                data[k] = pf["trajectory"]
                data[f"{k}__current_points"] = pf["current_points"]
                data[f"{k}__validity_mask"] = pf["validity_mask"]
                data[f"{k}__sampled_uv"] = pf["sampled_uv"]
                continue

            future_offset = spec.get("future_offset", 0)
            read_interval = self._resolve_bounded_read_interval(
                anchor_idx=idx,
                horizon=horizon,
                future_offset=future_offset,
            )

            raw_data = {zarr_key: self._read_zarr_key_interval(zarr_key, read_interval)}
            self._pad_sequences(raw_data, horizon)  # should be able to pad images
            data[k] = self._decode_loaded_value(zarr_key, raw_data[zarr_key])

        bound = self._segment_bounds_for_index(
            idx
        )  # sanity check for segment bounds resolution
        data["bound"] = np.array(bound, dtype=np.int64)

        # TODO add the transform list code here
        if self.transform:
            for transform in self.transform or []:
                try:
                    data = transform.transform(data)
                except Exception as e:
                    logger.error(f"Error transforming data: {e}")
                    logger.error(f"Data: {data}")
                    logger.error(f"Transform: {transform}")
                    logger.error(f"Error: {e}")
                    if idx == 0:
                        logger.error("Error in first frame")
                        raise e
                    else:
                        return self.__getitem__(0)

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(torch.float32)

        return data

    def get_item_keys(self, idx: int, keys) -> dict[str, torch.Tensor]:
        idx = self._resolve_dataset_index(idx)
        requested = self._normalize_keys_arg(keys)
        out = {}

        for k in requested:
            if k not in self.key_map:
                raise KeyError(
                    f"Unknown key '{k}'. Available keys: {list(self.key_map.keys())}"
                )

            spec = self.key_map[k]
            zarr_key = spec["zarr_key"]
            horizon = spec.get("horizon", None)
            future_offset = spec.get("future_offset", 0)

            if spec.get("source") in EXTERNAL_FLOW_SOURCES:
                out[k] = self._load_external_flow_target(idx, spec)
                continue

            if spec.get("source") == POINT_FLOW_SOURCE:
                pf = self._load_point_flow_target(idx, spec)
                out[k] = pf["trajectory"]
                out[f"{k}__current_points"] = pf["current_points"]
                out[f"{k}__validity_mask"] = pf["validity_mask"]
                out[f"{k}__sampled_uv"] = pf["sampled_uv"]
                continue

            interval = self._resolve_bounded_read_interval(
                anchor_idx=idx,
                horizon=horizon,
                future_offset=future_offset,
            )

            raw = {zarr_key: self._read_zarr_key_interval(zarr_key, interval)}
            self._pad_sequences(raw, horizon)
            out[k] = self._decode_loaded_value(zarr_key, raw[zarr_key])

        if self.transform:
            for transform in self.transform or []:
                try:
                    out = transform.transform(out)
                except Exception as e:
                    logger.error(f"Error transforming data: {e}")
                    # NOTE: avoid dumping full arrays into logs
                    logger.error(f"Data keys: {list(out.keys())}")
                    logger.error(f"Transform: {transform}")
                    logger.error(f"Error: {e}")
                    if idx == 0:
                        logger.error("Error in first frame")
                        raise e
                    else:
                        return self.get_item_keys(0, keys)

        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v).to(torch.float32)

        return out

    def _normalize_keys_arg(self, keys):
        """
        Normalize keys argument:
          None -> all dataset keys
          str  -> single key
          Iterable[str] -> list of keys
        """
        if keys is None:
            return list(self.key_map.keys())
        if isinstance(keys, str):
            return [keys]
        if isinstance(keys, Iterable):
            return list(keys)
        raise TypeError(f"keys must be None, str, or iterable[str], got {type(keys)}")


class ZarrEpisode:
    """
    Lightweight wrapper around a single Zarr episode store.
    Designed for efficient PyTorch DataLoader usage with direct store access.
    """

    __slots__ = (
        "_path",
        "_store",
        "metadata",
        "keys",
    )

    def __init__(self, path: str | Path):
        """
        Initialize ZarrEpisode wrapper.
        Args:
            path: Path to the .zarr episode directory
        """
        self._path = Path(path)
        self._store = zarr.open_group(str(self._path), mode="r")
        self.metadata = dict(self._store.attrs)
        self.keys = self.metadata["features"]

    def read(
        self, keys_with_ranges: dict[str, tuple[int, int | None]]
    ) -> dict[str, np.ndarray]:
        """
        Read data for specified keys, each with their own index or range.
        Args:
            keys_with_ranges: Dictionary mapping keys to (start, end) tuples.
                - start: Starting frame index
                - end: Ending frame index (exclusive). If None, reads single frame at start.
        Returns:
            Dictionary mapping keys to numpy arrays
        Example:
            >>> episode.read({
            ...     "obs/image": (0, 10),      # Read frames 0-10
            ...     "actions": (5, 15),        # Read frames 5-15
            ...     "rewards": (20, None),     # Read single frame at index 20
            ... })
        """
        result = {}
        for key, (start, end) in keys_with_ranges.items():
            try:
                arr = self._store[key]
            except (KeyError, FileNotFoundError):
                # Some EVA exports omit gripper arrays; key_map may still request them.
                # Zarr v3 can surface FileNotFoundError for missing nodes.
                _grip = {
                    "left.gripper",
                    "right.gripper",
                    "left.obs_gripper",
                    "right.obs_gripper",
                    "left.cmd_gripper",
                    "right.cmd_gripper",
                }
                if key not in _grip:
                    raise
                if end is not None:
                    n = max(0, int(end) - int(start))
                    data = np.zeros((n, 1), dtype=np.float32)
                else:
                    # Match real zarr reads: one frame as length-1 vector (not 0-D), for ConcatKeys.
                    data = np.zeros((1,), dtype=np.float32)
                result[key] = data
                continue
            if end is not None:
                data = arr[start:end]
            else:
                # Single frame read - use slicing to avoid 0D array issues with VariableLengthBytes
                # arr[start:start+1] gives us a 1D array, then [0] extracts the actual object
                data = arr[start : start + 1][0]
            result[key] = data
        return result

    def _collect_keys(self) -> list[str]:
        """
        Collect all array keys from the store.
        Returns:
            List of array keys (flat structure with dot-separated names)
        """
        if isinstance(self.keys, dict):
            return list(self.keys.keys())
        return list(self.keys)

    def __len__(self) -> int:
        """
        Get total number of frames in the episode.
        Returns:
            Number of frames
        """
        return self.metadata["total_frames"]

    def __repr__(self) -> str:
        """String representation of the episode."""
        return f"ZarrEpisode(path={self._path}, frames={len(self)})"
