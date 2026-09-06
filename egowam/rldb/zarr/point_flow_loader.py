"""
Per-episode loader for the new point-conditioned flow supervision.

Each episode on disk (under aria_cup_saucer_3d_flow_track4world/ or
eva_cup_saucer_1_3d_flow_track4world/) has the following layout:

    <episode_root>/
        sampled_uv.npy                 int32   [N=1120, 2]         static pixel anchors
        sampled_points_t.npy           float32 [T, N, 3]           world-frame 3D at t
        sampled_points_tk.npy          float32 [T, N, 3]           world-frame 3D at t+k
        projected_3d_flow_2d.npy       float32 [T, 448, 640, 2]    dense 3D-projected 2D flow
        metadata.json                  small                        includes frame_skip, valid_frame_indices
        segment_metadata.json          small                        (optional)

This module is responsible for:
  (a) indexing episodes under a given root and mapping zarr episode paths -> point-flow episode dir;
  (b) memmapping the large arrays for sparse-indexed reads;
  (c) building per-frame trajectories for 2D (projected_3d_flow_2d-based) and 3D heads;
  (d) producing the per-frame NaN validity mask.

Does NOT run the InterpolatePointTrajectory resample to FLOW_CHUNK_LENGTH — that
lives in the embodiment's transform chain so flow keys share the action
temporal grid.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Source tags used in key_map specs.
POINT_FLOW_SOURCE = "point_flow"

# Per-embodiment documented frame_skip (k). Not consumed by the loader —
# the actual k is read per-episode from metadata.json. Kept here as a reference
# so callers don't rely on the mistaken values that were in the plan docstring.
POINT_FLOW_FRAME_SKIP = {
    "eva_bimanual": 45,
    "aria_bimanual": 30,
}

# Expected dense 2D flow array shape.
POINT_FLOW_2D_IMAGE_SHAPE = (448, 640)


@dataclass(frozen=True)
class PointFlowEpisode:
    """Handles to one segment's point-flow artifacts, all memmapped or small.

    For EVA the whole zarr episode is a single segment (start_frame=0,
    end_frame=num_frames). For Aria each zarr recording is split into multiple
    segments (episode0, episode1, ...) each covering [start_frame, end_frame)
    of the underlying zarr, with its own num_frames.
    """

    episode_dir: Path
    sampled_uv: np.ndarray  # [N, 2] int32
    sampled_points_t: np.memmap | np.ndarray  # [T, N, 3] float32
    sampled_points_tk: np.memmap | np.ndarray  # [T, N, 3] float32
    # Compact per-anchor 2D flow. Preferred path is a (T, N, 2) on-disk array
    # (``sampled_proj_flow_2d.npy``); if only the legacy dense
    # ``projected_3d_flow_2d.npy`` (T, H, W, 2) is available, ``_open_episode``
    # materializes the anchor-sampled slice into memory once at load time so
    # downstream callers always see the compact shape.
    sampled_flow_2d: np.memmap | np.ndarray | None  # [T, N, 2] float32
    frame_skip: int
    num_frames: int
    start_frame: int  # lower bound of zarr frame indices this segment covers
    end_frame: int  # exclusive upper bound
    # (H, W) of the image plane the flow uv / 2D projection were computed on
    # (from metadata.json "image_size"); None if absent. Used at viz time to
    # rescale anchors to the displayed frame. EVA=(448,640), LIBERO=(448,448).
    image_hw: tuple[int, int] | None = None

    @property
    def num_points(self) -> int:
        return int(self.sampled_uv.shape[0])

    def contains(self, zarr_frame_idx: int) -> bool:
        return self.start_frame <= int(zarr_frame_idx) < self.end_frame

    def local_index(self, zarr_frame_idx: int) -> int:
        return int(zarr_frame_idx) - self.start_frame


class PointFlowIndex:
    """Index over segment dirs under a single point-flow root.

    Maps source zarr episode paths -> list of segment dirs. For EVA the list
    has one entry per zarr episode; for Aria multiple segments share one zarr.
    Each segment's metadata.json records `zarr_path`, `start_frame`, `end_frame`
    so we can pick the right segment for a given frame index.
    """

    _CACHE: dict[tuple[Path, str], PointFlowEpisode] = {}
    # Per-root reverse index: {root: {canonical_zarr_path: [(start, end, dir), ...]}}
    _INDEX: dict[Path, dict[str, list[tuple[int, int, Path]]]] = {}

    @staticmethod
    def _canonical_zarr_path(path: str | Path) -> str:
        """Normalize a zarr episode path so different-but-equivalent inputs hit
        the same index entry. Strips trailing slashes and `.zarr` suffix, and
        resolves to absolute."""
        p = Path(path).expanduser()
        p = Path(str(p).rstrip("/"))
        if p.suffix == ".zarr":
            p = p.with_suffix("")
        try:
            return str(p.resolve())
        except OSError:
            return str(p)

    @classmethod
    def _build_index(cls, root: Path) -> dict[str, list[tuple[int, int, Path]]]:
        if root in cls._INDEX:
            return cls._INDEX[root]
        index: dict[str, list[tuple[int, int, Path]]] = {}
        for meta_path in sorted(root.glob("*/metadata.json")):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"[point_flow] could not read {meta_path}: {e}")
                continue
            zarr_path = meta.get("zarr_path")
            if not zarr_path:
                continue
            # Prefer segment_metadata.json's ABSOLUTE (segment_start, segment_end)
            # which maps this point-flow segment to the zarr's absolute frame
            # range. metadata.json's (start_frame, end_frame) are LOCAL to the
            # segment on Aria (always [0, num_frames)), which cannot disambiguate
            # multiple segments per zarr (each episode has its own local 0..N).
            start = None
            end = None
            seg_meta_path = meta_path.parent / "segment_metadata.json"
            if seg_meta_path.exists():
                try:
                    with open(seg_meta_path, "r") as f:
                        seg_meta = json.load(f)
                    if "segment_start" in seg_meta and "segment_end" in seg_meta:
                        start = int(seg_meta["segment_start"])
                        end = (
                            int(seg_meta["segment_end"]) + 1
                        )  # inclusive end -> exclusive
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning(f"[point_flow] could not read {seg_meta_path}: {e}")
            if start is None:
                # EVA case: single segment per zarr, absolute == local.
                start = int(meta.get("start_frame", 0))
                end = int(meta.get("end_frame", meta.get("num_frames", 0)))
            # Upstream Track4World processor writes (T-1) flow pairs for T valid
            # frames (range(T-1) loop), but segment_end reflects the full valid
            # range. Clamp `end` to the actual array length so filter coverage
            # matches what _local_idx will accept.
            num_processed = meta.get("num_processed")
            if num_processed is not None:
                end = min(int(end), start + int(num_processed))
            canonical = cls._canonical_zarr_path(zarr_path)
            entry = (start, end, meta_path.parent)
            index.setdefault(canonical, []).append(entry)
            # Also index the bare basename for convenience.
            index.setdefault(Path(zarr_path).name, []).append(entry)
        # Sort segments by start_frame so bisect/linear scan picks the first match.
        for segs in index.values():
            segs.sort(key=lambda t: t[0])
        cls._INDEX[root] = index
        num_recordings = len({k for k in index.keys() if "/" in k})
        num_segments = sum(len(v) for k, v in index.items() if "/" in k)
        logger.info(
            f"[point_flow] indexed {num_segments} segments across "
            f"{num_recordings} zarr recordings under {root}"
        )
        return index

    @classmethod
    def segments_for(
        cls, root: str | Path, zarr_path: str | Path
    ) -> list[tuple[int, int, Path]]:
        """Return all (start, end, episode_dir) segments for this zarr_path.

        Empty if the root has no coverage for this zarr (unindexed, missing on
        disk, or filtered out). Callers use this to build the set of zarr frame
        indices that have point-flow supervision.
        """
        root = Path(root).expanduser().resolve()
        index = cls._build_index(root)
        key = cls._canonical_zarr_path(zarr_path)
        segs = index.get(key) or index.get(Path(zarr_path).name) or []
        return list(segs)

    @classmethod
    def episode_dir_for(
        cls, root: str | Path, zarr_path: str | Path, *, frame_idx: int = 0
    ) -> Path:
        """Return the segment dir that owns `frame_idx` for this zarr_path."""
        root = Path(root).expanduser().resolve()
        index = cls._build_index(root)
        key = cls._canonical_zarr_path(zarr_path)
        segs = index.get(key) or index.get(Path(zarr_path).name)
        if segs:
            for start, end, ep_dir in segs:
                if start <= int(frame_idx) < end:
                    return ep_dir
            # frame_idx out of all segment ranges.
            bounds = ", ".join(f"[{s},{e})" for s, e, _ in segs)
            raise IndexError(
                f"zarr frame {frame_idx} outside covered point-flow segments "
                f"for {zarr_path} (segments: {bounds})"
            )
        # Fallback: naive root/basename layout for legacy / test fixtures.
        name = Path(zarr_path).name
        stripped = name[: -len(".zarr")] if name.endswith(".zarr") else name
        candidate = root / stripped
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"point-flow episode not found under {root} for zarr_path "
            f"{zarr_path!r}; canonical key {key!r}. Indexed {len(index)} entries."
        )

    @classmethod
    def load_episode(
        cls,
        root: str | Path,
        zarr_path: str | Path,
        *,
        load_2d: bool,
        frame_idx: int = 0,
    ) -> PointFlowEpisode:
        root = Path(root).expanduser().resolve()
        episode_dir = cls.episode_dir_for(root, zarr_path, frame_idx=frame_idx)
        cache_key = (root, str(episode_dir) + (":2d" if load_2d else ":3d"))
        if cache_key in cls._CACHE:
            return cls._CACHE[cache_key]
        episode = cls._open_episode(episode_dir, load_2d=load_2d)
        cls._CACHE[cache_key] = episode
        return episode

    @classmethod
    def _open_episode(
        cls,
        episode_dir: Path,
        *,
        load_2d: bool,
    ) -> PointFlowEpisode:
        if not episode_dir.exists():
            raise FileNotFoundError(f"point-flow episode dir not found: {episode_dir}")

        uv_path = episode_dir / "sampled_uv.npy"
        points_t_path = episode_dir / "sampled_points_t.npy"
        points_tk_path = episode_dir / "sampled_points_tk.npy"
        meta_path = episode_dir / "metadata.json"
        for p in (uv_path, points_t_path, points_tk_path):
            if not p.exists():
                raise FileNotFoundError(f"missing point-flow artifact: {p}")

        sampled_uv = np.load(uv_path)  # small, eager-load
        points_t = np.load(points_t_path, mmap_mode="r")
        points_tk = np.load(points_tk_path, mmap_mode="r")

        if points_t.shape != points_tk.shape:
            raise ValueError(
                f"sampled_points_t {points_t.shape} != sampled_points_tk "
                f"{points_tk.shape} in {episode_dir}"
            )
        T, N, D = points_t.shape
        if D != 3 or N != sampled_uv.shape[0]:
            raise ValueError(
                f"shape mismatch in {episode_dir}: uv {sampled_uv.shape}, points "
                f"{points_t.shape}"
            )

        frame_skip = cls._read_frame_skip(meta_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        # Prefer absolute segment bounds from segment_metadata.json. metadata.json's
        # start/end are LOCAL to the segment on Aria (always 0..num_frames), which
        # would collide across multiple segments per zarr. EVA has a single
        # segment per zarr so local == absolute.
        seg_meta_path = episode_dir / "segment_metadata.json"
        start_frame = None
        end_frame = None
        if seg_meta_path.exists():
            try:
                with open(seg_meta_path, "r") as f:
                    seg_meta = json.load(f)
                if "segment_start" in seg_meta and "segment_end" in seg_meta:
                    start_frame = int(seg_meta["segment_start"])
                    end_frame = int(seg_meta["segment_end"]) + 1
            except (OSError, json.JSONDecodeError):
                pass
        if start_frame is None:
            start_frame = int(meta.get("start_frame", 0))
            end_frame = int(meta.get("end_frame", T))
        # T (actual array length) is authoritative: upstream Track4World writes
        # (T-1) flow pairs for T valid frames, so segment_end can overshoot the
        # real coverage by one. Align end_frame with num_frames=T.
        end_frame = min(int(end_frame), int(start_frame) + int(T))

        sampled_flow_2d = None
        if load_2d:
            compact_path = episode_dir / "sampled_proj_flow_2d.npy"
            dense_path = episode_dir / "projected_3d_flow_2d.npy"
            if compact_path.exists():
                sampled_flow_2d = np.load(compact_path, mmap_mode="r")
                if (
                    sampled_flow_2d.ndim != 3
                    or sampled_flow_2d.shape[0] != T
                    or sampled_flow_2d.shape[1] != sampled_uv.shape[0]
                    or sampled_flow_2d.shape[2] != 2
                ):
                    raise ValueError(
                        f"unexpected sampled_proj_flow_2d shape "
                        f"{sampled_flow_2d.shape} in {episode_dir}; "
                        f"expected (T, N, 2) with T={T}, N={sampled_uv.shape[0]}"
                    )
            elif dense_path.exists():
                dense = np.load(dense_path, mmap_mode="r")
                if dense.ndim != 4 or dense.shape[0] != T or dense.shape[-1] != 2:
                    raise ValueError(
                        f"unexpected projected_3d_flow_2d shape {dense.shape} in "
                        f"{episode_dir}; expected (T, H, W, 2) with T={T}"
                    )
                # Legacy path: materialize the N anchors into memory once.
                # (T, N, 2) is ~2 MB for T=240, N=1120 — cheap vs. the (T, H, W, 2)
                # mmap we were touching per __getitem__ call.
                uv64 = sampled_uv.astype(np.int64)
                sampled_flow_2d = np.asarray(
                    dense[:, uv64[:, 1], uv64[:, 0]], dtype=np.float32
                )
            else:
                raise FileNotFoundError(
                    f"missing sampled_proj_flow_2d.npy / projected_3d_flow_2d.npy "
                    f"in {episode_dir}"
                )

        _img_size = meta.get("image_size")
        image_hw = (
            (int(_img_size[0]), int(_img_size[1]))
            if _img_size is not None and len(_img_size) == 2
            else None
        )

        return PointFlowEpisode(
            episode_dir=episode_dir,
            sampled_uv=sampled_uv,
            sampled_points_t=points_t,
            sampled_points_tk=points_tk,
            sampled_flow_2d=sampled_flow_2d,
            frame_skip=frame_skip,
            num_frames=int(T),
            start_frame=start_frame,
            end_frame=end_frame,
            image_hw=image_hw,
        )

    @staticmethod
    def _read_frame_skip(meta_path: Path) -> int:
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json missing at {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if "frame_skip" not in meta:
            raise KeyError(f"metadata.json at {meta_path} lacks 'frame_skip'")
        return int(meta["frame_skip"])


# ---------------------------------------------------------------------------
# Trajectory construction (native k steps; resampled later by InterpolatePointTrajectory)
# ---------------------------------------------------------------------------


def _local_idx(episode: PointFlowEpisode, zarr_frame_idx: int) -> int:
    """Map zarr frame index -> point-flow array index. For single-segment EVA
    episodes start_frame=0 so this is a no-op; for Aria it subtracts the
    segment's start_frame."""
    li = int(zarr_frame_idx) - episode.start_frame
    if li < 0 or li >= episode.num_frames:
        raise IndexError(
            f"zarr frame {zarr_frame_idx} maps to local index {li} outside "
            f"segment [{episode.start_frame},{episode.end_frame}) with "
            f"num_frames={episode.num_frames} under {episode.episode_dir}"
        )
    return li


def build_3d_endpoint_trajectory(
    episode: PointFlowEpisode, frame_idx: int
) -> np.ndarray:
    """Return [k, N, 3] linear interp from P_t -> P_tk for this frame (native-k).
    `frame_idx` is a zarr-frame index; translated to segment-local internally."""
    li = _local_idx(episode, frame_idx)
    P_t = np.asarray(episode.sampled_points_t[li], dtype=np.float32)
    P_tk = np.asarray(episode.sampled_points_tk[li], dtype=np.float32)
    return _linear_endpoint_chunk(P_t, P_tk, k=episode.frame_skip)


def build_2d_endpoint_trajectory(
    episode: PointFlowEpisode, frame_idx: int
) -> np.ndarray:
    """Return [k, N, 2] linear interp from U_t -> U_tk using the baked 2D flow.
    `frame_idx` is a zarr-frame index; translated to segment-local internally."""
    if episode.sampled_flow_2d is None:
        raise RuntimeError(
            f"sampled_flow_2d not loaded for {episode.episode_dir}; "
            f"call load_episode with load_2d=True"
        )
    li = _local_idx(episode, frame_idx)
    U_t = episode.sampled_uv.astype(np.float32)
    flow_at_uv = np.asarray(episode.sampled_flow_2d[li], dtype=np.float32)  # [N, 2]
    U_tk = U_t + flow_at_uv
    return _linear_endpoint_chunk(U_t, U_tk, k=episode.frame_skip)


def build_threshold_mask(
    episode: PointFlowEpisode,
    frame_idx: int,
    *,
    dim: int,
    threshold: float,
) -> np.ndarray:
    """Per-point validity mask based on minimum-movement thresholding.

    True where the point's endpoint-to-endpoint displacement magnitude
    (``||P_tk - P_t||_2`` for 3D, ``||U_tk - U_t||_2`` for 2D) is >= threshold
    AND both endpoints are finite. Non-finite entries automatically fail the
    comparison (NaN compares False), so this subsumes the old finite-only
    check with threshold=0.0 as a special case.

    Purpose: suppress near-stationary background points that would otherwise
    inject noise into both the normalization stats and the flow head's loss.
    """
    li = _local_idx(episode, frame_idx)
    if dim == 3:
        P_t = np.asarray(episode.sampled_points_t[li], dtype=np.float32)
        P_tk = np.asarray(episode.sampled_points_tk[li], dtype=np.float32)
        disp = P_tk - P_t
    elif dim == 2:
        if episode.sampled_flow_2d is None:
            raise RuntimeError(
                f"2D threshold mask requires load_2d=True for {episode.episode_dir}"
            )
        disp = np.asarray(episode.sampled_flow_2d[li], dtype=np.float32)  # [N, 2]
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    mag = np.linalg.norm(disp, axis=-1)
    # NaN in disp -> NaN in mag -> `>= threshold` returns False. Cast to bool
    # explicitly so callers get a stable dtype.
    return (mag >= float(threshold)).astype(np.bool_)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _linear_endpoint_chunk(P_t: np.ndarray, P_tk: np.ndarray, *, k: int) -> np.ndarray:
    """Linear interp (k, N, D) between two endpoints."""
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    steps = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None, None]
    return P_t[None] + steps * (P_tk - P_t)[None]
