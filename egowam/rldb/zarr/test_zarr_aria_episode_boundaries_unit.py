from pathlib import Path

import numpy as np

from egowam.rldb.zarr.zarr_dataset_multi import (
    ARIA_TIMESTAMP_KEY,
    ZarrDataset,
)
from egowam.rldb.zarr.zarr_writer import ZarrWriter


def _write_dataset(
    episode_path: Path,
    *,
    embodiment: str,
    values: np.ndarray,
    timestamps: np.ndarray | None,
    chunk_timesteps: int = 8,
) -> Path:
    numeric_data = {"values": values}
    if timestamps is not None:
        numeric_data[ARIA_TIMESTAMP_KEY] = timestamps

    return ZarrWriter.create_and_write(
        episode_path=episode_path,
        numeric_data=numeric_data,
        embodiment=embodiment,
        chunk_timesteps=chunk_timesteps,
    )


def _build_dataset(episode_path: Path) -> ZarrDataset:
    return ZarrDataset(
        Episode_path=episode_path,
        key_map={"values": {"zarr_key": "values"}},
        transform_list=None,
    )


def test_detects_internal_episode_ends_from_timestamp_gaps(tmp_path: Path) -> None:
    values = np.arange(8, dtype=np.float32).reshape(-1, 1)
    timestamps = np.array(
        [
            0,
            33_333_333,
            66_666_666,
            2_000_000_000,
            2_033_333_333,
            2_066_666_666,
            4_000_000_000,
            4_033_333_333,
        ],
        dtype=np.int64,
    )

    episode_path = _write_dataset(
        tmp_path / "aria_multi_episode.zarr",
        embodiment="aria_bimanual",
        values=values,
        timestamps=timestamps,
    )

    dataset = _build_dataset(episode_path)

    assert dataset.episode_starts == [0, 3, 6]
    assert dataset.episode_ends == [2, 5, 7]


def test_ignores_padded_frames_beyond_total_frames_when_detecting_cuts(
    tmp_path: Path,
) -> None:
    values = np.arange(5, dtype=np.float32).reshape(-1, 1)
    timestamps = np.array(
        [0, 33_333_333, 66_666_666, 2_000_000_000, 2_033_333_333],
        dtype=np.int64,
    )

    episode_path = _write_dataset(
        tmp_path / "aria_padded.zarr",
        embodiment="aria_bimanual",
        values=values,
        timestamps=timestamps,
        chunk_timesteps=8,
    )

    dataset = _build_dataset(episode_path)

    assert dataset.total_frames == 5
    assert dataset.episode_reader._store[ARIA_TIMESTAMP_KEY].shape[0] == 8
    assert dataset.episode_starts == [0, 3]
    assert dataset.episode_ends == [2, 4]


def test_non_aria_store_keeps_single_segment_even_if_timestamp_key_exists(
    tmp_path: Path,
) -> None:
    values = np.arange(6, dtype=np.float32).reshape(-1, 1)
    timestamps = np.array(
        [0, 33_333_333, 66_666_666, 2_000_000_000, 2_033_333_333, 2_066_666_666],
        dtype=np.int64,
    )

    episode_path = _write_dataset(
        tmp_path / "eva_single_segment.zarr",
        embodiment="eva_bimanual",
        values=values,
        timestamps=timestamps,
    )

    dataset = _build_dataset(episode_path)

    assert dataset.episode_starts == [0]
    assert dataset.episode_ends == [5]


def test_missing_timestamp_key_falls_back_to_single_segment(tmp_path: Path) -> None:
    values = np.arange(6, dtype=np.float32).reshape(-1, 1)

    episode_path = _write_dataset(
        tmp_path / "aria_missing_timestamps.zarr",
        embodiment="aria_bimanual",
        values=values,
        timestamps=None,
    )

    dataset = _build_dataset(episode_path)

    assert dataset.episode_starts == [0]
    assert dataset.episode_ends == [5]
