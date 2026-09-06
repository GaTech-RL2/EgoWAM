from pathlib import Path

import numpy as np
import torch

from egowam.rldb.zarr.zarr_dataset_multi import (
    ARIA_TIMESTAMP_KEY,
    ZarrDataset,
)
from egowam.rldb.zarr.zarr_writer import ZarrWriter


def _build_segmented_dataset(tmp_path: Path) -> ZarrDataset:
    values = np.arange(8, dtype=np.float32).reshape(-1, 1)
    timestamps = np.array(
        [
            0,
            33_333_333,
            66_666_666,
            99_999_999,
            2_000_000_000,
            2_033_333_333,
            2_066_666_666,
            2_099_999_999,
        ],
        dtype=np.int64,
    )

    episode_path = ZarrWriter.create_and_write(
        episode_path=tmp_path / "aria_segment_reads.zarr",
        numeric_data={"values": values, ARIA_TIMESTAMP_KEY: timestamps},
        embodiment="aria_bimanual",
        chunk_timesteps=8,
    )

    return ZarrDataset(
        Episode_path=episode_path,
        key_map={
            "values": {"zarr_key": "values", "horizon": 3},
            "future": {"zarr_key": "values", "future_offset": 2},
        },
        transform_list=None,
    )


def test_horizon_read_inside_segment_is_unchanged(tmp_path: Path) -> None:
    dataset = _build_segmented_dataset(tmp_path)

    sample = dataset[0]

    torch.testing.assert_close(
        sample["values"],
        torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
    )


def test_horizon_read_at_internal_episode_end_pads_last_frame(tmp_path: Path) -> None:
    dataset = _build_segmented_dataset(tmp_path)

    sample = dataset[2]

    torch.testing.assert_close(
        sample["values"],
        torch.tensor([[2.0], [3.0], [3.0]], dtype=torch.float32),
    )


def test_future_offset_clamps_to_segment_end(tmp_path: Path) -> None:
    dataset = _build_segmented_dataset(tmp_path)

    sample = dataset[2]

    torch.testing.assert_close(
        sample["future"],
        torch.tensor([3.0], dtype=torch.float32),
    )


def test_get_item_keys_uses_same_internal_episode_boundary_logic(
    tmp_path: Path,
) -> None:
    dataset = _build_segmented_dataset(tmp_path)

    item_keys = dataset.get_item_keys(2, ["values", "future"])

    torch.testing.assert_close(
        item_keys["values"],
        torch.tensor([[2.0], [3.0], [3.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        item_keys["future"],
        torch.tensor([3.0], dtype=torch.float32),
    )


def test_len_remains_total_frames_not_number_of_unpadded_segment_starts(
    tmp_path: Path,
) -> None:
    dataset = _build_segmented_dataset(tmp_path)

    assert len(dataset) == 8
    assert dataset.total_frames == 8
