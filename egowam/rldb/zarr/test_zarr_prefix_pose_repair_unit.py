from pathlib import Path

import numpy as np
import torch

from egowam.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, ZarrDataset
from egowam.rldb.zarr.zarr_writer import ZarrWriter

BOTH_PREFIX_REPAIR = {
    "zarr_keys": ["left.cmd_ee_pose", "right.cmd_ee_pose"],
    "threshold": 0.1,
}


def _pad_pose_tail(arr: np.ndarray, total_frames: int = 60) -> np.ndarray:
    if arr.shape[0] >= total_frames:
        return arr
    tail = np.repeat(arr[-1:], total_frames - arr.shape[0], axis=0)
    return np.concatenate([arr, tail], axis=0)


def _write_pose_episode(
    episode_path: Path,
    *,
    left_cmd: np.ndarray,
    right_cmd: np.ndarray,
    left_obs: np.ndarray | None = None,
    right_obs: np.ndarray | None = None,
) -> Path:
    zeros_gripper = np.zeros((left_cmd.shape[0], 1), dtype=np.float32)
    numeric_data = {
        "left.cmd_ee_pose": left_cmd.astype(np.float32),
        "right.cmd_ee_pose": right_cmd.astype(np.float32),
        "left.gripper": zeros_gripper,
        "right.gripper": zeros_gripper.copy(),
    }
    if left_obs is not None:
        numeric_data["left.obs_ee_pose"] = left_obs.astype(np.float32)
    if right_obs is not None:
        numeric_data["right.obs_ee_pose"] = right_obs.astype(np.float32)

    return ZarrWriter.create_and_write(
        episode_path=episode_path,
        numeric_data=numeric_data,
        embodiment="eva_bimanual",
        metadata_override={"robot_name": "eva_bimanual"},
        chunk_timesteps=8,
    )


def _build_dataset(
    episode_path: Path,
    *,
    prefix_pose_repair: dict | None = None,
) -> ZarrDataset:
    return ZarrDataset(
        Episode_path=episode_path,
        key_map={
            "left.cmd_ee_pose": {"zarr_key": "left.cmd_ee_pose", "horizon": 4},
            "right.cmd_ee_pose": {"zarr_key": "right.cmd_ee_pose", "horizon": 4},
            "left.obs_ee_pose": {"zarr_key": "left.obs_ee_pose"},
            "right.obs_ee_pose": {"zarr_key": "right.obs_ee_pose"},
        },
        transform_list=None,
        prefix_pose_repair=prefix_pose_repair,
    )


def test_prefix_pose_repair_chooses_left_when_left_breaks_first(tmp_path: Path) -> None:
    left_cmd = _pad_pose_tail(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    right_cmd = _pad_pose_tail(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    left_obs = left_cmd.copy()
    right_obs = right_cmd.copy()
    episode_path = _write_pose_episode(
        tmp_path / "repair_enabled.zarr",
        left_cmd=left_cmd,
        right_cmd=right_cmd,
        left_obs=left_obs,
        right_obs=right_obs,
    )
    dataset = _build_dataset(episode_path, prefix_pose_repair=BOTH_PREFIX_REPAIR)

    sample = dataset[0]
    requested = dataset.get_item_keys(0, ["left.cmd_ee_pose", "right.cmd_ee_pose"])

    expected_left = torch.tensor(left_cmd[[2, 2, 2, 3]], dtype=torch.float32)
    expected_right = torch.tensor(right_cmd[:4], dtype=torch.float32)
    torch.testing.assert_close(sample["left.cmd_ee_pose"], expected_left)
    torch.testing.assert_close(sample["right.cmd_ee_pose"], expected_right)
    torch.testing.assert_close(requested["left.cmd_ee_pose"], expected_left)
    torch.testing.assert_close(requested["right.cmd_ee_pose"], expected_right)


def test_prefix_pose_repair_is_disabled_by_default(tmp_path: Path) -> None:
    left_cmd = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    right_cmd = left_cmd.copy()
    episode_path = _write_pose_episode(
        tmp_path / "repair_disabled.zarr",
        left_cmd=left_cmd,
        right_cmd=right_cmd,
        left_obs=left_cmd,
        right_obs=right_cmd,
    )
    dataset = _build_dataset(episode_path, prefix_pose_repair=None)

    sample = dataset[0]

    torch.testing.assert_close(
        sample["left.cmd_ee_pose"],
        torch.tensor(left_cmd[:4], dtype=torch.float32),
    )
    torch.testing.assert_close(
        sample["right.cmd_ee_pose"],
        torch.tensor(right_cmd[:4], dtype=torch.float32),
    )


def test_prefix_pose_repair_does_not_touch_non_targeted_obs_or_other_cmd_key(
    tmp_path: Path,
) -> None:
    left_cmd = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    right_cmd = left_cmd.copy()
    left_obs = left_cmd.copy()
    right_obs = right_cmd.copy()
    episode_path = _write_pose_episode(
        tmp_path / "obs_untouched.zarr",
        left_cmd=left_cmd,
        right_cmd=right_cmd,
        left_obs=left_obs,
        right_obs=right_obs,
    )
    dataset = _build_dataset(episode_path, prefix_pose_repair=BOTH_PREFIX_REPAIR)

    sample = dataset[0]

    torch.testing.assert_close(
        sample["right.cmd_ee_pose"], torch.tensor(right_cmd[:4], dtype=torch.float32)
    )
    torch.testing.assert_close(
        sample["left.obs_ee_pose"], torch.tensor(left_obs[0], dtype=torch.float32)
    )
    torch.testing.assert_close(
        sample["right.obs_ee_pose"], torch.tensor(right_obs[0], dtype=torch.float32)
    )


def test_prefix_pose_repair_no_break_leaves_values_unchanged_and_preserves_padding(
    tmp_path: Path,
) -> None:
    left_cmd = np.array(
        [
            [0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    right_cmd = left_cmd.copy()
    episode_path = _write_pose_episode(
        tmp_path / "no_break.zarr",
        left_cmd=left_cmd,
        right_cmd=right_cmd,
        left_obs=left_cmd,
        right_obs=right_cmd,
    )
    dataset = _build_dataset(episode_path, prefix_pose_repair=BOTH_PREFIX_REPAIR)

    sample = dataset[3]
    requested = dataset.get_item_keys(3, "left.cmd_ee_pose")

    expected = torch.tensor(
        np.stack([left_cmd[3], left_cmd[4], left_cmd[4], left_cmd[4]], axis=0),
        dtype=torch.float32,
    )
    torch.testing.assert_close(sample["left.cmd_ee_pose"], expected)
    torch.testing.assert_close(requested["left.cmd_ee_pose"], expected)


def test_local_episode_resolver_passes_prefix_pose_repair_dataset_option(
    tmp_path: Path,
) -> None:
    left_cmd = _pad_pose_tail(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    right_cmd = left_cmd.copy()
    _write_pose_episode(
        tmp_path / "resolver_episode.zarr",
        left_cmd=left_cmd,
        right_cmd=right_cmd,
        left_obs=left_cmd,
        right_obs=right_cmd,
    )
    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map={
            "left.cmd_ee_pose": {"zarr_key": "left.cmd_ee_pose", "horizon": 4},
        },
        transform_list=None,
        dataset_options={"prefix_pose_repair": BOTH_PREFIX_REPAIR},
    )

    resolved = resolver.resolve(filters={})
    dataset = resolved["resolver_episode"]
    sample = dataset[0]

    torch.testing.assert_close(
        sample["left.cmd_ee_pose"],
        torch.tensor(left_cmd[[2, 2, 2, 3]], dtype=torch.float32),
    )


def test_prefix_pose_repair_chooses_right_when_right_breaks_first(
    tmp_path: Path,
) -> None:
    left_cmd = _pad_pose_tail(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    right_cmd = _pad_pose_tail(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    episode_path = _write_pose_episode(
        tmp_path / "repair_right_first.zarr",
        left_cmd=left_cmd,
        right_cmd=right_cmd,
        left_obs=left_cmd,
        right_obs=right_cmd,
    )
    dataset = _build_dataset(episode_path, prefix_pose_repair=BOTH_PREFIX_REPAIR)

    sample = dataset[0]

    torch.testing.assert_close(
        sample["left.cmd_ee_pose"], torch.tensor(left_cmd[:4], dtype=torch.float32)
    )
    torch.testing.assert_close(
        sample["right.cmd_ee_pose"],
        torch.tensor(right_cmd[[2, 2, 2, 3]], dtype=torch.float32),
    )


def test_prefix_pose_repair_accepts_single_key_legacy_config() -> None:
    normalized = ZarrDataset._normalize_prefix_pose_repair(
        {
            "zarr_key": "left.cmd_ee_pose",
            "threshold": 0.1,
        }
    )
    assert normalized == {
        "zarr_keys": ["left.cmd_ee_pose"],
        "threshold": 0.1,
    }


def test_prefix_pose_break_ignores_quaternion_sign_flip() -> None:
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    break_info = ZarrDataset._find_prefix_pose_break(
        poses, threshold=0.1, min_tail_frames=0
    )

    assert break_info is not None
    assert break_info[0] == 1
    assert np.isclose(break_info[1], 0.3)


def test_prefix_pose_repair_ignores_breaks_with_less_than_50_tail_frames(
    tmp_path: Path,
) -> None:
    num_frames = 120
    left_cmd = np.zeros((num_frames, 7), dtype=np.float32)
    right_cmd = np.zeros((num_frames, 7), dtype=np.float32)
    left_cmd[71:, 0] = np.linspace(0.5, 2.9, num_frames - 71, dtype=np.float32)

    episode_path = _write_pose_episode(
        tmp_path / "short_tail_break_ignored.zarr",
        left_cmd=left_cmd,
        right_cmd=right_cmd,
        left_obs=left_cmd,
        right_obs=right_cmd,
    )
    dataset = _build_dataset(episode_path, prefix_pose_repair=BOTH_PREFIX_REPAIR)

    sample = dataset[70]
    requested = dataset.get_item_keys(70, "left.cmd_ee_pose")

    expected = torch.tensor(left_cmd[70:74], dtype=torch.float32)
    torch.testing.assert_close(sample["left.cmd_ee_pose"], expected)
    torch.testing.assert_close(requested["left.cmd_ee_pose"], expected)

    break_info = ZarrDataset._find_prefix_pose_break(left_cmd, threshold=0.1)
    assert break_info is None
