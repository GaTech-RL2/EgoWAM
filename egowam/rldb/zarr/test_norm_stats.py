import logging

import numpy as np
import torch

from egowam.rldb.embodiment.embodiment import get_embodiment_id
from egowam.rldb.zarr.action_chunk_transforms import ConcatKeys
from egowam.rldb.zarr.utils import DataSchematic
from egowam.rldb.zarr.zarr_dataset_multi import MultiDataset

logger = logging.getLogger(__name__)


class _LegacyDataSchematic(DataSchematic):
    def infer_norm_from_dataset_legacy(self, dataset):
        """
        dataset: huggingface dataset or zarr dataset
        returns: dictionary of means and stds for proprio and action keys
        """
        norm_columns = []

        embodiment = dataset.embodiment
        if isinstance(embodiment, str):
            embodiment = get_embodiment_id(embodiment)

        norm_columns.extend(self.keys_of_type("proprio_keys", embodiment))
        norm_columns.extend(self.keys_of_type("action_keys", embodiment))

        logger.info(
            f"[NormStats] Starting norm inference for embodiment={embodiment}, "
            f"{len(norm_columns)} columns"
        )

        def get_zarr_data(ds, col):
            if hasattr(ds, "episode_reader"):
                if col in ds.episode_reader._store:
                    return ds.episode_reader._store[col][:]
                return None
            if hasattr(ds, "datasets"):
                data_list = []
                for child_dataset in ds.datasets.values():
                    res = get_zarr_data(child_dataset, col)
                    if res is not None:
                        data_list.append(res)
                if data_list:
                    return np.concatenate(data_list, axis=0)
            return None

        for column in norm_columns:
            if not self.is_key_with_embodiment(column, embodiment):
                continue
            column_name = self.keyname_to_zarr_key(column, embodiment)
            logger.info(f"[NormStats] Processing column={column_name}")

            column_data = get_zarr_data(dataset, column_name)

            if column_data is None:
                logger.warning(
                    f"Skipping {column_name}, data not found given dataset type"
                )
                continue

            if column_data.ndim not in (2, 3):
                raise ValueError(
                    f"Column {column} has shape {column_data.shape}, "
                    "expected 2 or 3 dims"
                )

            mean = np.mean(column_data, axis=0)
            std = np.std(column_data, axis=0)
            minv = np.min(column_data, axis=0)
            maxv = np.max(column_data, axis=0)
            median = np.median(column_data, axis=0)
            q1 = np.percentile(column_data, 1, axis=0)
            q99 = np.percentile(column_data, 99, axis=0)

            self.norm_stats[embodiment][column] = {
                "mean": torch.from_numpy(mean).float(),
                "std": torch.from_numpy(std).float(),
                "min": torch.from_numpy(minv).float(),
                "max": torch.from_numpy(maxv).float(),
                "median": torch.from_numpy(median).float(),
                "quantile_1": torch.from_numpy(q1).float(),
                "quantile_99": torch.from_numpy(q99).float(),
            }

        logger.info("[NormStats] Finished norm inference")


class _DummyNormDataset:
    def __init__(self) -> None:
        self.embodiment = "eva_bimanual"

        obs_base = np.arange(14, dtype=np.float32)
        action_dim = np.arange(14, dtype=np.float32).reshape(1, 14)
        action_time = np.arange(100, dtype=np.float32).reshape(100, 1)

        ee_pose = np.stack([obs_base + i for i in range(4)], axis=0)
        actions = np.stack(
            [action_time + action_dim + i for i in range(4)],
            axis=0,
        )
        flow = np.zeros((4, 2, 28, 40), dtype=np.float32)
        flow[:, 0] = 0.25
        flow[:, 1] = -0.5

        self.episode_reader = type(
            "_EpisodeReader",
            (),
            {
                "_store": {
                    "observations.state.ee_pose": ee_pose,
                    "actions_cartesian": actions,
                    "future.observations.flow.front_img_1": flow,
                }
            },
        )()

    def __len__(self) -> int:
        return self.episode_reader._store["observations.state.ee_pose"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "observations.state.ee_pose": self.episode_reader._store[
                "observations.state.ee_pose"
            ][idx],
            "actions_cartesian": self.episode_reader._store["actions_cartesian"][idx],
            "future.observations.flow.front_img_1": self.episode_reader._store[
                "future.observations.flow.front_img_1"
            ][idx],
        }


class _ArrayEpisodeDataset:
    def __init__(self, ee_pose: np.ndarray) -> None:
        self.embodiment = "eva_bimanual"
        self.ee_pose = np.asarray(ee_pose, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.ee_pose.shape[0])

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {"observations.state.ee_pose": self.ee_pose[idx]}


class _TrackingKeyMapDataset:
    def __init__(self) -> None:
        self.embodiment = "eva_bimanual"
        self.loaded_key_names: list[tuple[str, ...]] = []
        self.key_map = {
            "observations.state.ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.ee_pose",
            },
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
            "observations.images.front_img_1": {
                "key_type": "camera_keys",
                "zarr_key": "observations.images.front_img_1",
            },
            "future_flow": {
                "key_type": "flow_keys",
                "zarr_key": "future.observations.flow.front_img_1",
            },
        }
        self._store = {
            "observations.state.ee_pose": np.arange(12, dtype=np.float32).reshape(4, 3),
            "actions_cartesian": np.arange(24, dtype=np.float32).reshape(4, 2, 3),
            "observations.images.front_img_1": np.zeros((4, 3, 8, 8), dtype=np.float32),
            "future_flow": np.ones((4, 2, 8, 8), dtype=np.float32),
        }

    def __len__(self) -> int:
        return int(self._store["observations.state.ee_pose"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        self.loaded_key_names.append(tuple(self.key_map.keys()))
        return {key: self._store[key][idx] for key in self.key_map}


class _TrackingTransformDataset:
    def __init__(self) -> None:
        self.embodiment = "eva_bimanual"
        self.loaded_key_names: list[tuple[str, ...]] = []
        self.key_map = {
            "left.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_ee_pose",
            },
            "left.gripper": {
                "key_type": "action_keys",
                "zarr_key": "left.gripper",
            },
            "observations.images.front_img_1": {
                "key_type": "camera_keys",
                "zarr_key": "observations.images.front_img_1",
            },
            "future_flow": {
                "key_type": "flow_keys",
                "zarr_key": "future.observations.flow.front_img_1",
            },
        }
        self.transform = [
            ConcatKeys(
                key_list=["left.cmd_ee_pose", "left.gripper"],
                new_key_name="actions_cartesian",
                delete_old_keys=True,
            )
        ]
        cmd = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        grip = np.arange(8, dtype=np.float32).reshape(4, 2, 1)
        self._store = {
            "left.cmd_ee_pose": cmd,
            "left.gripper": grip,
            "observations.images.front_img_1": np.zeros((4, 3, 8, 8), dtype=np.float32),
            "future_flow": np.ones((4, 2, 8, 8), dtype=np.float32),
        }

    def __len__(self) -> int:
        return int(self._store["left.cmd_ee_pose"].shape[0])

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        self.loaded_key_names.append(tuple(self.key_map.keys()))
        batch = {key: self._store[key][idx] for key in self.key_map}
        for transform in self.transform:
            batch = transform.transform(batch)
        return batch


def _expected_stats(array: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "mean": torch.from_numpy(np.mean(array, axis=0)).float(),
        "std": torch.from_numpy(np.std(array, axis=0)).float(),
        "min": torch.from_numpy(np.min(array, axis=0)).float(),
        "max": torch.from_numpy(np.max(array, axis=0)).float(),
        "median": torch.from_numpy(np.median(array, axis=0)).float(),
        "quantile_1": torch.from_numpy(np.percentile(array, 1, axis=0)).float(),
        "quantile_99": torch.from_numpy(np.percentile(array, 99, axis=0)).float(),
    }


def _print_norm_stats(
    stats: dict[int, dict[str, dict[str, torch.Tensor]]], embodiment_id: int
) -> None:
    for key, key_stats in stats[embodiment_id].items():
        print(f"{key}:")
        for stat_name, value in key_stats.items():
            print(f"  {stat_name}: shape={tuple(value.shape)}")
            print(f"  {stat_name} value=\n{value}")


def test_infer_norm_from_dataset_legacy_matches_current_on_dummy_dataset() -> None:
    dataset = _DummyNormDataset()

    sample = dataset[0]
    assert sample["observations.state.ee_pose"].shape == (14,)
    assert sample["actions_cartesian"].shape == (100, 14)
    print(
        f"observations.state.ee_pose sample shape: {sample['observations.state.ee_pose'].shape}"
    )
    print(f"actions_cartesian sample shape: {sample['actions_cartesian'].shape}")

    schematic_dict = {
        "eva_bimanual": {
            "observations.state.ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.ee_pose",
            },
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
        }
    }
    viz_img_key = {"eva_bimanual": "observations.images.front_img_1"}

    legacy_schematic = _LegacyDataSchematic(schematic_dict, viz_img_key)
    current_schematic = DataSchematic(schematic_dict, viz_img_key)

    legacy_schematic.infer_norm_from_dataset_legacy(dataset)
    current_schematic.infer_norm_from_dataset(
        dataset,
        dataset.embodiment,
        sample_frac=1.0,
        seed=0,
        batch_size=len(dataset),
        num_workers=0,
    )

    embodiment_id = get_embodiment_id(dataset.embodiment)
    expected = {
        "observations.state.ee_pose": _expected_stats(
            dataset.episode_reader._store["observations.state.ee_pose"]
        ),
        "actions_cartesian": _expected_stats(
            dataset.episode_reader._store["actions_cartesian"]
        ),
    }

    print("legacy norm stats")
    _print_norm_stats(legacy_schematic.norm_stats, embodiment_id)
    print("current norm stats")
    _print_norm_stats(current_schematic.norm_stats, embodiment_id)

    assert set(legacy_schematic.norm_stats[embodiment_id]) == set(expected)
    assert set(current_schematic.norm_stats[embodiment_id]) == set(expected)

    for key, expected_stats in expected.items():
        legacy_stats = legacy_schematic.norm_stats[embodiment_id][key]
        current_stats = current_schematic.norm_stats[embodiment_id][key]

        for stat_name, expected_value in expected_stats.items():
            torch.testing.assert_close(legacy_stats[stat_name], expected_value)
            torch.testing.assert_close(current_stats[stat_name], expected_value)
            torch.testing.assert_close(
                legacy_stats[stat_name], current_stats[stat_name]
            )


def test_flow_is_excluded_from_norm_stats_and_passthrough_for_normalization() -> None:
    dataset = _DummyNormDataset()
    schematic_dict = {
        "eva_bimanual": {
            "observations.state.ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.ee_pose",
            },
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
            "future_flow": {
                "key_type": "action_keys",
                "zarr_key": "future.observations.flow.front_img_1",
            },
        }
    }
    viz_img_key = {"eva_bimanual": "observations.images.front_img_1"}
    schematic = DataSchematic(schematic_dict, viz_img_key, norm_mode="quantile")

    schematic.infer_norm_from_dataset(
        dataset,
        dataset.embodiment,
        sample_frac=1.0,
        seed=0,
        batch_size=len(dataset),
        num_workers=0,
    )

    embodiment_id = get_embodiment_id(dataset.embodiment)
    assert "future_flow" not in schematic.norm_stats[embodiment_id]
    assert "actions_cartesian" in schematic.norm_stats[embodiment_id]
    assert "observations.state.ee_pose" in schematic.norm_stats[embodiment_id]

    flow = torch.from_numpy(
        dataset.episode_reader._store["future.observations.flow.front_img_1"][:2]
    ).float()
    actions = torch.from_numpy(dataset.episode_reader._store["actions_cartesian"][:2])
    ee_pose = torch.from_numpy(
        dataset.episode_reader._store["observations.state.ee_pose"][:2]
    )
    batch = {
        "future_flow": flow.clone(),
        "actions_cartesian": actions.clone(),
        "observations.state.ee_pose": ee_pose.clone(),
    }

    normalized = schematic.normalize_data(batch, embodiment_id)
    torch.testing.assert_close(normalized["future_flow"], flow)
    assert not torch.equal(normalized["actions_cartesian"], actions)
    assert not torch.equal(normalized["observations.state.ee_pose"], ee_pose)

    denormalized = schematic.unnormalize_data(normalized, embodiment_id)
    torch.testing.assert_close(denormalized["future_flow"], flow)
    torch.testing.assert_close(
        denormalized["actions_cartesian"], actions, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        denormalized["observations.state.ee_pose"], ee_pose, atol=1e-5, rtol=1e-5
    )


def test_infer_norm_restricts_dataset_loading_to_norm_keys_and_restores_key_map() -> (
    None
):
    dataset = _TrackingKeyMapDataset()
    original_key_map = dataset.key_map

    schematic_dict = {
        "eva_bimanual": {
            "observations.state.ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.ee_pose",
            },
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
            "future_flow": {
                "key_type": "flow_keys",
                "zarr_key": "future.observations.flow.front_img_1",
            },
        }
    }
    viz_img_key = {"eva_bimanual": "observations.images.front_img_1"}
    schematic = DataSchematic(schematic_dict, viz_img_key)

    schematic.infer_norm_from_dataset(
        dataset,
        dataset.embodiment,
        sample_frac=1.0,
        seed=0,
        batch_size=2,
        num_workers=0,
    )

    assert dataset.key_map is original_key_map
    assert dataset.loaded_key_names
    assert all(
        loaded == ("observations.state.ee_pose", "actions_cartesian")
        for loaded in dataset.loaded_key_names
    )

    embodiment_id = get_embodiment_id(dataset.embodiment)
    assert set(schematic.norm_stats[embodiment_id]) == {
        "observations.state.ee_pose",
        "actions_cartesian",
    }
    expected = _expected_stats(dataset._store["observations.state.ee_pose"])
    torch.testing.assert_close(
        schematic.norm_stats[embodiment_id]["observations.state.ee_pose"]["mean"],
        expected["mean"],
    )


def test_infer_norm_keeps_transform_dependencies_but_excludes_camera_and_flow() -> None:
    dataset = _TrackingTransformDataset()
    original_key_map = dataset.key_map

    schematic_dict = {
        "eva_bimanual": {
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
            "future_flow": {
                "key_type": "flow_keys",
                "zarr_key": "future.observations.flow.front_img_1",
            },
        }
    }
    viz_img_key = {"eva_bimanual": "observations.images.front_img_1"}
    schematic = DataSchematic(schematic_dict, viz_img_key)

    schematic.infer_norm_from_dataset(
        dataset,
        dataset.embodiment,
        sample_frac=1.0,
        seed=0,
        batch_size=2,
        num_workers=0,
    )

    assert dataset.key_map is original_key_map
    assert dataset.loaded_key_names
    assert all(
        loaded == ("left.cmd_ee_pose", "left.gripper")
        for loaded in dataset.loaded_key_names
    )

    expected_actions = np.concatenate(
        [dataset._store["left.cmd_ee_pose"], dataset._store["left.gripper"]], axis=-1
    )
    embodiment_id = get_embodiment_id(dataset.embodiment)
    torch.testing.assert_close(
        schematic.norm_stats[embodiment_id]["actions_cartesian"]["mean"],
        _expected_stats(expected_actions)["mean"],
    )


def test_multi_episode_quantiles_match_a_direct_global_percentile() -> None:
    """Streaming accumulation across episodes must equal a direct percentile over all values.

    NOTE: this test previously asserted a per-episode *envelope* -- min of each episode's 1st
    percentile, max of each 99th -- which the implementation has never computed. It was added in
    the same commit (8728cdf4) that wrote the global-percentile implementation and failed from the
    start. The global reading is also what the paper describes ("maps each action dimension's 1st
    and 99th percentiles to [-1,1]"), and it is what every released checkpoint was normalized with,
    so the implementation is authoritative here and the assertion was corrected to match it.
    """
    episode_a = _ArrayEpisodeDataset(np.array([[0.0], [100.0]], dtype=np.float32))
    episode_b = _ArrayEpisodeDataset(np.array([[10.0], [20.0]], dtype=np.float32))
    dataset = MultiDataset(
        {"episode_a": episode_a, "episode_b": episode_b},
        mode="total",
        valid_ratio=0.0,
    )

    schematic_dict = {
        "eva_bimanual": {
            "observations.state.ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.ee_pose",
            },
        }
    }
    viz_img_key = {"eva_bimanual": "observations.images.front_img_1"}
    schematic = DataSchematic(schematic_dict, viz_img_key)

    schematic.infer_norm_from_dataset(
        dataset,
        "eva_bimanual",
        sample_frac=1.0,
        seed=0,
        batch_size=2,
        num_workers=0,
    )

    embodiment_id = get_embodiment_id("eva_bimanual")
    stats = schematic.norm_stats[embodiment_id]["observations.state.ee_pose"]

    all_values = np.concatenate([episode_a.ee_pose, episode_b.ee_pose], axis=0)
    expected_q1 = np.percentile(all_values, 1, axis=0)
    expected_q99 = np.percentile(all_values, 99, axis=0)

    torch.testing.assert_close(
        stats["mean"], torch.from_numpy(np.mean(all_values, axis=0)).float()
    )
    torch.testing.assert_close(
        stats["std"], torch.from_numpy(np.std(all_values, axis=0)).float()
    )
    torch.testing.assert_close(
        stats["quantile_1"], torch.from_numpy(np.asarray(expected_q1)).float()
    )
    torch.testing.assert_close(
        stats["quantile_99"], torch.from_numpy(np.asarray(expected_q99)).float()
    )
