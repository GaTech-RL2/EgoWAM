import ast
import json
import logging
import math
import os
import random
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from egowam.rldb.embodiment.embodiment import get_embodiment_id
from egowam.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    CartesianWithGripperCoordinateTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePointTrajectory,
    InterpolatePose,
    NormalizeFlowDelta,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    QuaternionPoseToYPR,
    XYZWXYZ_to_XYZYPR,
)
from egowam.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset

logger = logging.getLogger(__name__)

FLOW_ZARR_KEYS = frozenset({"future.observations.flow.front_img_1"})
FLOW_KEY_NAMES = frozenset(
    {
        "future_flow",
        # New point-conditioned flow keys + their sidecars (current_points / validity_mask / sampled_uv).
        # Excluded from generic quantile normalization (handled by NormalizeFlowDelta
        # via the loader-side flow_stats registry). Single application point contract.
        "future_flow_3d_points",
        "future_flow_3d_points__current_points",
        "future_flow_3d_points__validity_mask",
        "future_flow_3d_points__sampled_uv",
        "future_flow_2d_points",
        "future_flow_2d_points__current_points",
        "future_flow_2d_points__validity_mask",
        "future_flow_2d_points__sampled_uv",
    }
)


def set_global_seed(seed: int = 42):
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


# Quantile-normalization robustness (see normalize_data / unnormalize_data):
# floor the (q99-q1) scale and clamp the normalized output so a near-constant dimension
# (e.g. a gripper held closed through the future horizon, q99-q1 ~ 2e-4) can't turn rare
# excursions into ~1e3 targets and explode the loss. No-op for healthy dims (scale >= floor).
_NORM_MIN_SCALE = 0.1
_NORM_CLAMP = 10.0


class DataSchematic(object):
    def __init__(self, schematic_dict, viz_img_key, norm_mode="zscore"):
        """
        Initialize with a schematic dictionary and create a DataFrame.


        Args:
            schematic_dict:
                {embodiment_name}:
                    front_img_1_line:
                        key_type: camera_keys
                        zarr_key: front_img_1
                    right_wrist_img:
                        key_type: camera_keys
                        zarr_key: right_wrist_img
                    joint_positions:
                        key_type: proprio_keys
                        zarr_key: joint_positions
                    actions_joints_act:
                        key_type: action_keys
                        zarr_key: actions_joints
                    .
                    .
                    .
                    .


        Attributes:
            df (pd.DataFrame): Columns include 'key_name', 'key_type', and 'shape', 'embodiment'.
        """

        rows = []
        self.embodiments = set()

        for embodiment, schematic in schematic_dict.items():
            embodiment_id = get_embodiment_id(embodiment)
            self.embodiments.add(embodiment_id)
            for key_name, key_info in schematic.items():
                rows.append(
                    {
                        "key_name": key_name,
                        "key_type": key_info["key_type"],
                        "zarr_key": key_info["zarr_key"],
                        "shape": None,
                        "embodiment": embodiment_id,
                    }
                )

        self.df = pd.DataFrame(rows)
        self._viz_img_key = {get_embodiment_id(k): v for k, v in viz_img_key.items()}
        self.shapes_infered = False
        self.norm_mode = norm_mode
        self.norm_stats = {emb: {} for emb in self.embodiments}

    def is_flow_key(self, key_name: str, embodiment: int) -> bool:
        zarr_key = self.keyname_to_zarr_key(key_name, embodiment)
        return key_name in FLOW_KEY_NAMES or zarr_key in FLOW_ZARR_KEYS

    def zarr_key_to_keyname(self, zarr_key, embodiment):
        """
        Get the key name from the zarr key.


        Args:
            zarr_key (str): zarr key, e.g., "front_img_1".
            embodiment (int): int id corresponding to embodiment




        Returns:
            str: Key name, e.g., "front_img_1".
        """
        df_filtered = self.df[
            (self.df["zarr_key"] == zarr_key) & (self.df["embodiment"] == embodiment)
        ]

        if df_filtered.empty:
            return None

        return df_filtered["key_name"].item()

    def keyname_to_zarr_key(self, key_name, embodiment):
        """
        Get the zarr key from the key name.


        Args:
            key_name (str): Key name, e.g., "front_img_1_line".
            embodiment (int): int id corresponding to embodiment


        Returns:
            str: zarr key, e.g., "front_img_1".
        """
        df_filtered = self.df[
            (self.df["key_name"] == key_name) & (self.df["embodiment"] == embodiment)
        ]

        if df_filtered.empty:
            return None
        return df_filtered["zarr_key"].item()

    def infer_shapes_from_batch(self, batch):
        """
        Update shapes in the DataFrame based on a batch.


        Args:
            batch (dict): Maps key names (str) to tensors with shapes, e.g.,
                {"key": tensor of shape (3, 480, 640, 3)}.


        Updates:
            The 'shape' column in the DataFrame is updated to match the inferred shapes (stored as tuples).
        """
        for key, tensor in batch.items():
            if hasattr(tensor, "shape"):
                shape = tuple(tensor.shape)
            elif isinstance(tensor, int):
                shape = (1,)
            else:
                shape = None
            if key in self.df["zarr_key"].values:
                self.df.loc[self.df["zarr_key"] == key, "shape"] = str(shape)

        self.shapes_infered = True

    def infer_norm_from_dataset(
        self,
        dataset,
        dataset_name,
        sample_frac: float = 0.10,
        seed: int = 42,
        max_samples: int | None = None,
        batch_size: int = 512,
        num_workers: int = 10,
        multi_thread: bool = True,
        benchmark_dir: str | None = None,
    ):
        """
        Args:
            norm_dataset: the dataset to infer norm from (should not have image keys to increase performance)
            dataset_name: the name of the dataset
            sample_frac: the fraction of the dataset to sample
            seed: the seed for the random number generator
            max_samples: the maximum number of samples to use
            log_every: the number of samples to log after
        """
        embodiment = dataset_name
        if isinstance(embodiment, str):
            embodiment = get_embodiment_id(embodiment)

        benchmark_stats = None
        if benchmark_dir is not None:
            os.makedirs(benchmark_dir, exist_ok=True)
            benchmark_file = os.path.join(benchmark_dir, "benchmark.json")
            benchmark_stats = {}
            benchmark_stats["stats"] = {}

        norm_keys = []
        norm_keys.extend(self.keys_of_type("proprio_keys", embodiment))
        norm_keys.extend(self.keys_of_type("action_keys", embodiment))
        norm_keys = [key for key in norm_keys if not self.is_flow_key(key, embodiment)]
        if len(norm_keys) == 0:
            logger.warning(
                f"[NormStats] No proprio/action keys for embodiment={embodiment}"
            )
            return

        N = len(dataset)
        if N <= 0:
            raise ValueError("Dataset is empty")

        n_samples = int(math.ceil(sample_frac * N))
        n_samples = max(1, min(n_samples, N))
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        # Resolve once to avoid repeated DataFrame lookups in the inner loop.
        key_to_batch_key = {}
        for key in norm_keys:
            batch_key = self.keyname_to_zarr_key(key, embodiment)
            if batch_key is not None:
                key_to_batch_key[key] = batch_key

        if len(key_to_batch_key) == 0:
            logger.warning(
                f"[NormStats] No zarr key mapping found for embodiment={embodiment}"
            )
            return

        logger.info(f"[NormStats] embodiment={embodiment} norm_keys={norm_keys}")
        logger.info(
            f"[NormStats] sampling {n_samples}/{N} (~{100 * sample_frac:.1f}%) indices"
        )

        # Streaming accumulators:
        # - exact mean/std/min/max via sum/sum_sq/min/max
        # - exact quantiles/median via per-key value buffers concatenated at finalize
        accum = {
            k: {
                "count": 0,
                "sum": None,
                "sum_sq": None,
                "min": None,
                "max": None,
                "quantile_values": [],
                "is_spatial": False,
                "broadcast_shape": None,
                "feature_shape": None,
            }
            for k in norm_keys
        }

        def _project_values_for_stats(x: np.ndarray):
            """
            Project input tensor to a 2D matrix [N_reduce, N_features] for exact
            per-feature stats while preserving old broadcasting behavior.
            """
            is_spatial = x.ndim == 4
            if is_spatial:
                # [N, C, H, W] if channel dim is the smallest among non-batch dims.
                if x.shape[1] < x.shape[2] and x.shape[1] < x.shape[3]:
                    values = np.moveaxis(x, 1, -1).reshape(-1, x.shape[1])
                    return values, True, (-1, 1, 1), (x.shape[1],)
                # Otherwise treat as [N, H, W, C].
                values = x.reshape(-1, x.shape[-1])
                return values, True, (1, 1, -1), (x.shape[-1],)

            if x.ndim == 0:
                values = x.reshape(1, 1)
                return values, False, None, ()

            values = x.reshape(x.shape[0], -1)
            feature_shape = tuple(x.shape[1:])
            return values, False, None, feature_shape

        key_map_patches = self._temporarily_patch_key_maps_for_norm_loading(
            dataset=dataset,
            allowed_key_names=set(norm_keys),
            allowed_batch_keys=set(key_to_batch_key.values()),
        )

        loading_start_time = time.time()
        cur_num_samples = 0
        logger.info(
            f"[NormStats] Starting to load data for norm inference with batch_size={batch_size} and num_workers={num_workers}"
        )
        try:
            sample_indices = torch.randperm(
                N, generator=torch.Generator().manual_seed(seed)
            )[:n_samples]
            sample_indices = torch.sort(sample_indices).values.tolist()
            sampled_dataset = torch.utils.data.Subset(dataset, sample_indices)

            loader_kwargs = {
                "dataset": sampled_dataset,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle": False,
            }
            if num_workers > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = 4

            loader = torch.utils.data.DataLoader(**loader_kwargs)

            total_batches = int(math.ceil(n_samples / batch_size))
            for batch in tqdm(loader, total=total_batches):
                any_tensor = next(
                    (v for v in batch.values() if isinstance(v, torch.Tensor)), None
                )
                if any_tensor is None:
                    continue
                take = int(any_tensor.shape[0])
                cur_num_samples += take

                for k, batch_key in key_to_batch_key.items():
                    if batch_key not in batch:
                        continue

                    x = batch[batch_key][:take]
                    if hasattr(x, "detach"):
                        x = x.detach().cpu().numpy()
                    else:
                        x = np.asarray(x)

                    if x.size == 0:
                        continue

                    values, is_spatial, broadcast_shape, feature_shape = (
                        _project_values_for_stats(x)
                    )
                    if values.size == 0:
                        continue

                    reduce_count = int(values.shape[0])

                    values64 = values.astype(np.float64, copy=False)
                    sumv = np.sum(values64, axis=0)
                    sumsq = np.sum(np.square(values64), axis=0)
                    minv = np.min(values64, axis=0)
                    maxv = np.max(values64, axis=0)

                    acc = accum[k]
                    if acc["count"] == 0:
                        acc["sum"] = sumv
                        acc["sum_sq"] = sumsq
                        acc["min"] = minv
                        acc["max"] = maxv
                        acc["is_spatial"] = is_spatial
                        acc["broadcast_shape"] = broadcast_shape
                        acc["feature_shape"] = feature_shape
                    else:
                        acc["sum"] += sumv
                        acc["sum_sq"] += sumsq
                        acc["min"] = np.minimum(acc["min"], minv)
                        acc["max"] = np.maximum(acc["max"], maxv)

                    acc["count"] += reduce_count
                    acc["quantile_values"].append(values.astype(np.float32, copy=False))
        finally:
            self._restore_key_maps(key_map_patches)

        norm_keys = [k for k in norm_keys if accum[k]["count"] > 0]
        if len(norm_keys) == 0:
            logger.warning(
                f"[NormStats] No valid data collected for embodiment={embodiment}"
            )
            return

        loading_end_time = time.time()
        loading_time = loading_end_time - loading_start_time
        if benchmark_stats is not None:
            benchmark_stats["loading_time"] = loading_time

        def _finalize_key(k: str):
            acc = accum[k]
            if acc["count"] <= 0:
                return None

            count = float(acc["count"])
            feature_shape = acc["feature_shape"]

            mean = (acc["sum"] / count).reshape(feature_shape)
            var = np.maximum(acc["sum_sq"] / count - np.square(acc["sum"] / count), 0.0)
            std = np.sqrt(var).reshape(feature_shape)
            minv = acc["min"].reshape(feature_shape)
            maxv = acc["max"].reshape(feature_shape)

            q_values = np.concatenate(acc["quantile_values"], axis=0)
            median = np.median(q_values, axis=0).reshape(feature_shape)
            q1 = np.percentile(q_values, 1, axis=0).reshape(feature_shape)
            q99 = np.percentile(q_values, 99, axis=0).reshape(feature_shape)

            return {
                "key": k,
                "count": int(count),
                "mean": mean,
                "std": std,
                "min": minv,
                "max": maxv,
                "median": median,
                "quantile_1": q1,
                "quantile_99": q99,
                "is_spatial": acc["is_spatial"],
                "broadcast_shape": acc["broadcast_shape"],
            }

        computing_start_time = time.time()
        reduced_stats = []
        if multi_thread and len(norm_keys) > 1:
            max_workers = max(1, min(len(norm_keys), os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_finalize_key, k): k for k in norm_keys}
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="[NormStats] Finalizing exact quantiles",
                ):
                    result = future.result()
                    if result is not None:
                        reduced_stats.append(result)
        else:
            for k in norm_keys:
                result = _finalize_key(k)
                if result is not None:
                    reduced_stats.append(result)

        for result in reduced_stats:
            k = result["key"]

            # Reshape stats for broadcasting back to 4D/5D tensors if needed
            def wrap_stat(v):
                v_torch = torch.from_numpy(np.array(v, dtype=np.float32)).float()
                if result["is_spatial"]:
                    return v_torch.view(*result["broadcast_shape"])
                return v_torch

            self.norm_stats[embodiment][k] = {
                "mean": wrap_stat(result["mean"]),
                "std": wrap_stat(result["std"]),
                "min": wrap_stat(result["min"]),
                "max": wrap_stat(result["max"]),
                "median": wrap_stat(result["median"]),
                "quantile_1": wrap_stat(result["quantile_1"]),
                "quantile_99": wrap_stat(result["quantile_99"]),
            }

            if benchmark_stats is not None:
                os.makedirs(
                    os.path.join(benchmark_dir, str(embodiment), k), exist_ok=True
                )
                mean_path = os.path.join(benchmark_dir, str(embodiment), k, "mean.pt")
                std_path = os.path.join(benchmark_dir, str(embodiment), k, "std.pt")
                min_path = os.path.join(benchmark_dir, str(embodiment), k, "min.pt")
                max_path = os.path.join(benchmark_dir, str(embodiment), k, "max.pt")
                median_path = os.path.join(
                    benchmark_dir, str(embodiment), k, "median.pt"
                )
                quantile_1_path = os.path.join(
                    benchmark_dir, str(embodiment), k, "quantile_1.pt"
                )
                quantile_99_path = os.path.join(
                    benchmark_dir, str(embodiment), k, "quantile_99.pt"
                )
                torch.save(self.norm_stats[embodiment][k]["mean"], mean_path)
                torch.save(self.norm_stats[embodiment][k]["std"], std_path)
                torch.save(self.norm_stats[embodiment][k]["min"], min_path)
                torch.save(self.norm_stats[embodiment][k]["max"], max_path)
                torch.save(self.norm_stats[embodiment][k]["median"], median_path)
                torch.save(
                    self.norm_stats[embodiment][k]["quantile_1"], quantile_1_path
                )
                torch.save(
                    self.norm_stats[embodiment][k]["quantile_99"], quantile_99_path
                )
                if benchmark_stats["stats"].get(embodiment, None) is None:
                    benchmark_stats["stats"][embodiment] = {}
                if benchmark_stats["stats"][embodiment].get(k, None) is None:
                    benchmark_stats["stats"][embodiment][k] = {}
                benchmark_stats["stats"][embodiment][k] = {
                    "mean": mean_path,
                    "std": std_path,
                    "min": min_path,
                    "max": max_path,
                    "median": median_path,
                    "quantile_1": quantile_1_path,
                    "quantile_99": quantile_99_path,
                }

            logger.info(
                f"[NormStats] key={k} samples={result['count']} stat_shape={result['mean'].shape}"
            )

        computing_end_time = time.time()
        computing_time = computing_end_time - computing_start_time
        if benchmark_stats is not None:
            benchmark_stats["computing_time"] = computing_time
            benchmark_stats["frames"] = int(cur_num_samples)

        logger.info(
            f"[NormStats] Finished norm inference, loading_time={loading_time:.2f}s, computing_time={computing_time:.2f}s"
        )
        if benchmark_stats is not None:
            with open(benchmark_file, "w") as f:
                json.dump(benchmark_stats, f, indent=4)

    def viz_img_key(self):
        """
        Get the key that should be used for offline visualization
        """
        return self._viz_img_key

    def all_keys(self):
        """
        Get all key names.


        Returns:
            list: Key names (str).
        """
        return self.df["key_name"].tolist()

    def is_key_with_embodiment(self, key_name, embodiment):
        """
        Check if a key_name exists with a given embodiment


        Args:
            key_name (str): name of key, e.g. actions_joints
            embodiment (int): integer id of embodiment


        Returns:
            bool: if the key exists.
        """
        return (
            (self.df["key_name"] == key_name) & (self.df["embodiment"] == embodiment)
        ).any()

    def keys_of_type(self, key_type, embodiment):
        """
        Get keys of a specific type.


        Args:
            key_type (str): Type of keys, e.g., "camera_keys", "proprio_keys", "action_keys", "metadata_keys".


        Returns:
            list: Key names (str) of the given type.
        """
        return self.df[
            (self.df["key_type"] == key_type) & (self.df["embodiment"] == embodiment)
        ]["key_name"].tolist()

    def action_keys(self, embodiment):
        return self.keys_of_type("action_keys", embodiment)

    def key_shape(self, key, embodiment):
        """
        Get the shape of a specific key.


        Args:
            key (str): Name of the key.
            embodiment (int): integer id of embodiment


        Returns:
            tuple or None: Shape as a tuple, or None if not found.
        """
        if key not in self.df["key_name"].values:
            raise ValueError(f"Keyname '{key}' is not in the schematic")

        df_filtered = self.df[
            (self.df["key_name"] == key) & (self.df["embodiment"] == embodiment)
        ]

        if df_filtered.empty:
            raise ValueError(f"Keyname '{key}' with embodiment {embodiment} not found.")

        shape = df_filtered["shape"].item()
        return ast.literal_eval(shape)

    def normalize_data(self, data, embodiment):
        """
        Normalize data using the stored normalization statistics.


        Args:
            data (dict): Maps key names to tensors.
                joint_positions: tensor of shape (B, S, 7)
            embodiment (int): Id of the embodiment.


        Returns:
            dict: Maps key names to normalized tensors.
        """
        if self.norm_stats is None:
            raise ValueError(
                "Normalization statistics not set. Call infer_norm_from_dataset_zarr() first."
            )

        norm_data = {}
        for key, tensor in data.items():
            if self.is_flow_key(key, embodiment):
                norm_data[key] = tensor
                continue
            if key in self.keys_of_type(
                "proprio_keys", embodiment
            ) or key in self.keys_of_type("action_keys", embodiment):
                if (
                    embodiment not in self.norm_stats
                    or key not in self.norm_stats[embodiment]
                ):
                    raise ValueError(
                        f"Missing normalization stats for key {key} and embodiment {embodiment}."
                    )

                stats = self.norm_stats[embodiment][key]
                mode = self.norm_mode

                if mode == "zscore":
                    mean = stats["mean"].to(tensor.device)
                    std = stats["std"].to(tensor.device)
                    norm_data[key] = (tensor - mean) / (std + 1e-6)
                elif mode == "minmax":
                    min_val = stats["min"].to(tensor.device)
                    max_val = stats["max"].to(tensor.device)
                    ndata = (tensor - min_val) / (max_val - min_val + 1e-6)
                    norm_data[key] = 2.0 * ndata - 1.0
                elif mode == "quantile":
                    quantile_1 = stats["quantile_1"].to(tensor.device)
                    quantile_99 = stats["quantile_99"].to(tensor.device)
                    # Floor the scale so a near-constant dim (e.g. a gripper held closed through
                    # the future horizon -> q99-q1 ~ 2e-4) can't send rare excursions to ~1e3 and
                    # blow up the loss; clamp bounds any residual. No-op for healthy dims (scale
                    # >= floor, normalized within +/-1). See _NORM_MIN_SCALE / _NORM_CLAMP.
                    scale = torch.clamp(quantile_99 - quantile_1, min=_NORM_MIN_SCALE)
                    ndata = (tensor - quantile_1) / scale
                    norm_data[key] = torch.clamp(
                        2.0 * ndata - 1.0, -_NORM_CLAMP, _NORM_CLAMP
                    )
                else:
                    raise ValueError(f"Invalid normalization mode: {mode}")
            else:
                norm_data[key] = tensor

        return norm_data

    def unnormalize_data(self, data, embodiment):
        """
        Unnormalize data using the stored normalization statistics.


        Args:
            data (dict): Maps key names to tensors.
                joint_positions: tensor of shape (B, S, 7)
            embodiment (int): Id of the embodiment.


        Returns:
            dict: Maps key names to denormalized tensors.
        """
        if self.norm_stats is None:
            raise ValueError(
                "Normalization statistics not set. Call infer_norm_from_dataset_zarr() first."
            )

        denorm_data = {}
        for key, tensor in data.items():
            if self.is_flow_key(key, embodiment):
                denorm_data[key] = tensor
                continue
            if key in self.keys_of_type(
                "proprio_keys", embodiment
            ) or key in self.keys_of_type("action_keys", embodiment):
                if (
                    embodiment not in self.norm_stats
                    or key not in self.norm_stats[embodiment]
                ):
                    raise ValueError(
                        f"Missing normalization stats for key {key} and embodiment {embodiment}."
                    )

                stats = self.norm_stats[embodiment][key]
                mode = self.norm_mode

                if mode == "zscore":
                    mean = stats["mean"].to(tensor.device)
                    std = stats["std"].to(tensor.device)
                    denorm_data[key] = tensor * (std + 1e-6) + mean

                elif mode == "minmax":
                    min_val = stats["min"].to(tensor.device)
                    max_val = stats["max"].to(tensor.device)
                    denorm_data[key] = (tensor + 1.0) * 0.5 * (
                        max_val - min_val + 1e-6
                    ) + min_val

                elif mode == "quantile":
                    quantile_1 = stats["quantile_1"].to(tensor.device)
                    quantile_99 = stats["quantile_99"].to(tensor.device)
                    # Match the floored scale used in normalize_data (consistent invert).
                    scale = torch.clamp(quantile_99 - quantile_1, min=_NORM_MIN_SCALE)
                    denorm_data[key] = (tensor + 1.0) * 0.5 * scale + quantile_1

                else:
                    raise ValueError(f"Invalid normalization mode: {mode}")
            else:
                denorm_data[key] = tensor

        return denorm_data

    @staticmethod
    def _iter_leaf_datasets(ds):
        if isinstance(ds, ZarrDataset):
            yield ds
        elif isinstance(ds, MultiDataset):
            for child in ds.datasets.values():
                yield from DataSchematic._iter_leaf_datasets(child)
        else:
            yield ds

    @staticmethod
    def _key_map_for_any(ds) -> dict:
        km = getattr(ds, "key_map", None)
        return km

    @staticmethod
    def _temporarily_patch_key_maps_for_norm_loading(
        dataset,
        *,
        allowed_key_names: set[str],
        allowed_batch_keys: set[str],
    ) -> list[tuple[object, object]]:
        patches = []
        for leaf in DataSchematic._iter_leaf_datasets(dataset):
            key_map = getattr(leaf, "key_map", None)
            if not isinstance(key_map, Mapping):
                continue

            required_key_names = DataSchematic._required_key_map_keys_for_norm_loading(
                leaf=leaf,
                desired_output_keys=allowed_key_names | allowed_batch_keys,
            )
            filtered = {}
            for key_name, spec in key_map.items():
                spec_zarr_key = (
                    spec.get("zarr_key") if isinstance(spec, Mapping) else None
                )
                if (
                    key_name in required_key_names
                    or spec_zarr_key in required_key_names
                    or key_name in allowed_key_names
                    or spec_zarr_key in allowed_batch_keys
                ):
                    filtered[key_name] = spec
                    continue
                if DataSchematic._is_norm_excluded_key_spec(key_name, spec):
                    continue
                # For safety, keep non-heavy keys only when a transform chain was
                # not fully understood and the dependency resolver fell back to the
                # leaf's original key set.
                if key_name in required_key_names:
                    filtered[key_name] = spec

            if len(filtered) != len(key_map):
                patches.append((leaf, key_map))
                leaf.key_map = filtered
        return patches

    @staticmethod
    def _restore_key_maps(patches: list[tuple[object, object]]) -> None:
        for leaf, key_map in patches:
            leaf.key_map = key_map

    @staticmethod
    def _is_norm_excluded_key_spec(key_name: str, spec) -> bool:
        if not isinstance(spec, Mapping):
            return False
        zarr_key = spec.get("zarr_key")
        key_type = spec.get("key_type")
        return (
            key_type == "camera_keys"
            or key_type == "flow_keys"
            or key_name in FLOW_KEY_NAMES
            or zarr_key in FLOW_ZARR_KEYS
        )

    @staticmethod
    def _required_key_map_keys_for_norm_loading(
        *,
        leaf,
        desired_output_keys: set[str],
    ) -> set[str]:
        required = set(desired_output_keys)
        transforms = list(getattr(leaf, "transform", None) or [])
        for transform in reversed(transforms):
            required_before = DataSchematic._required_keys_before_transform(
                transform, required
            )
            if required_before is None:
                logger.warning(
                    "[NormStats] Unknown transform %s during norm key pruning for %s; "
                    "falling back to the original key_map for this leaf",
                    type(transform).__name__,
                    getattr(leaf, "episode_path", "<unknown>"),
                )
                key_map = getattr(leaf, "key_map", None)
                if isinstance(key_map, Mapping):
                    return set(key_map.keys())
                return required
            required = required_before
        return required

    @staticmethod
    def _required_keys_before_transform(
        transform, required_after: set[str]
    ) -> set[str] | None:
        required_before = set(required_after)

        def replace_outputs(outputs: set[str], inputs: set[str]) -> None:
            if required_before & outputs:
                required_before.difference_update(outputs)
                required_before.update(inputs)

        if isinstance(transform, ActionChunkCoordinateFrameTransform):
            replace_outputs(
                {transform.transformed_key_name},
                {transform.target_world, transform.chunk_world},
            )
            return required_before

        if isinstance(transform, CartesianWithGripperCoordinateTransform):
            replace_outputs(
                {transform.transformed_key_name},
                {
                    transform.left_target_world,
                    transform.right_target_world,
                    transform.chunk_world,
                },
            )
            return required_before

        if isinstance(transform, PoseCoordinateFrameTransform):
            replace_outputs(
                {transform.transformed_key_name},
                {transform.target_world, transform.pose_world},
            )
            return required_before

        if isinstance(
            transform,
            (
                InterpolatePose,
                InterpolateLinear,
                InterpolatePointTrajectory,
                NormalizeFlowDelta,
            ),
        ):
            replace_outputs({transform.output_action_key}, {transform.action_key})
            return required_before

        if isinstance(transform, QuaternionPoseToYPR):
            replace_outputs({transform.output_key}, {transform.pose_key})
            return required_before

        if isinstance(transform, ConcatKeys):
            if transform.new_key_name in required_after:
                required_before.remove(transform.new_key_name)
                required_before.update(transform.key_list)
            if transform.delete_old_keys:
                required_before.difference_update(
                    set(transform.key_list) & set(required_after)
                )
            return required_before

        if isinstance(transform, DeleteKeys):
            required_before.difference_update(transform.keys_to_delete)
            return required_before

        if isinstance(transform, (XYZWXYZ_to_XYZYPR, NumpyToTensor)):
            return required_before

        return None

    @staticmethod
    def dataset_raw_norm_keys(
        ds,
        key_types=("proprio_keys", "action_keys"),
        extra_keys=(),
        include_all_key_map_keys=False,
    ) -> list[str]:
        out = set(extra_keys)
        for leaf in DataSchematic._iter_leaf_datasets(ds):
            km = DataSchematic._key_map_for_any(leaf)
            if include_all_key_map_keys:
                out |= set(km.keys())
            else:
                for k, info in km.items():
                    if info.get("key_type") in set(key_types):
                        out.add(k)
        return sorted(out)
