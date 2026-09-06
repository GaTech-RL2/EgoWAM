# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
EgoVerse robot tasks dataloader.

Wraps one or two S3RLDBDataset instances (demo + optional rollout) built from
a structured ``train_datasets`` dict that mirrors egomimic/hydra_configs/data/debug.yaml.

    demo    S3RLDBDataset: is_rollout=False  → rollout_data_mask=0
    rollout S3RLDBDataset: is_rollout=True   → rollout_data_mask=1

Video sequence per sample (use_proprio=True, use_values=True, D=num_duplicates_per_image):
  (1)  blank first input frame         1 repeat
  (2)  current proprio  (blank)        D repeats   ← only if use_proprio
  (3)  current left wrist image        D repeats   ← only if key present
  (4)  current right wrist image       D repeats   ← only if key present
  (5)  current primary image           D repeats
  (6)  action chunk     (blank)        D repeats
  (7)  future proprio   (blank)        D repeats   ← only if use_proprio
  (8)  future left wrist image         D repeats   ← only if key present
  (9)  future right wrist image        D repeats   ← only if key present
  (10) future primary image            D repeats
  (11) value            (blank)        D repeats   ← only if use_values
"""

import copy
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, ConcatDataset

from egomimic.rldb.utils import S3RLDBDataset
from cosmos_policy.datasets.dataset_utils import preprocess_image, resize_images


class EgoVerseDataset(Dataset):
    """
    Single-sample wrapper around S3RLDBDataset that returns cosmos-policy format items.

    Configured exclusively via a ``train_datasets`` dict, e.g.::

        train_datasets = {
            "demo": {
                "embodiment": "eva_bimanual", "mode": "total",
                "filters": {"episode_hash": "..."}, "action_chunk": 25,
            },
            "rollout": {
                "embodiment": "eva_bimanual", "mode": "total",
                "filters": {"episode_hash": "..."}, "action_chunk": 25,
                "success": True, "p_world_model": 0.5,
            },
        }

    Action and proprio are optionally min-max normalised to [-1, +1] using
    statistics computed from the demo split.
    """

    def __init__(
        self,
        # ── Shared infrastructure (defaults for all sub-datasets) ─────────────
        embodiment: str = None,
        main_prefix: str = "processed_v2",
        temp_root: str = "/coc/flash7/scratch/egowm/egoverseS3Dataset",
        cache_root: str = "/coc/flash7/scratch/.cache",
        wm_root: str = "/coc/flash7/scratch/egowm/wmprocessedDataset",
        # ── Cosmos-policy rendering params ────────────────────────────────────
        chunk_size: int = 25,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        use_proprio: bool = True,
        use_values: bool = True,
        num_duplicates_per_image: int = 4,
        debug: bool = False,
        # ── Camera / state key names ───────────────────────────────────────────
        primary_camera_key: str = "observations.images.front_img_1",
        left_wrist_camera_key: str = "observations.images.left_wrist_img",
        right_wrist_camera_key: str = "observations.images.right_wrist_img",
        proprio_key: str = "observations.state.ee_pose",
        action_key: str = "actions_cartesian",
        # ── Structured per-dataset config (mirrors debug.yaml train_datasets) ─
        train_datasets: dict = None,
        **kwargs,
    ):
        """
        Args:
            train_datasets: Required. Dict mapping slot names to sub-dataset config dicts.
                "demo"    → S3RLDBDataset with is_rollout=False
                "rollout" → S3RLDBDataset with is_rollout=True
                Each entry may contain any S3RLDBDataset kwarg (embodiment, mode,
                bucket_name, filters, action_chunk, success, p_world_model, …).
                Shared infrastructure params (temp_root, cache_root, …) are
                filled in as defaults from the top-level args above.
            normalize_actions: Min-max normalise action chunks to [-1,+1] using demo split stats.
            normalize_proprio: Min-max normalise proprio vectors to [-1,+1] using demo split stats.
        """
        if train_datasets is None:
            raise ValueError("train_datasets must be provided.")

        # ── Store rendering / normalisation hyper-parameters ──────────────────
        self.chunk_size             = chunk_size
        self.final_image_size       = final_image_size
        self.normalize_images       = normalize_images
        self.normalize_actions      = normalize_actions
        self.normalize_proprio      = normalize_proprio
        self.use_image_aug          = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio            = use_proprio
        self.use_values             = use_values
        self.num_duplicates_per_image = num_duplicates_per_image
        self.debug                  = debug

        # Rollout-specific defaults; overwritten by the "rollout" sub-cfg below
        self.rollout_success = True
        self.p_world_model   = 0.5

        # Camera / state key names
        self.primary_camera_key     = primary_camera_key
        self.left_wrist_camera_key  = left_wrist_camera_key
        self.right_wrist_camera_key = right_wrist_camera_key
        self.proprio_key            = proprio_key
        self.action_key             = action_key

        # ── Helper: normalise a filter dict (handles OmegaConf objects) ───────
        def _clean(f):
            if f is None:
                return {}
            try:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(f):
                    f = OmegaConf.to_container(f, resolve=True)
            except ImportError:
                pass
            return dict(f)

        # ── Helper: parse a sub-dataset config, fill shared defaults ──────────
        def _parse_cfg(raw):
            cfg = dict(raw)
            cfg.pop("_target_", None)
            # Map action_chunk ↔ chunk_size
            if "action_chunk" in cfg:
                cfg.setdefault("chunk_size", cfg["action_chunk"])
            elif "chunk_size" in cfg:
                cfg["action_chunk"] = cfg["chunk_size"]
            # Fill shared infrastructure defaults
            for k, v in {
                "temp_root":   temp_root,
                "cache_root":  cache_root,
                "wm_root":     wm_root,
                "main_prefix": main_prefix,
                "debug":       debug,
            }.items():
                cfg.setdefault(k, v)
            return cfg

        # ── Build S3RLDBDataset instances from train_datasets ─────────────────
        s3_datasets = []
        _embodiment  = embodiment

        for name, raw_cfg in train_datasets.items():
            cfg = _parse_cfg(raw_cfg)

            # Force is_rollout from slot name
            if name == "demo":
                cfg["is_rollout"] = False
            elif name == "rollout":
                cfg["is_rollout"] = True
                # Extract rollout-specific params before passing to S3RLDBDataset
                self.rollout_success = cfg.pop("success",      self.rollout_success)
                self.p_world_model   = cfg.pop("p_world_model", self.p_world_model)
                cfg["success"]       = self.rollout_success
                cfg["p_world_model"] = self.p_world_model

            # Resolve embodiment (top-level overrides per-cfg if both provided)
            if _embodiment is None:
                _embodiment = cfg.get("embodiment")
            if _embodiment is None:
                raise ValueError(
                    f"embodiment must be set as a top-level param or inside "
                    f"train_datasets['{name}']"
                )

            if "filters" in cfg:
                cfg["filters"] = _clean(cfg["filters"])

            s3_datasets.append(S3RLDBDataset(**cfg, **kwargs))

        if not s3_datasets:
            raise ValueError("train_datasets is empty — at least a 'demo' entry is required.")

        self._s3_datasets_list = s3_datasets
        self._combined         = s3_datasets[0] if len(s3_datasets) == 1 else ConcatDataset(s3_datasets)
        self.demo_dataset      = s3_datasets[0]
        self.rollout_dataset   = s3_datasets[1] if len(s3_datasets) > 1 else None
        self.s3_dataset        = self.demo_dataset   # backward-compat alias
        self.embodiment        = self.demo_dataset.embodiment

        # ── T5 text embeddings (loaded once at init, reused in __getitem__) ───
        self.t5_text_embeddings_dict = None
        self._t5_default_command     = None
        self._t5_default_embedding   = None
        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as f:
                self.t5_text_embeddings_dict = pickle.load(f)
            self._t5_default_command   = next(iter(self.t5_text_embeddings_dict))
            self._t5_default_embedding = torch.squeeze(
                self.t5_text_embeddings_dict[self._t5_default_command]
            )  # (512, 1024)

        self.unique_commands = set()

        # ── Epoch length = total samples across all sub-datasets ──────────────
        self._epoch_length  = len(self._combined)
        self._demo_count    = len(self.demo_dataset)
        self._rollout_count = len(self.rollout_dataset) if self.rollout_dataset else 0

        # ── Normalisation statistics (min-max, computed from demo split) ─────────
        self._norm_stats: dict = {}
        if (normalize_actions or normalize_proprio) and self.demo_dataset is not None:
            print("[EgoVerseDataset] Computing min-max normalisation statistics from demo split …")
            self._norm_stats = self._compute_norm_stats(
                self.demo_dataset,
                keys=(
                    ([action_key] if normalize_actions else [])
                    + ([proprio_key] if normalize_proprio else [])
                ),
            )
            for k, s in self._norm_stats.items():
                print(f"  {k}:  min={s['min'].tolist()}  max={s['max'].tolist()}")

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_norm_stats(s3_dataset, keys: list) -> dict:
        """
        Load per-dimension min/max statistics for normalisation.

        Fast path: read from ``s3_dataset.meta.stats`` (pre-computed and saved to
        ``stats.json`` by LeRobotDataset / lerobot pipeline — zero extra I/O).

        Fallback: scan the full HuggingFace dataset when ``meta.stats`` is absent
        or missing a requested key.

        Returns dict: key → {"min": Tensor(D,), "max": Tensor(D,), "rng": Tensor(D,)}
        """
        stats      = {}
        meta_stats = getattr(getattr(s3_dataset, "meta", None), "stats", None)
        hf_ds      = s3_dataset.hf_dataset

        for key in keys:
            if meta_stats and key in meta_stats:
                # ── Fast path: use pre-saved stats from stats.json ─────────────
                mn  = meta_stats[key]["min"].float().reshape(-1)   # (D,)
                mx  = meta_stats[key]["max"].float().reshape(-1)   # (D,)
                print(f"  [norm_stats] {key}: loaded from meta.stats")
            elif key in hf_ds.features:
                # ── Fallback: scan all frames ──────────────────────────────────
                print(f"  [norm_stats] {key}: meta.stats absent, scanning HF dataset …")
                hf_np = hf_ds.with_format("numpy", columns=[key])
                data  = hf_np[:][key].astype(np.float32)
                if data.ndim == 3:               # (N, T, D) chunk format
                    data = data.reshape(-1, data.shape[-1])
                elif data.ndim == 1:             # scalar per frame
                    data = data[:, np.newaxis]
                mn = torch.from_numpy(np.min(data, axis=0))
                mx = torch.from_numpy(np.max(data, axis=0))
            else:
                print(f"  [norm_stats] {key}: not found in meta.stats or HF dataset, skipping")
                continue

            rng = torch.clamp(mx - mn, min=1e-6)   # guard zero-range dims
            stats[key] = {"min": mn, "max": mx, "rng": rng}

        return stats

    def _normalise_chunk(self, arr: np.ndarray, key: str) -> np.ndarray:
        """Min-max normalise a (T, D) or (D,) array to [-1, +1]. No-op if key absent."""
        if key not in self._norm_stats:
            return arr
        mn  = self._norm_stats[key]["min"].numpy()   # (D,)
        rng = self._norm_stats[key]["rng"].numpy()   # (D,)
        return 2.0 * (arr - mn) / rng - 1.0

    # ------------------------------------------------------------------
    # Length / index helpers
    # ------------------------------------------------------------------

    def __len__(self):
        return 1 if self.debug else self._epoch_length

    def _resolve_idx(self, idx: int):
        """Return (sub_dataset, local_idx, is_rollout_sample)."""
        if not hasattr(self, "_dataset_mapping"):
            self._dataset_mapping = []
            cum = 0
            for s3_ds in self._s3_datasets_list:
                n     = len(s3_ds)
                is_ro = getattr(s3_ds, "is_rollout", False)
                self._dataset_mapping.append((cum, cum + n, s3_ds, is_ro))
                cum += n

        for start, end, sub_ds, is_ro in self._dataset_mapping:
            if start <= idx < end:
                return sub_ds, idx - start, is_ro

        # Fallback — should never happen
        return self.demo_dataset, idx % max(1, self._demo_count), False

    # ------------------------------------------------------------------
    # Image / array helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_numpy_uint8(raw) -> np.ndarray:
        """Convert a single image tensor/array to (H, W, C) uint8."""
        if isinstance(raw, torch.Tensor):
            if raw.dtype == torch.uint8:
                img_np = raw.cpu().numpy()
            elif raw.dtype in (torch.float32, torch.float64):
                img_np = raw.cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
            else:
                img_np = raw.cpu().numpy().astype(np.uint8)
        else:
            img_np = np.asarray(raw)
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        # (C, H, W) → (H, W, C)
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
        return img_np

    def _ensure_size(self, img: np.ndarray) -> np.ndarray:
        if img.shape[0] != self.final_image_size or img.shape[1] != self.final_image_size:
            return resize_images(np.expand_dims(img, axis=0), self.final_image_size).squeeze(0)
        return img

    def _to_chunk(self, arr) -> np.ndarray:
        """Trim or pad to (chunk_size, D) float32."""
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if len(arr) >= self.chunk_size:
            return arr[: self.chunk_size]
        last    = arr[-1] if len(arr) > 0 else np.zeros(arr.shape[-1], dtype=np.float32)
        padding = np.tile(last, (self.chunk_size - len(arr), 1))
        return np.concatenate([arr, padding], axis=0)

    def _get_image(self, item, key):
        raw = item.get(key)
        if raw is None:
            return None
        return self._ensure_size(self._tensor_to_numpy_uint8(raw))

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """Single-sample equivalent of egomimic's _robomimic_to_cosmos_policy_data."""
        if self.debug:
            idx = 0

        sub_dataset, local_idx, is_rollout_sample = self._resolve_idx(idx)
        item = sub_dataset[local_idx]

        # ── Camera images ──────────────────────────────────────────────────────
        primary_img            = self._get_image(item, self.primary_camera_key)
        left_wrist_img         = self._get_image(item, self.left_wrist_camera_key)
        right_wrist_img        = self._get_image(item, self.right_wrist_camera_key)
        future_primary_img     = self._get_image(item, f"future.{self.primary_camera_key}")
        future_left_wrist_img  = self._get_image(item, f"future.{self.left_wrist_camera_key}")
        future_right_wrist_img = self._get_image(item, f"future.{self.right_wrist_camera_key}")

        if primary_img is None:
            primary_img = np.zeros((self.final_image_size, self.final_image_size, 3), dtype=np.uint8)
        if future_primary_img is None:
            future_primary_img = np.zeros_like(primary_img)

        # ── Proprio ────────────────────────────────────────────────────────────
        assert self.proprio_key in item, f"Key '{self.proprio_key}' missing."
        curr_proprio = np.asarray(
            item[self.proprio_key].cpu().numpy()
            if isinstance(item[self.proprio_key], torch.Tensor)
            else item[self.proprio_key],
            dtype=np.float32,
        )
        if curr_proprio.ndim > 1:
            curr_proprio = curr_proprio[0]

        future_proprio_raw = item.get(f"future.{self.proprio_key}")
        if future_proprio_raw is not None:
            future_proprio = np.asarray(
                future_proprio_raw.cpu().numpy()
                if isinstance(future_proprio_raw, torch.Tensor)
                else future_proprio_raw,
                dtype=np.float32,
            )
            if future_proprio.ndim > 1:
                future_proprio = future_proprio[0]
        else:
            future_proprio = curr_proprio.copy()

        # ── Actions ────────────────────────────────────────────────────────────
        assert self.action_key in item, f"Key '{self.action_key}' missing."
        curr_actions   = self._to_chunk(item[self.action_key])
        future_key     = f"future.{self.action_key}"
        future_actions = self._to_chunk(item[future_key]) if future_key in item else curr_actions.copy()

        curr_action_chunk   = curr_actions[: self.chunk_size, :]
        future_action_chunk = future_actions[: self.chunk_size, :]

        # ── Normalisation (min-max → [-1, +1]) ────────────────────────────────
        raw_action_chunk = curr_action_chunk.copy()
        raw_proprio      = curr_proprio.copy()

        if self.normalize_actions:
            curr_action_chunk   = self._normalise_chunk(curr_action_chunk,   self.action_key)
            future_action_chunk = self._normalise_chunk(future_action_chunk, self.action_key)
        if self.normalize_proprio:
            curr_proprio   = self._normalise_chunk(curr_proprio[np.newaxis, :],   self.proprio_key)[0]
            future_proprio = self._normalise_chunk(future_proprio[np.newaxis, :], self.proprio_key)[0]

        # ── T5 text embeddings ─────────────────────────────────────────────────
        if self._t5_default_embedding is not None:
            command      = self._t5_default_command
            t5_embedding = self._t5_default_embedding
        else:
            command      = ""
            t5_embedding = torch.zeros(512, 1024)

        # ── Build video sequence ───────────────────────────────────────────────
        ref = copy.deepcopy(primary_img)
        frames, repeats, seg = [], [], 0

        # (1) blank first input frame
        frames.append(np.zeros_like(ref)); repeats.append(1); seg += 1

        # (2) current proprio (blank)
        if self.use_proprio:
            current_proprio_latent_idx = seg
            frames.append(np.zeros_like(ref)); repeats.append(self.num_duplicates_per_image); seg += 1
        else:
            current_proprio_latent_idx = -1

        # (3) current left wrist
        if left_wrist_img is not None:
            current_wrist_image_latent_idx = seg
            frames.append(left_wrist_img); repeats.append(self.num_duplicates_per_image); seg += 1
        else:
            current_wrist_image_latent_idx = -1

        # (4) current right wrist
        if right_wrist_img is not None:
            current_wrist_image2_latent_idx = seg
            frames.append(right_wrist_img); repeats.append(self.num_duplicates_per_image); seg += 1
        else:
            current_wrist_image2_latent_idx = -1

        # (5) current primary
        current_image_latent_idx = seg
        frames.append(primary_img); repeats.append(self.num_duplicates_per_image); seg += 1

        # (6) action chunk (blank)
        action_latent_idx = seg
        frames.append(np.zeros_like(ref)); repeats.append(self.num_duplicates_per_image); seg += 1

        # (7) future proprio (blank)
        if self.use_proprio:
            future_proprio_latent_idx = seg
            frames.append(np.zeros_like(ref)); repeats.append(self.num_duplicates_per_image); seg += 1
        else:
            future_proprio_latent_idx = -1

        # (8) future left wrist
        if future_left_wrist_img is not None:
            future_wrist_image_latent_idx = seg
            frames.append(future_left_wrist_img); repeats.append(self.num_duplicates_per_image); seg += 1
        else:
            future_wrist_image_latent_idx = -1

        # (9) future right wrist
        if future_right_wrist_img is not None:
            future_wrist_image2_latent_idx = seg
            frames.append(future_right_wrist_img); repeats.append(self.num_duplicates_per_image); seg += 1
        else:
            future_wrist_image2_latent_idx = -1

        # (10) future primary
        future_image_latent_idx = seg
        frames.append(future_primary_img); repeats.append(self.num_duplicates_per_image); seg += 1

        # (11) value (blank)
        if self.use_values:
            value_latent_idx = seg
            frames.append(np.zeros_like(ref)); repeats.append(self.num_duplicates_per_image); seg += 1
        else:
            value_latent_idx = -1

        # Sanity check
        num_segments = len(frames)
        for lname, val in (
            ("action_latent_idx",               action_latent_idx),
            ("value_latent_idx",                value_latent_idx),
            ("current_proprio_latent_idx",      current_proprio_latent_idx),
            ("current_wrist_image_latent_idx",  current_wrist_image_latent_idx),
            ("current_wrist_image2_latent_idx", current_wrist_image2_latent_idx),
            ("current_image_latent_idx",        current_image_latent_idx),
            ("future_proprio_latent_idx",       future_proprio_latent_idx),
            ("future_wrist_image_latent_idx",   future_wrist_image_latent_idx),
            ("future_wrist_image2_latent_idx",  future_wrist_image2_latent_idx),
            ("future_image_latent_idx",         future_image_latent_idx),
        ):
            if val != -1:
                assert 0 <= val < num_segments, f"{lname}={val} out of range [0, {num_segments})"

        # Preprocess and expand
        all_unique = np.stack(frames, axis=0)
        all_unique = preprocess_image(
            all_unique,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )  # (C, num_segments, H, W)
        lengths    = torch.as_tensor(repeats, dtype=torch.long, device=all_unique.device)
        all_images = torch.repeat_interleave(all_unique, lengths, dim=1)
        assert all_images.shape[1] == int(lengths.sum().item()), "T_total mismatch"

        # ── Demo / rollout masks (stamped by S3RLDBDataset, fallback below) ───
        rollout_data_mask          = int(item.get("rollout_data_mask",         int(is_rollout_sample)))
        rollout_data_success_mask  = int(item.get("rollout_data_success_mask", int(is_rollout_sample and self.rollout_success)))
        world_model_sample_mask    = int(item.get("world_model_sample_mask",   0))
        value_function_sample_mask = int(item.get("value_function_sample_mask", 0))

        global_rollout_idx = -1 if rollout_data_mask == 0 else int(
            item.get("global_rollout_idx", np.random.randint(0, 10000))
        )

        # ── Return sample dict ─────────────────────────────────────────────────
        return {
            # Video
            "video":                           all_images,
            # Actions (normalised) + raw
            "actions":                         torch.from_numpy(curr_action_chunk).float(),
            "next_action_chunk":               torch.from_numpy(future_action_chunk).float(),
            "actions_raw":                     torch.from_numpy(raw_action_chunk).float(),
            # Language
            "command":                         command,
            "t5_text_embeddings":              t5_embedding,
            "t5_text_mask":                    torch.ones(512, dtype=torch.int64),
            # Metadata
            "fps":                             torch.tensor(30.0),
            "padding_mask":                    torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size":                      self.final_image_size * torch.ones(4),
            "__key__":                         idx,
            # Proprio (normalised) + raw
            "proprio":                         torch.from_numpy(curr_proprio).float(),
            "future_proprio":                  torch.from_numpy(future_proprio).float(),
            "proprio_raw":                     torch.from_numpy(raw_proprio).float(),
            # Placeholder value returns (not computed — set return_value_function_returns in future if needed)
            "value_function_return":           1.0, # TODO: change the value
            "next_value_function_return":      1.0,
            # Demo / rollout masks
            "rollout_data_mask":               rollout_data_mask,
            "rollout_data_success_mask":       rollout_data_success_mask,
            "world_model_sample_mask":         world_model_sample_mask,
            "value_function_sample_mask":      value_function_sample_mask,
            "global_rollout_idx":              global_rollout_idx,
            # Latent indices
            "action_latent_idx":               action_latent_idx,
            "value_latent_idx":                value_latent_idx,
            "current_proprio_latent_idx":      current_proprio_latent_idx,
            "current_wrist_image_latent_idx":  current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx": current_wrist_image2_latent_idx,
            "current_image_latent_idx":        current_image_latent_idx,
            "future_proprio_latent_idx":       future_proprio_latent_idx,
            "future_wrist_image_latent_idx":   future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx":  future_wrist_image2_latent_idx,
            "future_image_latent_idx":         future_image_latent_idx,
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ds = EgoVerseDataset(
        train_datasets={
            "demo": {
                "embodiment":   "eva_bimanual",
                "mode":         "total",
                "bucket_name":  "rldb",
                "filters":      {"episode_hash": "2026-01-22-18-57-54-150000"},
                "local_files_only": True,
                "use_future":   True,
                "action_chunk": 25,
            },
            "rollout": {
                "embodiment":   "eva_bimanual",
                "mode":         "total",
                "bucket_name":  "rldb",
                "filters":      {"episode_hash": "2026-01-22-18-57-54-150000"},
                "local_files_only": True,
                "use_future":   True,
                "action_chunk": 25,
                "success":      True,
                "p_world_model": 0.5,
            },
        },
        temp_root="/coc/flash7/scratch/egowm/egoverseS3Dataset",
        cache_root="/coc/flash7/scratch/egowm/.cache",
        wm_root="/coc/flash7/scratch/egowm/wmprocessedDataset",
        t5_text_embeddings_path="/coc/flash7/scratch/egowm/wmprocessedDataset/t5_embeddings.pkl",
        chunk_size=25,
        normalize_actions=True,
        normalize_proprio=True,
        use_proprio=True,
        use_values=True,
    )
    print(f"Dataset length    : {len(ds)}")
    print(f"  demo_dataset    : {len(ds.demo_dataset)}")
    print(f"  rollout_dataset : {len(ds.rollout_dataset)}")
    s = ds[0]
    print(f"video             : {s['video'].shape}")
    print(f"actions (normed)  : mean={s['actions'].mean():.4f}")
    print(f"actions (raw)     : mean={s['actions_raw'].mean():.4f}")
    print(f"rollout_data_mask : {s['rollout_data_mask']}")
    print(f"world_model_mask  : {s['world_model_sample_mask']}")
