# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
run_egoverse_eval_visualize.py

Visualizes model predictions on EgoVerse training dataset episodes.

Produces one MP4 per episode with the same 3-panel (+ optional wrist row)
layout as egomimic's validation videos
(logs/.../videos/epoch_N/EVA_BIMANUAL/validation_video_0.mp4):

  Row 1:  [Current + actions (blue)]  [GT future (green)]  [WM pred (red)]
  Row 2:  [Wrist current (blue)]      [GT future wrist (green)]  [WM pred wrist (red)]

Actions are overlaid on the "Current" panel using egomimic's draw_actions:
  - Purple: predicted action chunk
  - Green:  ground-truth action chunk

Usage:
    python -m cosmos_policy.experiments.robot.egoverse.run_egoverse_eval_visualize \
        --config cosmos_predict2_2b_480p_egoverse \
        --ckpt_path /path/to/checkpoint.pt \
        --config_file cosmos_policy/config/config.py \
        --num_denoising_steps 5 \
        --save_video_path ./visualizations \
        --max_episodes 3
"""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import draccus
import imageio
import numpy as np
import torch
import tqdm
from PIL import Image as PILImage, ImageDraw as PILDraw

from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy.experiments.robot.cosmos_utils import (
    DEVICE,
    get_action,
    get_future_state_prediction,
    get_value_prediction,
    get_model,
)
from cosmos_policy.experiments.robot.robot_utils import log_message, setup_logging
from cosmos_policy.utils.utils import set_seed_everywhere

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Cosmos temporal compression factor (WAN 2.1 tokenizer)
TCF = 4


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

@dataclass
class PolicyEvalVisualizeConfig:
    # fmt: off
    # ── Model ─────────────────────────────────────────────────────────────
    config: str = "cosmos_predict2_2b_480p_egoverse"   # experiment name
    ckpt_path: str = ""                                 # local path or HF repo
    config_file: str = "cosmos_policy/config/config.py"

    # ── Inference ─────────────────────────────────────────────────────────
    suite: str = "egoverse"
    num_denoising_steps: int = 35                        # diffusion steps
    chunk_size: int = 25                                # must match dataset
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 2
    use_proprio: bool = True
    flip_images: bool = False
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True
    unnormalize_actions: bool = True   # de-normalize model predictions back to raw scale
    normalize_proprio: bool = False    # proprio is already normalized by EgoVerseDataset; do NOT re-normalize here
    randomize_seed: bool = False
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_future_state: int = 1
    use_ensemble_future_state_predictions: bool = False
    num_future_state_predictions_in_ensemble: int = 3
    future_state_ensemble_aggregation_scheme: str = "average"

    # ── Output ────────────────────────────────────────────────────────────
    save_video_path: str = "./visualizations"
    video_fps: int = 10                                 # output video fps
    panel_size: int = 224                               # H=W of each panel

    # ── Episode selection ──────────────────────────────────────────────────
    episode_indices: Optional[str] = None   # comma-separated, e.g. "0,1,2"; None=all
    max_episodes: Optional[int] = None      # cap total episodes; None=no cap
    n: Optional[int] = None                 # max frames to visualize per episode; None=all

    # ── Logging ───────────────────────────────────────────────────────────
    local_log_dir: str = "./experiments/logs"
    run_id_note: Optional[str] = None
    seed: int = 7
    deterministic: bool = True
    # fmt: on


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure (H,W,3) uint8, converting from float if needed."""
    if img is None:
        return None
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize to (h, w, 3) uint8 using PIL LANCZOS."""
    arr = np.ascontiguousarray(img).astype(np.uint8)
    pil = PILImage.fromarray(arr)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return np.array(pil.resize((w, h), PILImage.Resampling.LANCZOS))


def _labeled(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """Add a 18-px black header strip with colored text — matches egomimic."""
    arr = np.ascontiguousarray(img).astype(np.uint8)
    pil = PILImage.fromarray(arr)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    draw = PILDraw.Draw(pil)
    draw.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
    draw.text((3, 2), text, fill=color)
    return np.array(pil)


def _scale_intrinsics(intrinsics: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """
    Scale a (3, 4) camera intrinsics matrix so that the principal point and
    focal lengths match the actual image size.

    The ARIA reference intrinsics were calibrated for ~(2*cx) × (2*cy) resolution.
    We compute per-axis scale factors from the principal point coordinates.
    """
    K = np.array(intrinsics, dtype=np.float64).copy()
    ref_w = K[0, 2] * 2.0  # cx * 2 ≈ original width
    ref_h = K[1, 2] * 2.0  # cy * 2 ≈ original height
    if ref_w > 0:
        K[0, :] *= img_w / ref_w  # scale fx, cx (and the zero column)
    if ref_h > 0:
        K[1, :] *= img_h / ref_h  # scale fy, cy
    return K


def _extract_xyz(x: np.ndarray):
    """
    Extract xyz + rot from action array. Mirrors CosmosPolicy._extract_xyz.
    Supports 6, 7, 12, 14-dim actions.
    """
    d = x.shape[-1]
    if d == 6:
        return x[..., :3], x[..., 3:6]
    elif d == 7:
        return x[..., :3], x[..., 3:6]
    elif d == 12:
        return (np.concatenate([x[..., :3], x[..., 6:9]], axis=-1),
                np.concatenate([x[..., 3:6], x[..., 9:12]], axis=-1))
    elif d == 14:
        return (np.concatenate([x[..., :3], x[..., 7:10]], axis=-1),
                np.concatenate([x[..., 3:6], x[..., 10:13]], axis=-1))
    else:
        return x[..., :min(3, d)], x[..., min(3, d):min(6, d)]


def _batch_sample(
    sample: dict,
    device: torch.device,
    model_dtype: Optional[torch.dtype] = None,
) -> dict:
    """
    Add batch dim (B=1), move tensors to device, and (optionally) cast
    floating-point tensors to the model's dtype (e.g., bfloat16).
    """
    batched = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            t = v.unsqueeze(0).to(device)
            if model_dtype is not None and torch.is_floating_point(t):
                t = t.to(model_dtype)
            batched[k] = t
        elif isinstance(v, (int, float, np.integer, np.floating)):
            # Only cast floats to model_dtype; keep ints as default
            if isinstance(v, (float, np.floating)) and model_dtype is not None:
                batched[k] = torch.tensor([v], device=device, dtype=model_dtype)
            else:
                batched[k] = torch.tensor([v], device=device)
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v).unsqueeze(0).to(device)
            if model_dtype is not None and torch.is_floating_point(t):
                t = t.to(model_dtype)
            batched[k] = t
        else:
            batched[k] = v  # str, None, etc.
    return batched


# ---------------------------------------------------------------------------
# Core inference + visualization (ported from egomimic CosmosPolicy)
# ---------------------------------------------------------------------------

def decode_wm_future_images(
    model,
    generated_latent: torch.Tensor,
    orig_clean_latent_frames: torch.Tensor,
    cosmos_batch: dict,
) -> Optional[dict]:
    """
    Decode WM-predicted future images from the diffusion output latent.
    Exact port of egomimic CosmosPolicy._decode_wm_future_images.

    Returns dict with keys (all (B, H, W, 3) uint8 numpy):
        future_image_pred, future_image_gt,
        future_wrist_image_pred (optional), future_wrist_image_gt (optional)
    or None if future_image_latent_idx == -1.
    """
    def _idx(key):
        v = cosmos_batch.get(key, -1)
        return int(v.flatten()[0].item()) if isinstance(v, torch.Tensor) else int(v)

    future_image_latent_idx       = _idx("future_image_latent_idx")
    future_wrist_image_latent_idx = _idx("future_wrist_image_latent_idx")
    if future_image_latent_idx == -1:
        return None

    # Undo latent injection on non-image slots to avoid VAE artifacts
    cleaned = generated_latent.clone()
    for key in ("current_proprio_latent_idx", "action_latent_idx",
                "future_proprio_latent_idx", "value_latent_idx"):
        idx = _idx(key)
        if idx != -1:
            cleaned[:, :, idx] = orig_clean_latent_frames[:, :, idx]

    # Decode: (B, 3, T_raw, H, W) in [-1, 1]
    with torch.no_grad():
        decoded = model.decode(cleaned.float())
    decoded_u8 = (
        ((decoded + 1.0) * 127.5).clamp(0, 255)
        .permute(0, 2, 3, 4, 1).contiguous().to(torch.uint8).cpu().numpy()
    )  # (B, T_raw, H, W, 3)

    T_raw  = decoded_u8.shape[1]
    result = {}

    pred_idx = max(0, min((future_image_latent_idx - 1) * TCF + 1, T_raw - 1))
    result["future_image_pred"] = np.ascontiguousarray(decoded_u8[:, pred_idx])

    if future_wrist_image_latent_idx != -1:
        wi = max(0, min((future_wrist_image_latent_idx - 1) * TCF + 1, T_raw - 1))
        result["future_wrist_image_pred"] = np.ascontiguousarray(decoded_u8[:, wi])

    return result


def build_visualization_frame(
    current_primary: np.ndarray,
    gt_future_primary: np.ndarray,
    wm_future_primary: np.ndarray,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    camera_transforms,
    panel_size: int,
    current_left_wrist: Optional[np.ndarray] = None,
    gt_future_left_wrist: Optional[np.ndarray] = None,
    wm_future_left_wrist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build one video frame in the exact egomimic visualize_preds layout:

      Row 1: [Current + actions (blue)] [GT future (green)] [WM pred (red)]
      Row 2: [Wrist current (blue)]     [GT future wrist (green)] [WM pred wrist (red)]

    Actions are drawn with egomimic's draw_actions (Purples=pred, Greens=GT).
    """
    from egomimic.utils.egomimicUtils import draw_actions

    H, W = panel_size, panel_size

    # Scale intrinsics from their calibrated reference resolution to panel_size.
    # ARIA_INTRINSICS has cx=320, cy=240 → calibrated for ~640×480.
    # Projecting without rescaling puts all dots outside the 224×224 panel.
    scaled_intrinsics = _scale_intrinsics(camera_transforms.intrinsics, H, W)

    # ── Actions overlay on current primary (mirrors visualize_preds) ─────────
    current_vis = _resize(current_primary, H, W)
    if gt_actions is not None and pred_actions is not None and len(pred_actions) > 0:
        pred_xyz, _ = _extract_xyz(pred_actions)
        gt_xyz,   _ = _extract_xyz(gt_actions)

        d = pred_xyz.shape[-1]
        ac_type = "joints" if d in (7, 14) else "xyz"
        arm     = "right"  if d in (7, 3)  else "both"

        try:
            current_vis = draw_actions(
                current_vis, ac_type, "Purples", pred_xyz,
                camera_transforms.extrinsics, scaled_intrinsics, arm=arm,
            )
            current_vis = draw_actions(
                current_vis, ac_type, "Greens", gt_xyz,
                camera_transforms.extrinsics, scaled_intrinsics, arm=arm,
            )
        except Exception as e:
            logger.warning("draw_actions failed: %s", e)
            import traceback; traceback.print_exc()

    # ── Panel 1: current + actions (blue label) ───────────────────────────────
    p_current  = _labeled(_resize(current_vis,        H, W), "Current + actions", (180, 180, 255))
    p_gt       = _labeled(_resize(gt_future_primary,  H, W), "GT future",         (180, 255, 180))
    p_wm       = _labeled(_resize(wm_future_primary,  H, W), "WM pred",           (255, 180, 180))
    primary_row = np.concatenate([p_current, p_gt, p_wm], axis=1)  # (H, 3W, 3)

    # ── Optional wrist row ────────────────────────────────────────────────────
    if (current_left_wrist   is not None and
        gt_future_left_wrist is not None and
        wm_future_left_wrist is not None):
        pw_cur = _labeled(_resize(current_left_wrist,    H, W), "Wrist current",    (180, 180, 255))
        pw_gt  = _labeled(_resize(gt_future_left_wrist,  H, W), "Wrist GT future",  (180, 255, 180))
        pw_wm  = _labeled(_resize(wm_future_left_wrist,  H, W), "Wrist WM pred",    (255, 180, 180))
        wrist_row = np.concatenate([pw_cur, pw_gt, pw_wm], axis=1)
        return np.concatenate([primary_row, wrist_row], axis=0)   # (2H, 3W, 3)

    return primary_row  # (H, 3W, 3)


# ---------------------------------------------------------------------------
# Episode grouping
# ---------------------------------------------------------------------------

def group_episodes_from_dataset(
    dataset,
    max_episodes: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> dict:
    """
    Scan demo_dataset (S3RLDBDataset) and group item indices by episode_index.

    Stops scanning as soon as max_frames total frames have been collected,
    so --n avoids loading the entire dataset.

    Returns:
        OrderedDict: episode_id (int) → sorted list of (frame_index, global_idx)
    """
    s3_ds   = dataset.demo_dataset
    n_items = len(s3_ds) if max_frames is None else min(len(s3_ds), max_frames)
    episodes = defaultdict(list)

    log_message(f"Scanning {n_items} items into episodes...", None)
    for i in tqdm.tqdm(range(n_items), desc="Scanning episodes"):
        item   = s3_ds[i]
        ep_idx = int(item.get("episode_index", 0))
        fr_idx = int(item.get("frame_index",   i))
        episodes[ep_idx].append((fr_idx, i))

    # Sort each episode by frame_index
    for ep_idx in episodes:
        episodes[ep_idx].sort(key=lambda x: x[0])

    # Optionally cap number of episodes
    ep_ids = sorted(episodes.keys())
    if max_episodes is not None:
        ep_ids = ep_ids[:max_episodes]

    return {ep: episodes[ep] for ep in ep_ids}


# ---------------------------------------------------------------------------
# Per-episode visualizer
# ---------------------------------------------------------------------------

def visualize_episode(
    cfg: PolicyEvalVisualizeConfig,
    episode_idx: int,
    frame_pairs: list,       # list of (frame_index, global_dataset_idx)
    model,
    dataset,
    dataset_stats: dict,
    device: torch.device,
    camera_transforms,
    log_file=None,
) -> tuple:
    """
    Run inference and produce a visualization MP4 for one episode.

    For each timestep:
      1. Build observation from raw data
      2. get_action() → predicted actions and future images
      3. Compute losses (action MSE/L1, image MSE/L1)
      4. build_visualization_frame → 3-panel strip (egomimic style)
    
    Returns:
        tuple: (video_path, episode_losses_dict)
    """
    s3_ds  = dataset.demo_dataset

    os.makedirs(cfg.save_video_path, exist_ok=True)
    video_path  = os.path.join(
        cfg.save_video_path,
        f"episode_{episode_idx}_{cfg.run_id_note or 'eval'}.mp4",
    )
    writer = imageio.get_writer(video_path, fps=cfg.video_fps)

    log_message(f"Episode {episode_idx}: {len(frame_pairs)} steps → {video_path}", log_file)

    # Initialize loss accumulators
    action_mse_losses = []
    action_l1_losses = []
    image_mse_losses = []
    image_l1_losses = []
    wrist_image_mse_losses = []
    wrist_image_l1_losses = []

    for step, (_, global_idx) in enumerate(tqdm.tqdm(frame_pairs, desc=f"ep {episode_idx}")):
        try:
            # ── 1. Get sample data ─────────────────────────────────────────
            sample = dataset[global_idx]
            raw_item = s3_ds[global_idx]

            # ── 2. Build observation and run model prediction (ALOHA-consistent path) ──
            def _raw_img(key):
                v = raw_item.get(key)
                if v is None:
                    return None
                img = _to_uint8(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                return dataset._ensure_size(img)

            observation = {
                "primary_image": _raw_img(dataset.primary_camera_key),
                "left_wrist_image": _raw_img(dataset.left_wrist_camera_key),
                "right_wrist_image": _raw_img(dataset.right_wrist_camera_key),
                "proprio": (
                    sample["proprio"].cpu().numpy()
                    if isinstance(sample.get("proprio"), torch.Tensor)
                    else np.asarray(sample.get("proprio"), dtype=np.float32)
                ),
            }
            # Ensure proprio is 1-D for get_action.
            if observation["proprio"] is not None and observation["proprio"].ndim > 1:
                observation["proprio"] = observation["proprio"][0]

            task_description = sample.get("command", "")
            action_return_dict = get_action(
                cfg=cfg,
                model=model,
                dataset_stats=dataset_stats,
                obs=observation,
                task_label_or_embedding=task_description,
                seed=cfg.seed + step,
                randomize_seed=cfg.randomize_seed,
                num_denoising_steps_action=cfg.num_denoising_steps,
                generate_future_state_and_value_in_parallel=not (
                    cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                ),
            )

            pred_actions_chunk = action_return_dict["actions"]
            pred_actions = np.asarray(pred_actions_chunk, dtype=np.float32)
            if pred_actions.ndim == 3:
                pred_actions = pred_actions.squeeze(1)
            elif pred_actions.ndim == 1:
                pred_actions = pred_actions[None, :]

            # GT actions from sample (raw / un-normalized from S3RLDBDataset)
            raw_action = raw_item.get(dataset.action_key)
            if isinstance(raw_action, torch.Tensor):
                raw_action = raw_action.cpu().numpy()
            raw_action = np.asarray(raw_action, dtype=np.float32)
            if raw_action.ndim == 1:
                raw_action = raw_action[np.newaxis, :]
            gt_actions = raw_action[: cfg.chunk_size]  # (chunk_size, action_dim)

            # ── 3. Get WM image predictions (same flow as ALOHA visualize) ─
            wm_preds = action_return_dict.get("future_image_predictions", {})
            if cfg.ar_future_prediction:
                future_state_return_dict = get_future_state_prediction(
                    cfg,
                    model=model,
                    data_batch=action_return_dict["data_batch"],
                    generated_latent_with_action=action_return_dict["generated_latent"],
                    orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                    future_proprio_latent_idx=action_return_dict["latent_indices"]["future_proprio_latent_idx"],
                    future_wrist_image_latent_idx=action_return_dict["latent_indices"]["future_wrist_image_latent_idx"],
                    future_wrist_image2_latent_idx=action_return_dict["latent_indices"]["future_wrist_image2_latent_idx"],
                    future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                    future_image2_latent_idx=action_return_dict["latent_indices"]["future_image2_latent_idx"],
                    seed=cfg.seed + step,
                    randomize_seed=cfg.randomize_seed,
                    num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                    use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                    num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                    future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                )
                wm_preds = future_state_return_dict.get("future_image_predictions", wm_preds)

            current_primary  = _raw_img(dataset.primary_camera_key)
            gt_future_primary = _raw_img(f"future.{dataset.primary_camera_key}")
            current_left_wrist   = _raw_img(dataset.left_wrist_camera_key)
            gt_future_left_wrist = _raw_img(f"future.{dataset.left_wrist_camera_key}")

            if gt_future_primary is None:
                gt_future_primary = np.zeros_like(current_primary)

            # WM predicted images: (1, H, W, 3) → (H, W, 3)
            wm_primary = wm_preds.get("future_image", None)
            wm_left_wrist = wm_preds.get("future_wrist_image", None)

            if isinstance(wm_primary, torch.Tensor):
                wm_primary = wm_primary.cpu().numpy()
            if isinstance(wm_left_wrist, torch.Tensor):
                wm_left_wrist = wm_left_wrist.cpu().numpy()
            if wm_primary is not None and getattr(wm_primary, "ndim", 0) == 4:
                wm_primary = wm_primary[0]
            if wm_left_wrist is not None and getattr(wm_left_wrist, "ndim", 0) == 4:
                wm_left_wrist = wm_left_wrist[0]
            if wm_primary is None:
                wm_primary = np.zeros_like(current_primary)

            # ── 4. Compute losses ───────────────────────────────────────────
            # Action losses (compare first action of chunk)
            if len(pred_actions) > 0 and len(gt_actions) > 0:
                pred_action_first = pred_actions[0]  # (action_dim,)
                gt_action_first = gt_actions[0]    # (action_dim,)
                action_diff = pred_action_first - gt_action_first
                action_mse = np.mean(action_diff ** 2)
                action_l1 = np.mean(np.abs(action_diff))
                action_mse_losses.append(action_mse)
                action_l1_losses.append(action_l1)

            # Image losses (primary future image)
            if wm_primary is not None and gt_future_primary is not None:
                # Convert to float and normalize to [0, 1] for loss computation
                wm_primary_float = wm_primary.astype(np.float32) / 255.0
                gt_future_primary_float = gt_future_primary.astype(np.float32) / 255.0
                image_diff = wm_primary_float - gt_future_primary_float
                image_mse = np.mean(image_diff ** 2)
                image_l1 = np.mean(np.abs(image_diff))
                image_mse_losses.append(image_mse)
                image_l1_losses.append(image_l1)

            # Wrist image losses
            if wm_left_wrist is not None and gt_future_left_wrist is not None:
                wm_left_wrist_float = wm_left_wrist.astype(np.float32) / 255.0
                gt_future_left_wrist_float = gt_future_left_wrist.astype(np.float32) / 255.0
                wrist_image_diff = wm_left_wrist_float - gt_future_left_wrist_float
                wrist_image_mse = np.mean(wrist_image_diff ** 2)
                wrist_image_l1 = np.mean(np.abs(wrist_image_diff))
                wrist_image_mse_losses.append(wrist_image_mse)
                wrist_image_l1_losses.append(wrist_image_l1)

            # ── 5. Build frame (egomimic visualize_preds layout) ───────────
            frame = build_visualization_frame(
                current_primary=current_primary,
                gt_future_primary=gt_future_primary,
                wm_future_primary=wm_primary,
                gt_actions=gt_actions,
                pred_actions=pred_actions,
                camera_transforms=camera_transforms,
                panel_size=cfg.panel_size,
                current_left_wrist=current_left_wrist,
                gt_future_left_wrist=gt_future_left_wrist,
                wm_future_left_wrist=wm_left_wrist,
            )
            writer.append_data(frame)

        except Exception as e:
            logger.warning("Step %d of episode %d failed: %s", step, episode_idx, e)
            import traceback; traceback.print_exc()
            # Write a blank frame so video length stays consistent
            blank = np.zeros((cfg.panel_size * 2, cfg.panel_size * 3, 3), dtype=np.uint8)
            writer.append_data(blank)

    writer.close()
    log_message(f"Saved: {video_path}", log_file)
    
    # Compute episode-level average losses
    episode_losses = {
        "action_mse": np.mean(action_mse_losses) if action_mse_losses else 0.0,
        "action_l1": np.mean(action_l1_losses) if action_l1_losses else 0.0,
        "image_mse": np.mean(image_mse_losses) if image_mse_losses else 0.0,
        "image_l1": np.mean(image_l1_losses) if image_l1_losses else 0.0,
        "wrist_image_mse": np.mean(wrist_image_mse_losses) if wrist_image_mse_losses else 0.0,
        "wrist_image_l1": np.mean(wrist_image_l1_losses) if wrist_image_l1_losses else 0.0,
        "num_frames": len(frame_pairs),
    }
    
    return video_path, episode_losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def eval_egoverse_visualize(cfg: PolicyEvalVisualizeConfig):
    """Visualize EgoVerse model predictions on training dataset episodes."""
    assert cfg.ckpt_path, "ckpt_path must be set"
    assert cfg.config,    "config must be set"

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    device = DEVICE

    # ── Load model ────────────────────────────────────────────────────────
    log_message("Loading model...", None)
    model, cosmos_config = get_model(cfg)
    model.eval()
    model.to(device)

    # ── Load dataset from config ──────────────────────────────────────────
    log_message("Instantiating EgoVerseDataset from cosmos config...", None)
    dataset = instantiate(cosmos_config.dataloader_train.dataset)

    # Build dataset_stats from EgoVerseDataset's min-max norm stats so that
    # cosmos_utils.unnormalize_actions / rescale_proprio can work correctly.
    # Keys expected by cosmos_utils: "actions_min", "actions_max",
    #                                "proprio_min", "proprio_max"
    dataset_stats = {}
    norm = getattr(dataset, "_norm_stats", {})
    action_key = getattr(dataset, "action_key", "actions_cartesian")
    proprio_key = getattr(dataset, "proprio_key", "observations.state.ee_pose")
    if action_key in norm:
        dataset_stats["actions_min"] = norm[action_key]["min"].numpy()
        dataset_stats["actions_max"] = norm[action_key]["max"].numpy()
        log_message(
            f"dataset_stats: actions_min shape={dataset_stats['actions_min'].shape}, "
            f"actions_max shape={dataset_stats['actions_max'].shape}",
            None,
        )
    else:
        logger.warning(
            "No min-max stats found for action key '%s' in dataset._norm_stats; "
            "action de-normalization will be skipped.", action_key
        )
    if proprio_key in norm:
        dataset_stats["proprio_min"] = norm[proprio_key]["min"].numpy()
        dataset_stats["proprio_max"] = norm[proprio_key]["max"].numpy()
    else:
        logger.warning(
            "No min-max stats found for proprio key '%s' in dataset._norm_stats; "
            "proprio normalization in get_action will be skipped.", proprio_key
        )

    # Verify chunk size matches
    ds_chunk = getattr(dataset, "chunk_size", cfg.chunk_size)
    assert cfg.chunk_size == ds_chunk, (
        f"chunk_size mismatch: cfg={cfg.chunk_size}, dataset={ds_chunk}"
    )

    # Populate T5 embedding cache from dataset (for cosmos_utils.get_action compatibility)
    if getattr(dataset, "t5_text_embeddings", None) is not None:
        from cosmos_policy.experiments.robot.cosmos_utils import t5_text_embeddings_cache
        for k, v in dataset.t5_text_embeddings.items():
            t5_text_embeddings_cache[k] = (
                v.to(device) if isinstance(v, torch.Tensor) else v
            )
        log_message(f"T5 cache: {len(t5_text_embeddings_cache)} entries", None)

    # ── CameraTransforms (same as egomimic training config) ───────────────
    from egomimic.utils.egomimicUtils import CameraTransforms
    camera_transforms = CameraTransforms(
        intrinsics_key="base",
        extrinsics_key="x5Dec13_2",
    )

    # ── Group items by episode ────────────────────────────────────────────
    all_episodes = group_episodes_from_dataset(
        dataset, max_episodes=cfg.max_episodes, max_frames=cfg.n
    )

    if cfg.episode_indices is not None and cfg.episode_indices.strip():
        selected_keys = [int(x) for x in cfg.episode_indices.split(",")]
        episodes = {k: all_episodes[k] for k in selected_keys if k in all_episodes}
    else:
        episodes = all_episodes

    log_message(f"Visualizing {len(episodes)} episodes", None)

    # ── Setup logging ─────────────────────────────────────────────────────
    log_file, _, _ = setup_logging(
        cfg=cfg,
        task_identifier="visualize",
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=False,
        wandb_entity="",
        wandb_project="",
    )

    # ── Visualize ─────────────────────────────────────────────────────────
    os.makedirs(cfg.save_video_path, exist_ok=True)
    
    # Initialize global loss accumulators
    all_action_mse = []
    all_action_l1 = []
    all_image_mse = []
    all_image_l1 = []
    all_wrist_image_mse = []
    all_wrist_image_l1 = []
    total_frames = 0
    
    for ep_idx, frame_pairs in tqdm.tqdm(episodes.items(), desc="Episodes"):
        video_path, episode_losses = visualize_episode(
            cfg=cfg,
            episode_idx=ep_idx,
            frame_pairs=frame_pairs,
            model=model,
            dataset=dataset,
            dataset_stats=dataset_stats,
            device=device,
            camera_transforms=camera_transforms,
            log_file=log_file,
        )
        
        # Accumulate losses
        if episode_losses["action_mse"] > 0:
            all_action_mse.append(episode_losses["action_mse"])
        if episode_losses["action_l1"] > 0:
            all_action_l1.append(episode_losses["action_l1"])
        if episode_losses["image_mse"] > 0:
            all_image_mse.append(episode_losses["image_mse"])
        if episode_losses["image_l1"] > 0:
            all_image_l1.append(episode_losses["image_l1"])
        if episode_losses["wrist_image_mse"] > 0:
            all_wrist_image_mse.append(episode_losses["wrist_image_mse"])
        if episode_losses["wrist_image_l1"] > 0:
            all_wrist_image_l1.append(episode_losses["wrist_image_l1"])
        total_frames += episode_losses["num_frames"]
        
        log_message(
            f"Episode {ep_idx} losses - "
            f"Action MSE: {episode_losses['action_mse']:.6f}, L1: {episode_losses['action_l1']:.6f} | "
            f"Image MSE: {episode_losses['image_mse']:.6f}, L1: {episode_losses['image_l1']:.6f} | "
            f"Wrist MSE: {episode_losses['wrist_image_mse']:.6f}, L1: {episode_losses['wrist_image_l1']:.6f}",
            log_file
        )

    # ── Print final loss summary ──────────────────────────────────────────
    log_message("=" * 80, log_file)
    log_message("FINAL EVALUATION LOSS SUMMARY", log_file)
    log_message("=" * 80, log_file)
    log_message(f"Total frames evaluated: {total_frames}", log_file)
    log_message(f"Total episodes: {len(episodes)}", log_file)
    log_message("", log_file)
    
    if all_action_mse:
        log_message(
            f"Action Losses (averaged over {len(all_action_mse)} episodes):",
            log_file
        )
        log_message(f"  MSE: {np.mean(all_action_mse):.6f} (std: {np.std(all_action_mse):.6f})", log_file)
        log_message(f"  L1:  {np.mean(all_action_l1):.6f} (std: {np.std(all_action_l1):.6f})", log_file)
        log_message("", log_file)
    
    if all_image_mse:
        log_message(
            f"Primary Image Losses (averaged over {len(all_image_mse)} episodes):",
            log_file
        )
        log_message(f"  MSE: {np.mean(all_image_mse):.6f} (std: {np.std(all_image_mse):.6f})", log_file)
        log_message(f"  L1:  {np.mean(all_image_l1):.6f} (std: {np.std(all_image_l1):.6f})", log_file)
        log_message("", log_file)
    
    if all_wrist_image_mse:
        log_message(
            f"Wrist Image Losses (averaged over {len(all_wrist_image_mse)} episodes):",
            log_file
        )
        log_message(f"  MSE: {np.mean(all_wrist_image_mse):.6f} (std: {np.std(all_wrist_image_mse):.6f})", log_file)
        log_message(f"  L1:  {np.mean(all_wrist_image_l1):.6f} (std: {np.std(all_wrist_image_l1):.6f})", log_file)
        log_message("", log_file)
    
    log_message("=" * 80, log_file)
    log_message("Done.", log_file)
    if log_file:
        log_file.close()


if __name__ == "__main__":
    eval_egoverse_visualize()
