"""
hpt_video_wm.py

HPTVideoWM — HPT world-model variant with a Wan2.1 VACE-style future image latent
prediction head (``WanVACEFutureImageHead``).

All training behaviour is inherited from ``HPT`` (hpt_wm.py).

=== Evaluation: sparse future video prediction ===

The future-image head jointly predicts the configured sparse future frames. For
the five-frame Pixel-PT baseline this is:

    EVA:  obs_t → VAE(img_{t+[9,18,27,36,45]})
    Aria: obs_t → VAE(img_{t+[6,12,18,24,30]})

Single-frame inputs remain supported for older configs and checkpoints.

The frames live on the sequence axis of the latent stack ``[B, N, C*H*W]``, which
the official Wan backend maps onto the video transformer's frame axis — so the
pretrained temporal prior is used as intended rather than as N independent
images.

Metrics (all under ``Valid/``)
------------------------------
{pred_key}_tf_rollout_pixel_mse_avg     ← average over sparse future sequence
{pred_key}_tf_rollout_pixel_l1_avg
{pred_key}_tf_rollout_latent_mse_avg
{pred_key}_tf_rollout_pixel_mse_{cam_short}
{pred_key}_tf_rollout_pixel_l1_{cam_short}
{pred_key}_tf_rollout_latent_mse_{cam_short}

Per-offset metrics add a ``_t_plus_{offset}`` suffix.

Visualisation panels
--------------------
images_dict[embodiment_id]["future_tf_rollout"] — GT | VAE rec | joint prediction
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from overrides import override
from termcolor import cprint

from egowam.algo.hpt_wm import HPT
from egowam.models.wan_dit import WanVACEFutureImageHead
from egowam.rldb.embodiment.embodiment import get_embodiment

_STEP_SUFFIX_RE = re.compile(r"_s(\d+)$")


class HPTVideoWM(HPT):
    """HPT world-model with a Wan/VACE-style future image latent prediction head.

    Drop-in replacement for ``HPT`` (hpt_wm.py).  All existing functionality is
    preserved; only the future image prediction head is swapped out and eval
    logging is extended with sparse future-video metrics and visualization.

    New constructor parameters
    --------------------------
    wan_vae : WanImageVAEWrapper | CosmosVAEWrapper | None
        VAE used for encoding/decoding future image latents.  When None, falls
        back to the parent's ``cosmos_vae``.
    future_image_offsets : dict[str, list[int]] | None
        Per-embodiment frame offsets, one per entry of ``future_image_keys`` and
        in the same order (EVA ``[9,18,27,36,45]``, Aria ``[6,12,18,24,30]``).
        Used for metric labelling only — the actual temporal spacing is set by
        the data config's ``future_image_offsets``, which is what the dataloader
        reads.  A mismatched length falls back to the ``_s{i}`` step index parsed
        from the key name, so labels degrade rather than lie.
    rollout_steps : int
        Kept for config compatibility. The head predicts the whole sparse future
        sequence in one shot, so eval always runs a single prediction pass.
    """

    def __init__(
        self,
        *args,
        wan_vae=None,
        tf_noise_level: float = 0.5,  # kept for compat; no longer used in eval
        rollout_steps: int = 2,
        future_image_offsets: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.future_image_offsets = {
            str(domain).lower(): [int(o) for o in offsets]
            for domain, offsets in (future_image_offsets or {}).items()
        }
        for domain, offsets in self.future_image_offsets.items():
            if len(offsets) != len(self.future_image_keys):
                cprint(
                    f"[HPTVideoWM] future_image_offsets['{domain}'] has "
                    f"{len(offsets)} entries but future_image_keys has "
                    f"{len(self.future_image_keys)}; falling back to step-index "
                    "labels for this domain.",
                    "yellow",
                )

        # Override VAE with wan_vae if provided; otherwise reuse cosmos_vae.
        if wan_vae is not None:
            self.wan_vae = wan_vae.to(self.device)
            self.wan_vae.eval()
        else:
            self.wan_vae = getattr(self, "cosmos_vae", None)

        self.rollout_steps = 1

        # Inject the VAE reference into any WanVACEFutureImageHead heads so they
        # can decode latents for visualisation without a separate VAE attribute.
        policy = self.nets["policy"]
        for head_key, head in policy.heads.items():
            if isinstance(head, WanVACEFutureImageHead) and not hasattr(head, "_vae"):
                head._vae = self.wan_vae

        # Verify that the trunk uses joint_token so future_feats are available.
        tp = policy.token_postprocessing
        if tp != "joint_token":
            raise ValueError(
                f"HPTVideoWM requires trunk.token_postprocessing='joint_token' to "
                f"produce future tokens.  Got '{tp}'."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _active_vae(self):
        """Return the active VAE (wan_vae if set, else cosmos_vae)."""
        return getattr(self, "wan_vae", None) or getattr(self, "cosmos_vae", None)

    # ------------------------------------------------------------------
    # Sparse future-sequence labelling
    # ------------------------------------------------------------------

    @staticmethod
    def _camera_short_name(future_cam_key: str) -> str:
        """``future_front_img_1_s3`` → ``front``; ``future_left_wrist_img`` → ``left_wrist``."""
        base = _STEP_SUFFIX_RE.sub("", future_cam_key)
        return base.replace("future_", "").replace("_img_1", "").replace("_img", "")

    def _slot_offset_label(
        self, future_cam_key: str, embodiment_name: str
    ) -> Optional[str]:
        """Return the ``t_plus_{offset}`` label for one slot, or None.

        None means "this key is not part of a sparse sequence" — the single-frame
        configs then keep their historical metric names untouched.
        """
        step = _STEP_SUFFIX_RE.search(future_cam_key)
        if step is None:
            return None

        offsets = self.future_image_offsets.get(embodiment_name.lower())
        if offsets is not None and len(offsets) == len(self.future_image_keys):
            try:
                return f"t_plus_{offsets[self.future_image_keys.index(future_cam_key)]}"
            except ValueError:
                pass
        # No usable offset table: fall back to the step index in the key name.
        return f"step_{int(step.group(1))}"

    def _ordered_future_keys(self, layout) -> List[str]:
        """Layout keys in config order — i.e. ascending future offset."""
        return sorted(
            [entry[0] for entry in layout if entry[0] in self.future_image_keys],
            key=self.future_image_keys.index,
        )

    # ------------------------------------------------------------------
    # Future target encoding (override to support wan_vae)
    # ------------------------------------------------------------------

    @override
    def _attach_future_targets(self, batch, embodiment_id):
        """Encode future images → latents and attach to batch.

        Identical logic to the parent's implementation except that it uses
        ``_active_vae`` rather than ``self.cosmos_vae`` directly.
        """
        vae = self._active_vae

        if (
            self.future_proprio_key in batch
            and batch[self.future_proprio_key].ndim == 2
        ):
            batch[self.future_proprio_key] = batch[self.future_proprio_key].unsqueeze(1)

        if self.future_flow_key in batch and batch[self.future_flow_key].ndim == 4:
            flow = batch[self.future_flow_key]
            batch[self.future_flow_key] = flow.reshape(flow.shape[0], 1, -1)

        if (
            not self._has_future_image_head
            or vae is None
            or len(self.future_image_keys) == 0
        ):
            return

        latents = []
        layout = []
        for cam_key in self.future_image_keys:
            if cam_key not in batch:
                continue
            image = batch[cam_key].float()
            if image.max() > 1.5:
                image = image / 255.0
            image = F.interpolate(
                image,
                size=(self.future_image_resize, self.future_image_resize),
                mode="bilinear",
                align_corners=False,
            )
            videos = image.unsqueeze(2)  # [B, 3, 1, H, W]
            with torch.no_grad():
                latent = vae.encode(videos)  # [B, C, T_lat, H_lat, W_lat]

            cam_idx = len(latents)
            flat = latent.reshape(latent.shape[0], 1, -1).detach()
            latents.append(flat)
            layout.append(
                (cam_key, cam_idx, latent.shape[1], latent.shape[3], latent.shape[4])
            )

        if len(latents) == 0:
            return

        batch[self.future_image_latent_key] = torch.cat(latents, dim=1)
        self._future_latent_layout[embodiment_id] = layout

    # ------------------------------------------------------------------
    # Latent decoding (override to support wan_vae)
    # ------------------------------------------------------------------

    @override
    def _decode_future_latents(self, embodiment_id, latents):
        """Decode per-camera latent tokens to pixel images.

        Parameters
        ----------
        latents : Tensor [B, N_cam, cam_lat_dim]

        Returns
        -------
        dict  cam_key → Tensor [B, 3, H, W]
        """
        vae = self._active_vae
        layout = self._future_latent_layout.get(embodiment_id, [])
        decoded = {}
        if vae is None or len(layout) == 0:
            return decoded

        for cam_key, cam_idx, channels, h_lat, w_lat in layout:
            if cam_idx >= latents.shape[1]:
                continue
            cam_latent = latents[:, cam_idx, :]
            cam_latent = cam_latent.reshape(latents.shape[0], channels, 1, h_lat, w_lat)
            with torch.no_grad():
                rec = vae.decode(cam_latent)  # [B, 3, 1, H, W]
            decoded[cam_key] = rec[:, :, 0]  # [B, 3, H, W]
        return decoded

    # ------------------------------------------------------------------
    # Helper: find WanVACEFutureImageHead for a given embodiment
    # ------------------------------------------------------------------

    def _get_wan_head(self, embodiment_name: str) -> Optional[WanVACEFutureImageHead]:
        policy = self.nets["policy"]
        head_key = policy._resolve_aux_head_key(
            embodiment_name, self.future_image_latent_key
        )
        if head_key not in policy.heads:
            return None
        head = policy.heads[head_key]
        if isinstance(head, WanVACEFutureImageHead):
            return head
        return None

    # ------------------------------------------------------------------
    # Core: joint sparse future image prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout_future_images(
        self,
        batch: dict,
        embodiment_id: int,
        embodiment_name: str,
        wan_head: WanVACEFutureImageHead,
        n_rollout_steps: int = 1,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the joint sparse future image prediction used during eval.

        The model receives the current visual observation, runs the HPT trunk to
        obtain ``future_feats``, and calls ``wan_head.sample`` once to predict all
        configured future frames together — the head is not autoregressive, the
        whole sparse sequence is one denoising trajectory.

        Parameters
        ----------
        batch : dict
            Normalised eval batch for one embodiment.  Must contain:
            * camera observation keys (e.g. ``front_img_1``)
            * ``self.future_image_latent_key`` for GT comparison
        embodiment_id : int
        embodiment_name : str
        wan_head : WanVACEFutureImageHead
        n_rollout_steps : int
            Kept for call-site compatibility; always coerced to one pass. The
            temporal extent lives in the ``N`` axis, not in repeated calls.

        Returns
        -------
        pred_latent_seq : Tensor [B, T, N, C*H*W] or None
        gt_latent_seq   : Tensor [B, T, N, C*H*W] or None
            T=1; ``N`` covers every (camera, future offset) slot.
        """
        vae = self._active_vae
        if vae is None:
            return None, None

        n_rollout_steps = 1

        policy = self.nets["policy"]
        cam_keys = self.camera_keys[embodiment_id]
        proprio_keys = self.proprio_keys[embodiment_id]
        lang_keys = self.lang_keys[embodiment_id]
        ac_key = self.ac_keys[embodiment_id]
        aux_ac_keys = self.auxiliary_ac_keys.get(embodiment_name, [])
        N_cam = len(self.future_image_keys)

        gt_latent_step0 = batch.get(self.future_image_latent_key)
        if gt_latent_step0 is None:
            return None, None

        pred_latent_list: List[torch.Tensor] = []
        gt_latent_list: List[torch.Tensor] = []

        for step in range(n_rollout_steps):
            # Build HPT data dict from current observation
            data = self._robomimic_to_hpt_data(
                batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
            )

            # Run HPT trunk → future_feats [B, future_horizon, embed_dim]
            _, future_feats, _ = policy.forward_features(embodiment_name, data)
            if future_feats is None:
                cprint(
                    "[HPTVideoWM.rollout] future_feats is None — trunk may not use joint_token",
                    "yellow",
                )
                break

            # Predict next-frame latent (full ODE integration, no noisy-GT short-cut)
            pred_latent = wan_head.sample(
                future_feats, N_cam=N_cam
            )  # [B, N_cam, C*H*W]
            pred_latent_list.append(pred_latent)
            gt_latent_list.append(gt_latent_step0)

        if not pred_latent_list:
            return None, None

        pred_seq = torch.stack(pred_latent_list, dim=1)  # [B, T, N_cam, C*H*W]
        gt_seq = torch.stack(gt_latent_list, dim=1)  # [B, T, N_cam, C*H*W]
        return pred_seq, gt_seq

    # ------------------------------------------------------------------
    # Metric helper
    # ------------------------------------------------------------------

    def _compute_rollout_metrics(
        self,
        metrics: dict,
        pred_key: str,
        embodiment_id: int,
        pred_seq: torch.Tensor,  # [B, T, N_cam, C*H*W]
        gt_seq: torch.Tensor,  # [B, T, N_cam, C*H*W]
        prefix: str = "rollout",
    ) -> None:
        """Compute pixel- and latent-space metrics for a predicted future sequence.

        Every slot of the ``N`` axis is one (camera, future offset) pair, so with
        the five-frame Pixel-PT configs a single front camera yields five slots.
        Three levels of aggregation are logged:

        * per slot     ``..._{cam_short}_t_plus_{offset}``
        * per camera   ``..._{cam_short}``            (mean over that camera's offsets)
        * per offset   ``..._avg_t_plus_{offset}``    (mean over cameras)
        * overall      ``..._avg``                    (mean over cameras and offsets)

        Single-frame configs have no ``_s{i}`` suffix, hence no offset label, and
        keep exactly their historical metric names.

        Only steps where GT is non-zero are included.  Metrics are averaged over
        time and batch.

        Parameters
        ----------
        prefix : str
            Metric key suffix, e.g. ``tf_rollout`` for
            ``Valid/{pred_key}_tf_rollout_*``.
        """
        layout = self._future_latent_layout.get(embodiment_id, [])
        if not layout:
            return
        embodiment_name = get_embodiment(embodiment_id).lower()
        T = pred_seq.shape[1]

        # Accumulate across steps that have GT (non-zero gt_seq)
        total_pix_mse = {e[0]: 0.0 for e in layout}
        total_pix_l1 = {e[0]: 0.0 for e in layout}
        total_lat_mse = {e[0]: 0.0 for e in layout}
        n_steps_with_gt = 0

        for t in range(T):
            gt_t = gt_seq[:, t]  # [B, N, C*H*W]
            # Skip steps where GT is all-zeros (placeholder for unavailable GT)
            if gt_t.abs().max().item() < 1e-9:
                continue

            pred_t = pred_seq[:, t]

            pred_frames = self._decode_future_latents(embodiment_id, pred_t)
            gt_frames = self._decode_future_latents(embodiment_id, gt_t)
            n_steps_with_gt += 1

            for future_cam_key, cam_idx, _, _, _ in layout:
                if future_cam_key not in pred_frames or future_cam_key not in gt_frames:
                    continue
                p_img = pred_frames[future_cam_key].float().clamp(0.0, 1.0).cpu()
                g_img = gt_frames[future_cam_key].float().clamp(0.0, 1.0).cpu()
                total_pix_mse[future_cam_key] += F.mse_loss(p_img, g_img).item()
                total_pix_l1[future_cam_key] += F.l1_loss(p_img, g_img).item()
                if cam_idx < pred_t.shape[1] and cam_idx < gt_t.shape[1]:
                    total_lat_mse[future_cam_key] += F.mse_loss(
                        pred_t[:, cam_idx, :].float().cpu(),
                        gt_t[:, cam_idx, :].float().cpu(),
                    ).item()

        if n_steps_with_gt == 0:
            return

        # slot value tables, grouped both ways
        by_camera: dict[str, list[tuple[float, float, float]]] = {}
        by_offset: dict[str, list[tuple[float, float, float]]] = {}

        for future_cam_key in self._ordered_future_keys(layout):
            cam_short = self._camera_short_name(future_cam_key)
            offset_label = self._slot_offset_label(future_cam_key, embodiment_name)
            values = (
                total_pix_mse[future_cam_key] / n_steps_with_gt,
                total_pix_l1[future_cam_key] / n_steps_with_gt,
                total_lat_mse[future_cam_key] / n_steps_with_gt,
            )

            if offset_label is not None:
                slot_name = f"{cam_short}_{offset_label}"
                metrics[f"Valid/{pred_key}_{prefix}_pixel_mse_{slot_name}"] = values[0]
                metrics[f"Valid/{pred_key}_{prefix}_pixel_l1_{slot_name}"] = values[1]
                metrics[f"Valid/{pred_key}_{prefix}_latent_mse_{slot_name}"] = values[2]
                by_offset.setdefault(offset_label, []).append(values)

            by_camera.setdefault(cam_short, []).append(values)

        def _mean(rows, i):
            return sum(row[i] for row in rows) / len(rows)

        for cam_short, rows in by_camera.items():
            metrics[f"Valid/{pred_key}_{prefix}_pixel_mse_{cam_short}"] = _mean(rows, 0)
            metrics[f"Valid/{pred_key}_{prefix}_pixel_l1_{cam_short}"] = _mean(rows, 1)
            metrics[f"Valid/{pred_key}_{prefix}_latent_mse_{cam_short}"] = _mean(
                rows, 2
            )

        for offset_label, rows in by_offset.items():
            metrics[f"Valid/{pred_key}_{prefix}_pixel_mse_avg_{offset_label}"] = _mean(
                rows, 0
            )
            metrics[f"Valid/{pred_key}_{prefix}_pixel_l1_avg_{offset_label}"] = _mean(
                rows, 1
            )
            metrics[f"Valid/{pred_key}_{prefix}_latent_mse_avg_{offset_label}"] = _mean(
                rows, 2
            )

        # Overall average: mean of the per-camera means, so a multi-camera config
        # is not dominated by whichever camera happens to have more offsets.
        cam_means = [
            (_mean(rows, 0), _mean(rows, 1), _mean(rows, 2))
            for rows in by_camera.values()
        ]
        if cam_means:
            metrics[f"Valid/{pred_key}_{prefix}_pixel_mse_avg"] = _mean(cam_means, 0)
            metrics[f"Valid/{pred_key}_{prefix}_pixel_l1_avg"] = _mean(cam_means, 1)
            metrics[f"Valid/{pred_key}_{prefix}_latent_mse_avg"] = _mean(cam_means, 2)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _decode_latent_seq_to_images(
        self,
        pred_seq: torch.Tensor,  # [B, T, N_cam, C*H*W]
        embodiment_id: int,
        cell_h: int,
        cell_w: int,
        blank_zero_latents: bool = False,
    ) -> Optional[np.ndarray]:
        """Decode a latent sequence to a grid of images.

        Returns
        -------
        np.ndarray [B, H, T*N_cam*W, 3]   (time × cam concatenated horizontally)
        None if layout is empty or no cameras are available.
        """
        vae = self._active_vae
        layout = self._future_latent_layout.get(embodiment_id, [])
        if not layout or vae is None:
            return None

        B, T, N_cam, _ = pred_seq.shape

        ordered_keys = self._ordered_future_keys(layout)
        key_to_entry = {e[0]: e for e in layout}

        def _resize(img):
            return F.interpolate(
                img.float().clamp(0.0, 1.0),
                size=(cell_h, cell_w),
                mode="bilinear",
                align_corners=False,
            )

        # Build a strip of frames: [B, cell_h, T*N_cam*cell_w, 3]
        all_frames = []
        for t in range(T):
            for cam_key in ordered_keys:
                entry = key_to_entry.get(cam_key)
                if entry is None:
                    continue
                _, cam_idx, channels, h_lat, w_lat = entry
                if cam_idx >= pred_seq.shape[2]:
                    continue
                lat_flat = pred_seq[:, t, cam_idx, :]
                if blank_zero_latents and lat_flat.abs().max().item() < 1e-9:
                    all_frames.append(np.zeros((B, cell_h, cell_w, 3), dtype=np.uint8))
                    continue
                lat = lat_flat.reshape(B, channels, 1, h_lat, w_lat)
                with torch.no_grad():
                    img = vae.decode(lat)[:, :, 0]  # [B, 3, H, W]
                img_np = (
                    _resize(img).cpu().numpy().transpose(0, 2, 3, 1) * 255
                ).astype(np.uint8)  # [B, cell_h, cell_w, 3]
                all_frames.append(img_np)

        if not all_frames:
            return None
        return np.concatenate(all_frames, axis=2)  # [B, H, T*N_cam*W, 3]

    def _future_image_seq_to_images(
        self,
        unnorm_batch: dict,
        embodiment_id: int,
        ref_seq: torch.Tensor,  # [B, T, N_cam, C*H*W]
        cell_h: int,
        cell_w: int,
    ) -> Optional[np.ndarray]:
        """Build a raw-GT image strip aligned with a latent rollout sequence."""
        layout = self._future_latent_layout.get(embodiment_id, [])
        if not layout:
            return None

        B, T, _, _ = ref_seq.shape

        ordered_keys = self._ordered_future_keys(layout)
        if not ordered_keys:
            return None

        def _resize(img):
            return F.interpolate(
                img.float().clamp(0.0, 1.0),
                size=(cell_h, cell_w),
                mode="bilinear",
                align_corners=False,
            )

        all_frames = []
        blank = np.zeros((B, cell_h, cell_w, 3), dtype=np.uint8)
        for t in range(T):
            for cam_key in ordered_keys:
                if t != 0 or cam_key not in unnorm_batch:
                    all_frames.append(blank.copy())
                    continue

                img = unnorm_batch[cam_key].float()
                if img.ndim == 5:
                    if img.shape[1] == 1:
                        img = img[:, 0]
                    elif img.shape[2] == 1:
                        img = img[:, :, 0]
                if (
                    img.ndim == 4
                    and img.shape[1] not in (1, 3)
                    and img.shape[-1] in (1, 3)
                ):
                    img = img.permute(0, 3, 1, 2)
                if img.ndim != 4:
                    all_frames.append(blank.copy())
                    continue
                if img.shape[1] == 1:
                    img = img.repeat(1, 3, 1, 1)
                if img.max() > 1.5:
                    img = img / 255.0

                img_np = (
                    _resize(img).cpu().numpy().transpose(0, 2, 3, 1) * 255
                ).astype(np.uint8)
                all_frames.append(img_np)

        if not all_frames:
            return None
        return np.concatenate(all_frames, axis=2)

    def _build_rollout_panel(
        self,
        unnorm_batch: dict,
        embodiment_id: int,
        gt_seq: torch.Tensor,  # [B, T, N_cam, C*H*W]
        pred_seq_tf: torch.Tensor,  # [B, T, N_cam, C*H*W]
    ) -> Optional[np.ndarray]:
        """Build the sparse future-image visualisation panel.

        Layout (concatenated vertically):
            Row 0: GT frames           (raw future images)
            Row 1: VAE reconstruction  (decoded GT latents)
            Row 2: Prediction          (decoded predicted latents)

        Columns run over the ``N`` slots in config order, i.e. ascending future
        offset (t+9, t+18, … for EVA), so each column triple is one horizon.

        Returns
        -------
        np.ndarray [B, rows*H, T*N*W, 3]  or None
        """
        viz_img_key = self.data_schematic.viz_img_key()[embodiment_id]
        if viz_img_key not in unnorm_batch:
            return None
        base_hw = unnorm_batch[viz_img_key].shape[-2:]
        cell_h = int(base_hw[0]) - (int(base_hw[0]) % 2)
        cell_w = int(base_hw[1]) - (int(base_hw[1]) % 2)

        rows = []
        gt_strip = self._future_image_seq_to_images(
            unnorm_batch,
            embodiment_id,
            gt_seq,
            cell_h,
            cell_w,
        )
        if gt_strip is not None:
            rows.append(gt_strip)

        vae_rec_strip = self._decode_latent_seq_to_images(
            gt_seq,
            embodiment_id,
            cell_h,
            cell_w,
            blank_zero_latents=True,
        )
        if vae_rec_strip is not None:
            rows.append(vae_rec_strip)

        pred_strip = self._decode_latent_seq_to_images(
            pred_seq_tf,
            embodiment_id,
            cell_h,
            cell_w,
            blank_zero_latents=False,
        )
        if pred_strip is not None:
            rows.append(pred_strip)

        if not rows:
            return None

        # Trim to same width in case different rows decode to different widths.
        min_w = min(r.shape[2] for r in rows)
        rows = [r[:, :, :min_w, :] for r in rows]
        return np.concatenate(rows, axis=1)  # [B, rows*H, T*N_cam*W, 3]

    # ------------------------------------------------------------------
    # Eval logging (override to add one-step future-image metrics + visualisation)
    # ------------------------------------------------------------------

    @override
    def forward_eval_logging(self, batch):
        """Extends parent eval logging with sparse future-video eval.

        For each embodiment using ``WanVACEFutureImageHead``:

        1. Joint prediction: current observation → every configured future frame.
           Metrics prefix: ``tf_rollout_``
           Panel: ``images_dict[emb_id]["future_tf_rollout"]``

        The ``tf_rollout`` prefix is kept for metric/key compatibility with
        earlier eval runs.
        """
        # Run parent evaluation (action head + standard future-image prediction)
        metrics, images_dict = super().forward_eval_logging(batch)

        for embodiment_id, _batch_raw in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            wan_head = self._get_wan_head(embodiment_name)
            if wan_head is None:
                continue

            vae = self._active_vae
            if vae is None:
                continue

            # Unnormalise to get raw GT latents and future images
            unnorm_batch = self.data_schematic.unnormalize_data(
                self._prepare_eval_denorm_tensors(dict(_batch_raw)), embodiment_id
            )
            if self.future_image_latent_key not in unnorm_batch:
                continue

            gt_latents_step0 = unnorm_batch[self.future_image_latent_key].to(
                self.device
            )

            # Build a working batch with GT latents for scoring and raw future
            # images for the GT visualization row.
            rollout_batch = dict(_batch_raw)
            rollout_batch[self.future_image_latent_key] = gt_latents_step0
            for cam_key in self.future_image_keys:
                if cam_key in unnorm_batch:
                    rollout_batch[cam_key] = unnorm_batch[cam_key]

            n_steps = 1
            pred_key = f"{embodiment_name}_{self.future_image_latent_key}"

            # ---- One-step future image prediction ----
            pred_seq_tf = None
            gt_seq_tf = None
            if n_steps > 0:
                pred_seq_tf, gt_seq_tf = self.rollout_future_images(
                    batch=rollout_batch,
                    embodiment_id=embodiment_id,
                    embodiment_name=embodiment_name,
                    wan_head=wan_head,
                    n_rollout_steps=n_steps,
                )
                if pred_seq_tf is not None:
                    self._compute_rollout_metrics(
                        metrics,
                        pred_key,
                        embodiment_id,
                        pred_seq_tf,
                        gt_seq_tf,
                        prefix="tf_rollout",
                    )

            # ---- Visualisation ----
            if pred_seq_tf is not None:
                if embodiment_id not in images_dict:
                    images_dict[embodiment_id] = {}

                tf_panel = self._build_rollout_panel(
                    unnorm_batch, embodiment_id, gt_seq_tf, pred_seq_tf
                )
                if tf_panel is not None:
                    images_dict[embodiment_id]["future_tf_rollout"] = tf_panel

        return metrics, images_dict
