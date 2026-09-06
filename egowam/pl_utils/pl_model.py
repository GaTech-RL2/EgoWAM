import logging
import os
import random
from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torchvision.io as tvio
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger

import egowam.utils.tensor_utils as TensorUtils
from egowam.rldb.embodiment.embodiment import get_embodiment

_log = logging.getLogger(__name__)


class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    debug_loss_spike = False
    debug_loss_spike_factor = 1000.0
    debug_loss_spike_prob = 0.03
    grad_norm_mad_scale = 3.0
    grad_norm_mad_min_count = 100
    grad_norm_mad_window = 200

    def __init__(self, robomimic_model, optimizer, scheduler):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = robomimic_model
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        try:
            self.params = self.model.nets["policy"].params
        except Exception:
            pass
        self.grad_norm_history = deque(maxlen=self.grad_norm_mad_window)

        self.val_image_buffer, self.val_counter = {}, {}
        self.epoch_memory_stats = []  # Store memory stats per epoch
        # TODO __init__ should take the config, and init the model here.  Then save_hyperparameters will just save the config rather than the model

    def root_dir(self):
        return self.trainer.default_root_dir

    def video_dir(self):
        return os.path.join(self.root_dir(), "videos")

    # batch is now a dict, handle on model side
    def training_step(self, batch, batch_idx):
        # Lightning already manages train/eval mode through fit/validate. The
        # explicit `self.train()` here was redundant per-step overhead.
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model.forward_training(batch)
        losses = self.model.compute_losses(predictions, batch)

        # The historical `loss_dicts = [losses]` + torch.mean(torch.stack([losses[k]]))
        # construct was a leftover from a multi-task averaging loop that never
        # collected more than one loss dict; for a single-source step it just
        # rebuilt the dict and added a per-step Tensor allocation. Use the
        # result directly.

        if (
            self.debug_loss_spike
            and random.random() < self.debug_loss_spike_prob
            and self.global_step > 100
        ):
            losses["action_loss"] = losses["action_loss"] * self.debug_loss_spike_factor
            if self.trainer.is_global_zero:
                print(
                    "[LOSS_SPIKE] "
                    f"step={self.global_step} "
                    f"factor={self.debug_loss_spike_factor}",
                    flush=True,
                )

        info = {"losses": TensorUtils.detach(losses)}
        for k, v in self.model.log_info(info).items():
            self.log("Train/" + k, v, sync_dist=True, on_step=False, on_epoch=True)

        return losses["action_loss"]

    def on_after_backward(self):
        # We previously called `clip_grad_norm_` three times per step:
        #   1) here with max_norm=inf to *measure* the raw norm,
        #   2) sometimes (on spike) with max_norm=median to clip,
        #   3) in `on_before_optimizer_step` again with max_norm=inf to log
        #      the post-clip norm.
        # Each call walks every parameter to compute its L2; for ~7M params
        # that's per-step work we duplicated three times.
        #
        # Multi-GPU correctness: under DDP, gradients have already been
        # all-reduced when this hook fires, so the per-rank L2 norm should be
        # mathematically identical across ranks. With bf16 reductions or
        # find_unused_parameters quirks the floats can drift by a ULP, which
        # would let one rank trip the spike threshold while another doesn't,
        # silently desynchronizing the optimizer step. We pin the decision to
        # rank 0 and broadcast: every rank sees the same `raw_norm` and makes
        # the same clip choice. One small all_reduce-equivalent per step
        # replaces three full-parameter L2 traversals.
        raw_norm_t = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(raw_norm_t, src=0)
        raw_norm = float(raw_norm_t)
        clipped_norm = raw_norm
        info = {"policy_grad_norms_raw": raw_norm}

        if len(self.grad_norm_history) >= self.grad_norm_mad_min_count:
            values = np.asarray(self.grad_norm_history, dtype=np.float32)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad > 0.0:
                threshold = median + self.grad_norm_mad_scale * mad
                info["policy_grad_norms_mad_threshold"] = threshold
                info["policy_grad_norms_mad_flag"] = float(raw_norm > threshold)
                if raw_norm > threshold:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=median)
                    clipped_norm = min(raw_norm, median)
                    if self.trainer.is_global_zero:
                        print(
                            "[GRAD_NORM_SPIKE] "
                            f"step={self.global_step} "
                            f"grad_norm={raw_norm:.4f} "
                            f"median={median:.4f} "
                            f"mad={mad:.4f} "
                            f"threshold={threshold:.4f}",
                            flush=True,
                        )

        info["policy_grad_norms_clipped"] = clipped_norm
        self.grad_norm_history.append(raw_norm)
        for k, v in info.items():
            self.log("Train/" + k, v, on_step=False, on_epoch=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        # Intentional no-op. Previously this called `clip_grad_norm_(inf)`
        # purely to log the post-clip norm, but `on_after_backward` already
        # recorded it without paying for a second per-step parameter
        # traversal.
        pass

    def on_validation_start(self):
        self.model.device = self.device

        # Pass the current epoch + last-epoch flag to the model so the
        # visualization cadence in `forward_eval_logging` can decide whether
        # to render panels for this validation. Saves ~80% of val_step wall
        # time on epochs where viz isn't requested.
        setter = getattr(self.model, "set_current_epoch_for_viz", None)
        if callable(setter):
            current = int(self.trainer.current_epoch)
            max_epochs = self.trainer.max_epochs or 0
            setter(current, is_last=current >= max_epochs - 1)

        if self.trainer.is_global_zero:
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )
            # reset per-epoch validation metric accumulator
            self._val_metric_buffer = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Run one validation batch and stash everything for end-of-epoch flush.

        Per-batch hot-path simplifications (the GPU-utilization-relevant ones):

        1. **No per-metric GPU->CPU sync.** The previous version called
           `float(v.detach().cpu())` for every metric on every batch, each of
           which is a sync barrier that flushes the CUDA queue. We now stash
           the metric tensors as-is (`.detach()` only) and pull them to host
           in one `torch.stack(...).cpu()` call from `on_validation_end`.

        2. **No in-step `torch.stack` / `tvio.write_video`.** The old buffer
           did `torch.from_numpy + .extend` per batch and, every >= 1000
           frames, ran `torch.stack` + `write_video` synchronously - tens of
           ms of CPU work that GPU waited through. We keep raw `np.uint8`
           frames in the buffer here and write all videos at validation end.
        """
        print(f"[VAL_STEP] rank={self.global_rank}, batch_idx={batch_idx}", flush=True)

        batch = self.model.process_batch_for_training(batch)
        metrics, images_dict = self.model.forward_eval_logging(batch)

        # Backward compatibility: older models (e.g., plain HPT without WM) may
        # return a single numpy array of frames instead of a dict keyed by embodiment.
        if not isinstance(images_dict, dict):
            images_dict = {0: images_dict}

        # Build a metrics dict of GPU tensors (no .cpu() per metric) for the
        # Lightning logger AND for our own deferred-sync accumulator. We also
        # avoid `torch.tensor(v, device=...)` on already-tensor inputs because
        # that's a needless host->device copy.
        gpu_metrics = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                gpu_metrics[k] = v.detach().to(self.device, non_blocking=True)
            else:
                gpu_metrics[k] = torch.as_tensor(v, device=self.device)

        # Accumulate per-batch metrics on rank 0 without syncing to host.
        # `on_validation_end` does one bulk `.cpu()` call across the whole
        # epoch instead of len(metrics) * num_batches syncs per epoch.
        if self.trainer.is_global_zero:
            if not hasattr(self, "_val_metric_buffer_gpu"):
                self._val_metric_buffer_gpu: Dict[str, list] = {}
            for k, v in gpu_metrics.items():
                self._val_metric_buffer_gpu.setdefault(k, []).append(v)

        # Stash raw numpy frames per (embodiment, stream); the actual
        # torch.stack and tvio.write_video happen in `on_validation_end`.
        for emb_id, images in images_dict.items():
            if isinstance(images, dict):
                for stream_name, frames_np in images.items():
                    if frames_np is None or len(frames_np) == 0:
                        continue
                    buffer_key = (emb_id, stream_name)
                    if (
                        buffer_key not in self.val_image_buffer
                        or self.val_image_buffer[buffer_key] is None
                    ):
                        self.val_image_buffer[buffer_key] = []
                        self.val_counter[buffer_key] = 0
                    self.val_image_buffer[buffer_key].append(frames_np)
            else:
                if (
                    emb_id not in self.val_image_buffer
                    or self.val_image_buffer[emb_id] is None
                ):
                    self.val_image_buffer[emb_id] = []
                    self.val_counter[emb_id] = 0
                self.val_image_buffer[emb_id].append(images)

        self.log_dict(gpu_metrics, sync_dist=True)

    def on_validation_end(self):
        print(f"[ON_VALIDATION_END] rank={self.global_rank}", flush=True)
        # Each entry in `self.val_image_buffer[key]` is now a list of np.uint8
        # arrays accumulated by `validation_step` without any per-batch
        # `torch.from_numpy` / `torch.stack` / `tvio.write_video` work. We do
        # the heavy CPU-side stacking and video encoding here, after the
        # validation loop is done -- no GPU work is waiting on us.
        for key, buffer in self.val_image_buffer.items():
            if len(buffer) == 0:
                self.val_counter[key] = 0
                self.val_image_buffer[key] = []
                continue

            # key can be either emb_id or (emb_id, stream_name)
            if isinstance(key, tuple):
                emb_id, stream_name = key
            else:
                emb_id, stream_name = key, None

            emb_name = str(get_embodiment(emb_id))
            base_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                emb_name,
            )
            if stream_name is not None:
                out_dir = os.path.join(base_dir, stream_name)
            else:
                out_dir = base_dir

            os.makedirs(out_dir, exist_ok=True)

            # `buffer` is either a list of (T, H, W, 3) np.uint8 arrays
            # produced by validation_step, or (legacy, single-stream path) a
            # list of torch tensors. Normalize to a single torch tensor here
            # without paying the per-batch stack cost.
            if isinstance(buffer[0], np.ndarray):
                frames_np = (
                    np.concatenate(buffer, axis=0) if len(buffer) > 1 else buffer[0]
                )
                frames = torch.from_numpy(frames_np)
            else:
                frames = torch.stack(buffer)
            path = os.path.join(
                out_dir,
                f"validation_video_{self.val_counter[key]}.mp4",
            )
            tvio.write_video(path, frames, fps=30, video_codec="h264")

            # Optionally log only the latest video per embodiment/stream to WandB.
            if (
                self.trainer.is_global_zero
                and isinstance(self.logger, WandbLogger)
                and wandb is not None
            ):
                # Use a stable key so each epoch overwrites the previous video.
                if stream_name is not None:
                    wandb_key = f"val_videos/{emb_name}/{stream_name}"
                else:
                    wandb_key = f"val_videos/{emb_name}"
                wandb_video = wandb.Video(path, fps=30, format="mp4")
                self.logger.experiment.log(
                    {
                        wandb_key: wandb_video,
                        "epoch": self.trainer.current_epoch,
                    }
                )

            self.val_counter[key] = 0
            self.val_image_buffer[key] = []
        # Sync the deferred GPU metric buffer in one shot: stack then
        # `.cpu()` once per metric, then average. The previous code did
        # `float(v.detach().cpu())` per metric per batch inside
        # `validation_step` -- N_metrics * N_batches sync barriers per epoch.
        if self.trainer.is_global_zero and hasattr(self, "_val_metric_buffer_gpu"):
            avg_metrics = {}
            for k, tensors in self._val_metric_buffer_gpu.items():
                if not tensors:
                    continue
                stacked = torch.stack([t.float() for t in tensors])
                avg_metrics[k] = float(stacked.mean().cpu())
            _log.info(
                f"[ValMetrics][epoch={self.current_epoch}][avg_over_batches] {avg_metrics}"
            )
            self._val_metric_buffer_gpu = {}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        hparams = checkpoint.get("hyper_parameters")
        if not isinstance(hparams, dict) or "robomimic_model" not in hparams:
            return

        strategy = getattr(getattr(self, "trainer", None), "strategy", None)
        strategy_name = (
            strategy.__class__.__name__.lower() if strategy is not None else ""
        )
        if "fsdp" not in strategy_name:
            return

        # FSDP mixed precision installs local forward hooks on wrapped BatchNorm
        # modules. The live robomimic model in hparams then becomes unpicklable
        # when Lightning writes FSDP checkpoint metadata, while the actual weights
        # are already stored through the strategy state_dict path.
        hparams.pop("robomimic_model", None)
        hparams["_robomimic_model_omitted"] = "omitted from FSDP checkpoint metadata"

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        trainable_params = [
            p for p in self.trainer.model.parameters() if p.requires_grad
        ]
        optimizer = self.hparams.optimizer(params=trainable_params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                # "lr_scheduler": {
                #     "scheduler": scheduler,
                #     "monitor": "val/loss",
                #     "interval": "epoch",
                #     "frequency": 1,
                # },
            }
        return {"optimizer": optimizer}

    def on_fit_start(self):
        self.model.device = self.device
        dist_ready = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if not dist_ready:
            return
        print(
            f"Rank {self.global_rank} on fit start, waiting for all ranks to synchronize",
            flush=True,
        )
        torch.distributed.barrier()
        print(
            f"Rank {self.global_rank} on fit start, all ranks synchronized", flush=True
        )

    def on_train_epoch_start(self):
        log_all = {}
        for i, param_group in enumerate(self.optimizers().param_groups):
            log_all[f"Optimizer/param_group_{i}_lr"] = param_group["lr"]

        return super().on_train_epoch_start()
