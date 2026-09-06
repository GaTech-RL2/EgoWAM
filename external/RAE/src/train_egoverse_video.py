# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Stage-2 transport training on EgoVerse Zarr for future-frame prediction.

This script trains a latent diffusion/transport model to predict future-frame
latents conditioned on current-frame latents.
"""

from __future__ import annotations
import json
import argparse
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from dataset.egoverse_zarr_dset import load_egoverse_zarr_images_train_val
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import Sampler, create_transport
from utils import wandb_utils
from utils.dist_utils import cleanup_distributed, setup_distributed
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import configure_experiment_dirs, find_resume_checkpoint, save_worktree
from utils.train_utils import get_autocast_scaler, parse_configs, update_ema

# the first flag below was False when this script was tested but True can speed up A100 training.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _dist_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage-2 transport model on EgoVerse Zarr current->future frames."
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1 and stage_2 sections.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to EgoVerse processed Zarr dataset.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on rae.encode/model.forward when possible.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from config.")

    parser.add_argument("--eg-valid-ratio", type=float, default=0.1, help="Validation split ratio in EgoVerse dataset.")
    parser.add_argument("--eg-filter", type=json.loads, default=None, help="Optional filter passed to MultiDataset.")
    parser.add_argument("--eg-pixel-key-idx", type=int, default=0, help="Reserved pixel stream index argument for compatibility.")
    parser.add_argument(
        "--return-future",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether dataset returns (current_frame, future_frame).",
    )
    parser.add_argument(
        "--current-pixel-key",
        type=str,
        default="observations.images.front_img_1",
        help="Current frame pixel key in zarr sample.",
    )
    parser.add_argument(
        "--future-pixel-key",
        type=str,
        default="future.observations.images.front_img_1",
        help="Future frame pixel key in zarr sample.",
    )
    parser.add_argument(
        "--class-label",
        type=int,
        default=0,
        help="Class label used for y-conditioning when training class-conditional stage-2 models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")

    rank, world_size, device = setup_distributed()
    full_cfg = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        _guidance_config,
        misc_config,
        training_config,
        _eval_config,
    ) = parse_configs(full_cfg)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    training_cfg = to_dict(training_config)
    model_params = to_dict(model_config.get("params", {}))

    if not args.return_future:
        raise ValueError("Video prediction training expects --return-future (or omit --no-return-future).")

    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")

    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None

    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        if global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")
    else:
        batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = batch_size * world_size * grad_accum_steps

    num_workers = int(training_cfg.get("num_workers", 4))
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4))
    default_seed = int(training_cfg.get("global_seed", 0))

    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)

    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but current CUDA device does not support bfloat16.")
    scaler, autocast_kwargs = get_autocast_scaler(args)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    full_cfg.cmd_args = vars(args)
    full_cfg.experiment_dir = experiment_dir
    full_cfg.checkpoint_dir = checkpoint_dir

    # Model init
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)

    if args.compile:
        try:
            rae.encode = torch.compile(rae.encode)
        except Exception as exc:
            if rank == 0:
                logger.warning(f"Failed to compile RAE encoder, continuing without compile: {exc}")
        try:
            model.forward = torch.compile(model.forward)
        except Exception as exc:
            if rank == 0:
                logger.warning(f"Failed to compile model.forward, continuing without compile: {exc}")

    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    model.requires_grad_(True)

    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    model = ddp_model.module
    ddp_model.train()

    model_param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        logger.info(f"Model Parameters: {model_param_count / 1e6:.2f}M")

    optimizer, optim_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], training_cfg)

    scheduler: Optional[LambdaLR] = None
    sched_msg: Optional[str] = None

    # Data init
    first_crop_size = 384 if args.image_size == 256 else int(args.image_size * 1.5)
    # stage2_transform = transforms.Compose(
    #     [
    #         transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
    #         transforms.RandomCrop(args.image_size),
    #         transforms.RandomHorizontalFlip(),
    #     ]
    # )
    stage2_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    )

    datasets = load_egoverse_zarr_images_train_val(
        transform=stage2_transform,
        data_path=str(args.data_path),
        valid_ratio=float(args.eg_valid_ratio),
        filter=args.eg_filter,
        mode="train",
        pixel_key_idx=int(args.eg_pixel_key_idx),
        return_future=bool(args.return_future),
        pixel_key=str(args.current_pixel_key),
        future_pixel_key=str(args.future_pixel_key),
    )
    train_dataset = datasets["train"]

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    loader_batches = len(loader)
    if loader_batches == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps

    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)

    # Transport init
    transport_params = dict(transport_cfg.get("params", {}))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")
    # PRETRAINED_STAGE2_MODEL_CKPT_PATH = "models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt"
    # ddp_model.module.load_state_dict(torch.load(PRETRAINED_STAGE2_MODEL_CKPT_PATH, map_location="cpu"))
    # Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
        ckpt_path = Path(maybe_resume_ckpt_path) if maybe_resume_ckpt_path is not None else None

    if ckpt_path is not None:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                str(ckpt_path),
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
            if rank == 0:
                logger.info(f"Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")

    if rank == 0:
        num_params = sum(p.numel() for p in rae.parameters())
        logger.info(f"Stage-1 RAE parameters: {num_params / 1e6:.2f}M")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Stage-2 Model trainable parameters: {num_params / 1e6:.2f}M")
        logger.info(optim_msg)
        logger.info(sched_msg if sched_msg else "No LR scheduler.")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        logger.info(f"Training for {num_epochs} epochs, micro_batch_size={micro_batch_size} per GPU.")
        logger.info(f"Dataset contains {len(train_dataset)} samples, {steps_per_epoch} optimizer steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch}.")
        logger.info(f"Using current key '{args.current_pixel_key}' and future key '{args.future_pixel_key}'.")

    _dist_barrier()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_updates = 0
        running_loss = 0.0

        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            current_images, future_images = batch
            current_images = current_images.to(device, non_blocking=True)
            future_images = future_images.to(device, non_blocking=True)
            with torch.no_grad():
                with autocast(**autocast_kwargs):
                    z_current = rae.encode(current_images)
                    z_future = rae.encode(future_images)

            if step == 0 and epoch == start_epoch:
                if z_current.shape != z_future.shape:
                    raise RuntimeError(
                        f"Latent shape mismatch: z_current={tuple(z_current.shape)} vs z_future={tuple(z_future.shape)}"
                    )
                expected_channels = model_params.get("in_channels", None)
                expected_size = model_params.get("input_size", None)
                if expected_channels is not None and z_current.shape[1] != int(expected_channels):
                    raise RuntimeError(
                        f"Channel mismatch: latent C={z_current.shape[1]}, model in_channels={expected_channels}."
                    )
                if expected_size is not None and z_current.shape[-1] != int(expected_size):
                    raise RuntimeError(
                        f"Spatial mismatch: latent H/W={z_current.shape[-1]}, model input_size={expected_size}."
                    )
                if rank == 0:
                    logger.info(
                        f"Latent sanity check passed: shape={tuple(z_current.shape)}, expected in_channels={expected_channels}, input_size={expected_size}."
                    )

            y = torch.full(
                (z_future.shape[0],),
                int(args.class_label),
                device=device,
                dtype=torch.long,
            )
            model_kwargs = {"y": y, "s": z_current}
            with autocast(**autocast_kwargs):
                loss = transport.training_losses(ddp_model, z_future, model_kwargs)["loss"].mean()

            if scaler is not None:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()

            should_step = ((step + 1) % grad_accum_steps) == 0
            if should_step:
                if clip_grad is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                update_ema(ema_model, ddp_model.module, decay=ema_decay)
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                num_updates += 1

                loss_item = loss.detach().item()
                running_loss += loss_item
                epoch_metrics["loss"] += loss.detach()

                if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                    avg_loss = running_loss / log_interval
                    stats = {
                        "train/loss": avg_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                    logger.info(
                        f"[Epoch {epoch} | Step {global_step}] "
                        + ", ".join(f"{k}: {v:.6f}" for k, v in stats.items())
                    )
                    if args.wandb:
                        wandb_utils.log(stats, step=global_step)
                    running_loss = 0.0

                if sample_every > 0 and global_step % sample_every == 0 and rank == 0:
                    logger.info("Generating conditional future-frame predictions (EMA model)...")
                    with torch.no_grad():
                        n_vis = min(4, z_future.shape[0])
                        vis_noise = torch.randn_like(z_future[:n_vis])
                        vis_y = y[:n_vis]
                        vis_kwargs = {"y": vis_y, "s": z_current[:n_vis]}
                        with autocast(**autocast_kwargs):
                            pred_z = eval_sampler(vis_noise, ema_model.forward, **vis_kwargs)[-1]
                        pred = rae.decode(pred_z).cpu().float().clamp(0, 1)
                        cur = current_images[:n_vis].cpu().float().clamp(0, 1)
                        fut = future_images[:n_vis].cpu().float().clamp(0, 1)
                        triplets = torch.stack([cur, fut, pred], dim=1)   # row-wise groups
                        vis = triplets.reshape(-1, *cur.shape[1:])        # [cur0,fut0,pred0, cur1,fut1,pred1, ...]
                        if args.wandb:
                            wandb_utils.log_image(vis, step=global_step)
                    logger.info("Conditional prediction sampling done.")

        if rank == 0 and num_updates > 0:
            avg_loss = (epoch_metrics["loss"] / num_updates).item()
            epoch_stats = {
                "epoch/loss": avg_loss,
            }
            logger.info(
                f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.6f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)

    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt"
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )

    _dist_barrier()
    if rank == 0:
        logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
