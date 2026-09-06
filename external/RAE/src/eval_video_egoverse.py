# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate EgoVerse future-frame prediction and render side-by-side videos.

For each sample, the script predicts a future frame from the current frame and
writes a video where each frame is concatenated as:
    [ground_truth_future | predicted_future]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import write_video

from dataset.egoverse_zarr_dset import load_egoverse_zarr_images_from_config
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import Sampler, create_transport
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EgoVerse video prediction and save side-by-side video.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Stage-2 checkpoint (.pt).")
    parser.add_argument("--data-config", type=str, default=None, help="Optional dataset config YAML override.")

    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "total"], help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames in output video.")
    parser.add_argument("--fps", type=int, default=8, help="Output video FPS.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument("--image-size", type=int, default=256, help="Evaluation resolution.")

    parser.add_argument("--class-label", type=int, default=0, help="Class label for y-conditioning.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True, help="Load EMA weights from checkpoint if available.")
    parser.add_argument("--output-video", type=str, default="outputs/egoverse_eval_gt_vs_pred.mp4", help="Path to output side-by-side video.")
    return parser.parse_args()


def get_autocast_kwargs(precision: str) -> Dict[str, Any]:
    if precision == "fp16":
        return {"enabled": True, "dtype": torch.float16}
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 requested, but CUDA device does not support bfloat16.")
        return {"enabled": True, "dtype": torch.bfloat16}
    return {"enabled": False}


def load_model_weights(model: Stage2ModelProtocol, ckpt_path: str, use_ema: bool) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if use_ema and "ema" in ckpt:
            model.load_state_dict(ckpt["ema"], strict=True)
            return "ema"
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
            return "model"
    # Fallback for raw state dict checkpoints
    model.load_state_dict(ckpt, strict=True)
    return "raw"


def tensor_to_video_frame_uint8(frame: torch.Tensor) -> torch.Tensor:
    # frame: (C,H,W) float in [0,1] -> (H,W,C) uint8
    frame = frame.clamp(0, 1)
    frame = (frame * 255.0).round().to(torch.uint8)
    return frame.permute(1, 2, 0).cpu()


def save_side_by_side_video(frames: List[torch.Tensor], output_path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    video = torch.stack(frames, dim=0)  # (T,H,W,C), uint8
    try:
        write_video(output_path, video, fps=fps)
        print(f"Saved video to: {output_path}")
    except Exception as exc:
        # Fallback: save frames as PNGs.
        fallback_dir = os.path.splitext(output_path)[0] + "_frames"
        os.makedirs(fallback_dir, exist_ok=True)
        from PIL import Image

        for i, frame in enumerate(video):
            Image.fromarray(frame.numpy()).save(os.path.join(fallback_dir, f"frame_{i:06d}.png"))
        print(f"Video write failed ({exc}). Saved PNG frames to: {fallback_dir}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This evaluator currently expects CUDA.")

    full_cfg = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        _guidance_config,
        misc,
        _,
        _,
    ) = parse_configs(full_cfg)

    if rae_config is None or model_config is None:
        raise ValueError("Config must include stage_1 and stage_2 sections.")
    if misc is None:
        misc = {}

    data_config_path = args.data_config or full_cfg.get("data_config")
    if not data_config_path:
        raise ValueError("No dataset config found. Set data_config in the training config or pass --data-config.")

    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    rae.eval()
    model.eval()

    loaded_from = load_model_weights(model, args.ckpt, use_ema=bool(args.use_ema))
    print(f"Loaded Stage-2 weights from {loaded_from} in checkpoint: {args.ckpt}")

    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    transport = create_transport(
        **transport_config["params"],
        time_dist_shift=time_dist_shift,
    )
    sampler = Sampler(transport)

    sampler_mode = sampler_config["mode"].upper()
    sampler_params = sampler_config["params"]
    if sampler_mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    # Deterministic evaluation transform.
    eval_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
        ]
    )

    data_cfg_full = OmegaConf.load(data_config_path)
    data_cfg = OmegaConf.to_container(data_cfg_full, resolve=True)
    try:
        datasets = load_egoverse_zarr_images_from_config(
            data_cfg=data_cfg,
            transform=eval_transform,
            mode=args.split if args.split != "total" else "valid",
            return_future=True,
            pixel_key=str(data_cfg_full.get("pixel_key", "observations.images.front_img_1")),
            future_pixel_key=str(data_cfg_full.get("future_pixel_key", "future.observations.images.front_img_1")),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset from config {data_config_path}: {exc}")

    if args.split == "total":
        dataset = datasets["valid"]
    else:
        dataset = datasets[args.split]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    autocast_kwargs = get_autocast_kwargs(args.precision)

    frames: List[torch.Tensor] = []
    total = 0

    for current_images, future_images in loader:
        current_images = current_images.to(device, non_blocking=True)
        future_images = future_images.to(device, non_blocking=True)

        with autocast(**autocast_kwargs):
            z_current = rae.encode(current_images)

        n = z_current.shape[0]
        y = torch.full((n,), int(args.class_label), device=device, dtype=torch.long)

        model_kwargs: Dict[str, Any] = {"y": y, "s":z_current}
        z_noise = torch.randn_like(z_current)
        with autocast(**autocast_kwargs):
            pred_z = sample_fn(z_noise, model.forward, **model_kwargs)[-1]
            pred = rae.decode(pred_z)

        gt = future_images.clamp(0, 1)
        pred = pred.clamp(0, 1)

        for i in range(gt.shape[0]):
            side_by_side = torch.cat([gt[i], pred[i]], dim=2)  # along width
            frames.append(tensor_to_video_frame_uint8(side_by_side))
            total += 1
            if args.max_frames is not None and total >= args.max_frames:
                break
        if args.max_frames is not None and total >= args.max_frames:
            break

    if len(frames) == 0:
        raise RuntimeError("No frames were generated. Check dataset path/filter/split.")

    save_side_by_side_video(frames, args.output_video, args.fps)
    print(f"Rendered {len(frames)} side-by-side frames (GT | Pred).")


if __name__ == "__main__":
    main()
