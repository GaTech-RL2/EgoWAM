#!/usr/bin/env python3
"""
Run Stage-1 RAE reconstructions on EgoVerse zarr samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from dataset.egoverse_zarr_dset import load_egoverse_zarr_images_train_val
from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def get_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_autocast_kwargs(precision: str, device: torch.device) -> Dict[str, object]:
    if precision == "fp16":
        return {"enabled": device.type == "cuda", "dtype": torch.float16}
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
        return {"enabled": device.type == "cuda", "dtype": torch.bfloat16}
    return {"enabled": False}


def normalize_filter(value: Optional[str]) -> Optional[Dict]:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--eg-filter must decode to a JSON dictionary.")
    return parsed


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    )


def reconstruct(rae: RAE, images: torch.Tensor, autocast_kwargs: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        with autocast(**autocast_kwargs):
            latents = rae.encode(images)
            recon = rae.decode(latents)
    return latents, recon


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct EgoVerse zarr images using a Stage-1 RAE loaded from config."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config with a stage_1 section.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the EgoVerse zarr root.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1_egoverse_samples"), help="Directory to save reconstructions.")
    parser.add_argument("--split", choices=["train", "valid"], default="valid", help="Dataset split to sample from.")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to reconstruct.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used for inference.")
    parser.add_argument("--image-size", type=int, default=256, help="Input resolution before reconstruction.")
    parser.add_argument("--eg-valid-ratio", type=float, default=0.1, help="Validation split ratio used by the EgoVerse dataset wrapper.")
    parser.add_argument("--eg-filter", type=str, default=None, help="Optional JSON dictionary filter for processed dataset subfolders.")
    parser.add_argument("--eg-pixel-key-idx", type=int, default=0, help="Reserved for compatibility with the training script.")
    parser.add_argument("--pixel-key", type=str, default="observations.images.front_img_1", help="Observation image key to reconstruct.")
    parser.add_argument("--device", help="Torch device to use (e.g. cuda, cuda:1, cpu). Auto-detect if omitted.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32", help="Autocast precision to use during inference.")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = get_device(args.device)
    autocast_kwargs = get_autocast_kwargs(args.precision, device)

    if not args.data_path.exists():
        raise FileNotFoundError(f"EgoVerse data path not found: {args.data_path}")

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(f"No stage_1 section found in config {args.config}. Please supply a config with a stage_1 target.")

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    transform = build_transform(args.image_size)
    datasets = load_egoverse_zarr_images_train_val(
        transform=transform,
        data_path=str(args.data_path),
        valid_ratio=float(args.eg_valid_ratio),
        filter=normalize_filter(args.eg_filter),
        mode=args.split,
        pixel_key_idx=int(args.eg_pixel_key_idx),
        pixel_key=args.pixel_key,
    )
    dataset = datasets[args.split]
    if len(dataset) == 0:
        raise ValueError(f"No EgoVerse samples found for split='{args.split}' at {args.data_path}.")

    sample_count = min(args.num_samples, len(dataset))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    collected = 0
    overview_images: List[torch.Tensor] = []

    for images in loader:
        if collected >= sample_count:
            break

        images = images[: sample_count - collected].to(device, non_blocking=True)
        latents, recons = reconstruct(rae, images, autocast_kwargs)
        recons = recons.clamp(0.0, 1.0)
        images = images.clamp(0.0, 1.0)

        for index in range(images.shape[0]):
            sample_id = collected + index
            pair = torch.cat([images[index], recons[index]], dim=2).detach().cpu()
            save_image(pair, args.output_dir / f"sample_{sample_id:06d}.png")
            overview_images.append(pair)

        print(
            f"Batch {collected // args.batch_size + 1}: "
            f"input shape={tuple(images.shape)}, latent shape={tuple(latents.shape)}, recon shape={tuple(recons.shape)}"
        )
        collected += images.shape[0]

    if not overview_images:
        raise RuntimeError("No reconstructions were generated. Check the dataset path and sampling settings.")

    overview = make_grid(torch.stack(overview_images, dim=0), nrow=1)
    save_image(overview, args.output_dir / "overview.png")
    print(f"Saved {collected} reconstructions to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()