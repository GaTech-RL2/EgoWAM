#!/usr/bin/env python3
"""
Run a stage-1 RAE reconstruction from a config file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE

DEFAULT_IMAGE = Path("assets/pixabay_cat.png")


def get_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, C, H, W)
    return tensor


def reconstruct(rae: RAE, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        latent = rae.encode(image)
        recon = rae.decode(latent)
    return latent, recon


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct an input image using a Stage-1 RAE loaded from config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config with a stage_1 section.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image to reconstruct (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recon.png"),
        help="Where to save the reconstructed image (default: recon.png).",
    )
    parser.add_argument(
        "--device",
        help="Torch device to use (e.g. cuda, cuda:1, cpu). Auto-detect if omitted.",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {args.config}. "
            "Please supply a config with a stage_1 target."
        )

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    image = load_image(args.image).to(device)
    latent, recon = reconstruct(rae, image)

    recon = recon.clamp(0.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(recon, args.output)

    print(f"Saved reconstruction to {args.output.resolve()}")
    print(f"Input shape: {tuple(image.shape)}, latent shape: {tuple(latent.shape)}, recon shape: {tuple(recon.shape)}")


if __name__ == "__main__":
    main()
