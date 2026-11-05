"""Track4World model loading for the 3D-flow extraction scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)

COORDINATES = ("camera_base", "world_pi3", "world_depthanythingv3")


def load_model_from_config(
    config_path: str | Path,
    ckpt_init: str | Path,
    coordinate: str = "world_depthanythingv3",
    device: str | torch.device = "cuda",
    use_original_backbone: bool = False,
) -> torch.nn.Module:
    """Build Track4World from an eval config, load weights, freeze, and eval().

    Args:
        config_path: eval JSON with a ``model`` section, e.g.
            ``track4world/config/eval/v1.json``.
        ckpt_init: checkpoint matching ``coordinate``, e.g.
            ``checkpoints/track4world_da3.pth`` for ``world_depthanythingv3``.
        coordinate: one of ``COORDINATES``; its last ``_`` token picks the
            depth backbone (``base`` / ``pi3`` / ``depthanythingv3``).
        device: where to place the model.
        use_original_backbone: same as demo.py's ``--use_original_backbone``.
    """
    from track4world.nets.model import Track4World

    if coordinate not in COORDINATES:
        raise ValueError(f"coordinate must be one of {COORDINATES}, got {coordinate!r}")

    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    ckpt_init = Path(ckpt_init)
    if not ckpt_init.is_file():
        raise FileNotFoundError(
            f"Track4World checkpoint not found: {ckpt_init}. Download it with: "
            "hf download TencentARC/Track4World track4world_da3.pth "
            "--local-dir external/Track4World/checkpoints"
        )

    LOGGER.info("Building Track4World (%s) from %s", coordinate, config_path)
    model = Track4World(
        **config["model"],
        seqlen=16,
        use_3d=True,
        use_model=coordinate.split("_")[-1],
    )

    LOGGER.info("Loading weights from %s", ckpt_init)
    state_dict = torch.load(ckpt_init, map_location="cpu")
    # strict=False as in demo.py; report the mismatch so a wrong checkpoint for
    # the chosen coordinate shows up in the log instead of passing silently.
    result = model.load_state_dict(state_dict, strict=False)
    LOGGER.info(
        "Checkpoint keys: %d missing, %d unexpected",
        len(result.missing_keys),
        len(result.unexpected_keys),
    )
    if use_original_backbone:
        model.switch_to_original_backbone()

    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
