#!/usr/bin/env python3
"""Run RAE stage-1 (decoder) training against the vendored external/RAE.

Stage 1 trains the RAE *decoder* that reconstructs images from frozen DINOv2 features.
Its output checkpoint is what the DINO world-model variant loads at train time via
``EGOWAM_RAE_CKPT_ROOT`` (see model/hpt_wm_{eva,cotrain}_rae.yaml). Stage 2 -- the DiT-DH
denoiser -- is a separate upstream entrypoint and is not wrapped here.

This is a launcher, not a reimplementation: it delegates to
``external/RAE/src/train_stage1_egoverse.py`` and only fixes the two things that stop
that script from importing in this repo, both without editing the vendored tree:

  1. ``dataset/egoverse_zarr_dset.py`` imports ``egomimic.*``, the pre-rename package
     name. egowam.compat installs a meta-path alias so those resolve to ``egowam.*``.
  2. The script imports ``load_egoverse_zarr_images_from_config``, which that module does
     not define. rae_stage1_data supplies it and it is injected into the module namespace
     before the training script is imported.

Usage mirrors the upstream script; everything after ``--`` is forwarded verbatim:

    python egowam/scripts/rae_process/train_stage1.py -- \
        --config configs/stage1/training/DINOv2-B_decXL.yaml \
        --data-config <repo>/egowam/scripts/rae_process/data/eva_aria_cotrain.yaml \
        --results-dir ckpts/stage1_$(date +%Y%m%d_%H%M%S) \
        --image-size 256 --precision bf16 --finetune
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _rae_root() -> Path:
    from egowam.external_deps import RAE

    return Path(RAE.require())


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]

    rae_root = _rae_root()
    src = rae_root / "src"
    if not (src / "train_stage1_egoverse.py").is_file():
        raise SystemExit(f"RAE stage-1 entrypoint not found under {src}")

    # The upstream script uses src-relative imports (`from dataset...`, `from utils...`)
    # and config paths relative to the RAE root, so run it from there.
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(rae_root)

    from egowam.compat import install_legacy_module_alias

    install_legacy_module_alias()

    # Inject the missing loader before the training script imports it by name.
    import dataset.egoverse_zarr_dset as zarr_dset

    from egowam.scripts.rae_process.rae_stage1_data import (
        load_egoverse_zarr_images_from_config,
    )

    zarr_dset.load_egoverse_zarr_images_from_config = (
        load_egoverse_zarr_images_from_config
    )

    sys.argv = ["train_stage1_egoverse.py", *argv]
    runpy.run_path(str(src / "train_stage1_egoverse.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
