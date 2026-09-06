# RAE stage 1 — training the decoder

The **DINO** world-model variant predicts frozen DINOv2 features. To *see* what it
predicts — and to compute the reconstruction metrics in the paper — those features have to
be decoded back to pixels. That decoder is the RAE (Representation Autoencoder) stage-1
model, and this directory trains it.

Stage 1 is a prerequisite for the DINO variant, not part of it: its checkpoint is what
`model/hpt_wm_{eva,cotrain}_rae.yaml` load through `EGOWAM_RAE_CKPT_ROOT`. Stage 2 (the
DiT-DH denoiser) is a separate upstream entrypoint and is not wrapped here.

```bash
uv pip install -e '.[rae]'      # once: pulls torch-fidelity
sbatch egowam/scripts/rae_process/train_stage1.sh [tag]
```

`torch-fidelity` is a hard requirement even though stage 1 computes no FID: the training
script imports RAE's `eval` package at module scope, and that package imports it.

Stage 1 also needs the GAN discriminator's backbone, which is a public DINO ViT-S/8
checkpoint that is not vendored (`gan.disc.arch.dino_ckpt_path` in the stage-1 config):

```bash
mkdir -p external/RAE/models/discs
curl -fsSL https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth \
  -o external/RAE/models/discs/dino_vit_small_patch8_224.pth
```

RAE's own `.gitignore` covers `models/`, so it stays out of git.

Everything runs against the vendored `external/RAE`; nothing in that tree is modified, so
it stays byte-faithful to its base commit for provenance (see `external/README.md`).

## Layout

| Path | What it is |
|---|---|
| `train_stage1.py` | Launcher. Delegates to `external/RAE/src/train_stage1_egoverse.py`. |
| `rae_stage1_data.py` | The config-driven dataset loader that upstream is missing. |
| `data/eva_aria_cotrain.yaml` | Stage-1 data config: EVA + Aria, images only. |
| `train_stage1.sh` | Cluster launcher. |

## Two upstream gaps this closes

Both are properties of the vendored commit, and both are worked around here rather than
patched into `external/RAE`:

1. **`train_stage1_egoverse.py` imports a function that does not exist.** It asks for
   `load_egoverse_zarr_images_from_config` from `dataset.egoverse_zarr_dset`, which only
   defines `load_egoverse_zarr_images_train_val` — the caller was moved to a config-driven
   API and the loader never followed. `rae_stage1_data.py` supplies the missing function
   and `train_stage1.py` injects it before the training script is imported.

2. **`egoverse_zarr_dset.py` imports `egomimic.*`**, this package's pre-rename name.
   `train_stage1.py` calls `egowam.compat.install_legacy_module_alias()` first, so those
   resolve to `egowam.*`.

## The data config is deliberately not a Hydra config

The RAE script loads `--data-config` with OmegaConf directly, outside Hydra. A config from
`egowam/hydra_configs/data/` would fail: `${paths.data_root}` has no `paths` group to
interpolate against. `data/eva_aria_cotrain.yaml` uses `${oc.env:EGOWAM_DATA_ROOT,...}`
instead, which OmegaConf resolves on its own, and otherwise carries the same episodes and
filters as `hydra_configs/data/cup_saucer/indomain_wm.yaml`.

It reads only `pixel_key` / `future_pixel_key` images. The action and proprio entries in
the key maps are there because the resolver needs a complete map, not because stage 1
uses them.

## Knobs

| Variable | Effect |
|---|---|
| `EGOWAM_DATA_ROOT` | where the zarr datasets live (used by the data config) |
| `EGOWAM_RAE_ROOT` | an RAE checkout other than `external/RAE` |
| `EGOWAM_RAE_CKPT_ROOT` | where stage-1 results are written |
| `RAE_STAGE1_DATA` | a different stage-1 data config |
| `WANDB_API_KEY` | unset means `WANDB_MODE=offline`; no key is stored in the repo |
| `EXPERIMENT_NAME` | run directory under the results root; defaults to `stage1_<tag>` |
| `ENTITY`, `PROJECT` | W&B destination, read by RAE only when `--wandb` is passed |
| `NPROC`, `IMAGE_SIZE`, `PRECISION` | torchrun processes / resolution / dtype |

`--finetune` is passed by default, starting from the pretrained DINOv2-B decoder as the
released checkpoints did. Drop it from the launcher to train the decoder cold.
