# 🛠️ Installation

- [Environment](#environment)
- [Configure paths](#configure-paths)
- [Gated and third-party assets](#gated-and-third-party-assets)
- [Optional stacks](#optional-stacks)

Tested on Python 3.11, PyTorch 2.5, CUDA 12.1, A40 / L40S GPUs.

## Environment

Managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GaTech-RL2/EgoWAM.git && cd EgoWAM
uv venv emimic --python 3.11
source emimic/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv run pre-commit install          # optional, for contributors
```

Verify the package resolves to this checkout:

```bash
python -c "import egowam; print(egowam.__file__)"
```

**Third-party code is vendored, not submoduled.** The Wan VAE (`cosmos-policy`), the RAE encoder
and DiT-DH head (`RAE`), Track4World, and the RoboTwin benchmark with our policy bridge all arrive
with the clone, exactly as the paper's runs imported them. Nothing to initialize.

Two things `external/` does *not* carry: model weights and RoboTwin's simulation assets. See
`external/README.md`.

## Configure paths

Every dataset and checkpoint location resolves through a token in
`egowam/hydra_configs/paths/default.yaml`, each overridable by environment variable. Set these once
rather than editing configs:

| Variable | Contents |
|---|---|
| `EGOWAM_DATA_ROOT` | converted zarr datasets, one directory per dataset |
| `EGOWAM_FLOW_ROOT` | Track4World 3D-flow sidecars (defaults to `EGOWAM_DATA_ROOT`) |
| `EGOWAM_CKPT_ROOT` | trained run directories |
| `EGOWAM_RAE_ROOT`, `EGOWAM_RAE_CKPT_ROOT` | RAE assets for the DINO target |
| `EGOWAM_ROBOTWIN_ROOT` | RoboTwin sim scratch |
| `EGOWAM_ROBOTWIN_FLOW_ROOT` | RoboTwin 3D-flow sidecars (sibling tree to the dataset root) |
| `HF_HOME` | HuggingFace cache — the Pixel, Pixel-PT and DINO variants all pull weights from the Hub |
| `WANDB_ENTITY`, `WANDB_PROJECT` | logging destination |

## Gated and third-party assets

Only some variants need these. **BC and 3D-flow need none of them** and train from the dataset alone.

### Wan/Cosmos tokenizer — Pixel variant

The tokenizer lives in a **gated** HuggingFace repo. Request access at
[`nvidia/Cosmos-Predict2-2B-Video2World`](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World),
then:

```bash
hf auth login
python egowam/scripts/check_gated_assets.py     # ~2s, exits non-zero if unreachable
```

`sbatch/hpt_wm_vae.sh` and `sbatch/robotwin/train_rt.sh vae` run this preflight automatically. It
matters because the VAE is only built *after* the data sync and normalization pass — without the
preflight, a missing-access run fails with `GatedRepoError: 401` more than an hour in.

### Wan2.1 video weights — Pixel-PT variant

Pixel-PT initializes its future-image head from Wan2.1. Both repos are **public**, so there is no
access request and no preflight — but the pull is several GB, and on clusters that police login-node
processes it is worth fetching from a compute node first:

```bash
export HF_HOME=/path/to/shared/cache   # must be visible from compute nodes
python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B",
                  allow_patterns=["diffusion_pytorch_model*", "**/diffusion_pytorch_model*",
                                  "config.json", "**/config.json", "model_index.json", "**/*.json"],
                  ignore_patterns=["*t5*", "*umt5*"])
hf_hub_download(repo_id="Wan-AI/Wan2.1-VACE-1.3B", filename="Wan2.1_VAE.pth")
EOF
```

The text encoders are excluded deliberately — the config sets `no_text_conditioning: true`. The head
itself comes from `diffusers` (already pinned in `requirements.txt`), not from a separate checkout.

### RAE checkpoints — DINO variant

The DINO target decodes into RAE latent space, which takes **two** sets of weights: our fine-tuned
stage-1 decoder, and the upstream RAE assets its config loads around it. Neither is vendored —
RAE's own `.gitignore` covers `models/`.

**1. Our stage-1 decoder.** Published on the Hub, so this no longer requires training your own:

```bash
hf download boeyyyy/EgoWAM-checkpoints rae_stage1/ep-0000002.pt --local-dir ./ckpts
```

**2. Upstream RAE assets.** `hpt_wm_{eva,cotrain}_rae.yaml` point at RAE's own
`configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml`, and that config names three files by
path *relative to the RAE root* — so they must land under `$EGOWAM_RAE_ROOT/models/`:

```bash
hf download nyu-visionx/RAE-collections \
  decoders/dinov2/wReg_base/ViTXL_n08/model.pt \
  stats/dinov2/wReg_base/imagenet1k/stat.pt \
  DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt \
  --local-dir external/RAE/models
```

| File | Size | Why it is needed |
|---|---|---|
| `decoders/.../ViTXL_n08/model.pt` | 1.6 GB | `pretrained_decoder_path` — read when the stage-1 model is constructed. Our checkpoint then overwrites every decoder weight (`strict=True`), so the values do not survive, but the file must exist or construction raises. |
| `stats/.../imagenet1k/stat.pt` | 803 KB | `normalization_stat_path` — the *only* thing that sets `do_normalization=True`. Without it latents are left unnormalized and the world-model targets change scale, so this small file is load-bearing, not optional. |
| `DiTs/.../DiTDH-XL/stage2_model.pt` | 3.2 GB | Our configs set `use_stage2_features: true`, so the stage-2 DiT is built and its `ckpt` loaded. |

Omit the file list to fetch the whole collection (`--local-dir external/RAE/models` alone); the
extra decoders and DiTs are for encoders this repo does not use.

**3. The DINOv2 encoder** (`facebook/dinov2-with-registers-base`) is pulled from the Hub
automatically by `from_pretrained` — weights, config and image processor. It is public and needs no
access request, but the download happens at model-build time, so on an offline compute node warm
`HF_HOME` first.

Then point both roots at what you downloaded:

```bash
export EGOWAM_RAE_ROOT=external/RAE                  # vendored code + models/ from step 2
export EGOWAM_RAE_CKPT_ROOT=./ckpts/rae_stage1       # the stage-1 decoder from step 1
```

Set `EGOWAM_RAE_ROOT` explicitly: its default in `paths/default.yaml` is an internal cluster path,
and leaving it unset sends every one of the relative lookups above somewhere you cannot read.

Training your own stage-1 decoder needs one more file — the GAN discriminator's DINO ViT-S/8
backbone, also in this collection as `discs/dino_vit_small_patch8_224.pth` (83 MB). See
[training.md](training.md#rae-stage-1-optional) — a multi-day run: the full mixture is ~5.06M
samples at ~158k steps/epoch.

## Optional stacks

### Simulation

RoboTwin needs SAPIEN and mplib, which conflict with the training environment. Build it separately
as `.venv-sim` — see [robotwin.md](robotwin.md).

### Real robot

The ARX5 + Aria + RealSense stack under `egowam/robot/` needs vendor SDKs; see
[hardware.md](hardware.md).

### 3D-flow extraction

Regenerating flow targets from RGB uses Track4World, which has its own environment. See
[data.md](data.md#3d-flow-targets).
