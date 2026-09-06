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
| `HF_HOME` | HuggingFace cache (Pixel variant only) |
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

### RAE checkpoints — DINO variant

The DINO target decodes into RAE latent space and needs a stage-1 decoder checkpoint. Published
weights are on the Hub, so this no longer requires training your own:

```bash
hf download boeyyyy/EgoWAM-checkpoints rae_stage1/ep-0000002.pt --local-dir ./ckpts

export EGOWAM_RAE_ROOT=external/RAE                  # the vendored code
export EGOWAM_RAE_CKPT_ROOT=./ckpts/rae_stage1       # the downloaded decoder
```

To train your own instead, see [training.md](training.md#rae-stage-1-optional) — a multi-day run:
the full mixture is ~5.06M samples at ~158k steps/epoch.

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
