# external/

Third-party code EgoWAM builds on. Everything here is **vendored**: these are the working trees the
paper's runs actually imported, committed as ordinary files. There are no submodules to initialize
and no patches to apply — clone this repository and the code is present.

That is a deliberate reversal of the usual advice, for a concrete reason. Our copies of
`cosmos-policy` and `RAE` had drifted thousands of lines from their upstreams, and the drift lived
in commits on personal forks. A submodule pinned at those SHAs against the *public* upstream URL
resolves to nothing, so the checkout that produced the results could not be reconstructed by anyone
— including us. Vendoring makes the released code self-contained and exactly what ran.

## What each one is for

| Directory | Needed for | What it provides |
|---|---|---|
| `cosmos-policy` | **Pixel**, **Pixel-PT** | Wan 2.1 video VAE, the frozen target encoder for pixel-space world prediction. |
| `RAE` | **DINO** | Representation Autoencoder: the frozen DINOv2 encoder that defines the target, plus the DiT-DH denoiser used as the head. |
| `RoboTwin` | simulation | RoboTwin 2.0 SAPIEN benchmark, carrying our EgoWAM policy bridge. See `docs/robotwin.md`. |
| `Track4World` | regenerating 3D-flow targets | Feed-forward dense 3D point tracker. Only needed to *produce* flow sidecars — training the 3D-flow variant on the shipped sidecars does not touch it. |
| `rpl_vision_utils` | camera calibration | AprilTag detection used by `egowam/scripts/calibrate_camera/`. |

**BC and 3D-flow need none of these.** They are only on the import path of the variant that uses
them, so a missing dependency never blocks the variants that do not need it.

## Provenance

Each tree is upstream at a known commit plus our changes. To see exactly what we changed, clone the
upstream at the base commit and diff it against the directory here.

| Directory | Upstream | Base commit | Our divergence |
|---|---|---|---|
| `cosmos-policy` | [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy) | `18a2acc` | 11 commits, 45 files, +7711 / -33 |
| `RAE` | [bytetriper/RAE](https://github.com/bytetriper/RAE) | `a4d18c4` | 5 commits, 26 files, +3822 / -7 |
| `RoboTwin` | [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | `c3ddfa8` | 6 files patched, 8 files added |
| `Track4World` | [TencentARC/Track4World](https://github.com/TencentARC/Track4World) | `5b59b0c` | 5 files patched |
| `rpl_vision_utils` | UT-Austin RPL | — | unmodified |

Each upstream's own `LICENSE` is kept in place; vendoring changes nothing about their terms.

**cosmos-policy** — a `device` kwarg threaded through the Wan 2.1 VAE interface (what
`egowam/models/cosmos_vae.py` calls), plus the EgoVerse dataset adapter, configs and launch scripts
used to train the Pixel-PT checkpoints.

**RAE** — the EgoVerse video/image datasets and their Stage-1 and Stage-2 training entrypoints,
plus the stage-2 DDT hook that `egowam/models/rae_vae.py` loads.

**Track4World** — two things upstream tracks are not here: its paper PDF (77 MB under `assets/`;
it is on arXiv), and eleven demo videos, README images and the `bremm.png` colormap, which upstream
distributes over Git LFS and which were never materialized in the checkout we ran — all we had were
132-byte pointer stubs, so there was nothing to vendor. None is on the flow-extraction path. The
missing colormap is in fact why `improc.py` needed the fallback below. Our changes are a colormap-loading fallback in
`track4world/utils/improc.py` (visualization only, not flow numerics) and a reworked
`scripts/run_dino_sam2.py`. The EgoVerse glue — the Aria/EVA zarr loaders and `flow_vis` — lives in
`egowam/scripts/flow_extraction/overlay/` instead, and that directory's `sync_overlay.sh` symlinks it in
so there is one source of truth for it.

**RoboTwin** — our additions are confined to `policy/EgoWAM/` (the bridge RoboTwin's
`script/eval_policy.py` drives), `policy/{ACT,DP}/process_data_ee.py` (the ACT-EE and DP-EE
baselines, rewritten to the 14-D end-effector action space), and two dataset inspection helpers.
Patched upstream files: `envs/_base_task.py` (EE action path and episode-termination hooks),
`policy/ACT/{constants,imitate_episodes}.py` (EE action space in ACT training), `script/test_render.py`
(renderer diagnostics for bad Vulkan/EGL nodes), and `task_config/demo_{clean,randomized}.yml`
(the eval-split settings behind the paper's `demo_clean-50` runs).

## What is deliberately not here

Vendoring covers *code*. Three categories are excluded, and each has its own way in:

- **Model weights.** DINO needs a trained RAE encoder checkpoint (`EGOWAM_RAE_CKPT_ROOT`). Pixel
  needs the Wan/Cosmos tokenizer, which lives in a **gated** HuggingFace repo — request access,
  then point `HF_HOME` at a cache holding it, or you will see `GatedRepoError: 401`.
- **RoboTwin's simulation assets and `envs/curobo`.** ~200 MB of meshes, textures and a compiled
  motion-planning dependency that upstream fetches with its own setup scripts. Follow RoboTwin's
  install, then this tree is complete.
- **The Track4World model checkpoint** (`track4world_da3.pth`). Only needed to regenerate flow
  sidecars from raw video; place it at `external/Track4World/checkpoints/`. See
  `egowam/scripts/flow_extraction/README.md`.

## Pointing elsewhere

Already have a checkout you would rather use? Set the variable instead of editing configs:

| Variable | Replaces |
|---|---|
| `EGOWAM_RAE_ROOT` | `external/RAE` |
| `EGOWAM_ROBOTWIN_ROOT` | `external/RoboTwin` |

If a dependency really is missing, the code raises with the path it wanted and how to satisfy it
rather than a bare `ImportError` — see `egowam/external_deps.py`.
