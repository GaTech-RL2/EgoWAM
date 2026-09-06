# Track4World 3D-flow extraction (minimal pipeline)

Generates the 3D point-flow supervision (`sampled_{uv,points_t,points_tk,proj_flow_2d}.npy`
+ `metadata.json` + `_SUCCESS`) consumed by the WM 3D-flow heads, using the
[Track4World](https://github.com/TencentARC/Track4World) model.

## Layout

- `external/Track4World` (vendored) — upstream Track4World: the model
  (`track4world.nets`, `.utils`, `config`).
- `3dflow_process/overlay/track4world/` — egoverse glue upstream lacks:
  `data/` (aria/eva zarr loaders, `list_episode_segments`) and
  `utils/flow_vis.py`.
- `3dflow_process/scripts/` — the two compute entry points + the bulk Slurm
  submitters + `flow_extraction.md` (the how-to).

Third-party deps (`utils3d`, the `pi3` backbone) are **not** vendored here — they
are part of the upstream Track4World install (README "Install Third-Party
Modules") and are cloned into the Track4World checkout by `sync_overlay.sh`.

`sync_overlay.sh` symlinks the egoverse overlay into that checkout so
`import track4world.data…` / `…nets.model` resolve (overlay tracked here = single
source of truth; the clone stays pristine upstream) and clones
`utils3d` + `pi3` into it. The submitters export `PYTHONPATH=<Track4World root>`
so `track4world`/`utils3d` import regardless of CWD (`python scripts/x.py` does
not put that root on `sys.path`).

## Setup

```bash
conda activate track4world                 # env with torch + track4world deps
bash 3dflow_process/sync_overlay.sh        # idempotent; rerun if overlay changes
```

The model checkpoint (`track4world_da3.pth`, not in git) is symlinked from an
existing local copy if present; otherwise place it at
`external/Track4World/checkpoints/track4world_da3.pth`.

## Run

See `scripts/flow_extraction.md`. Quick verify (no GPU/model, lists segments):

```bash
bash 3dflow_process/scripts/submit_bulk_eva_3d_flow_slurm.sh \
  --dataset-root <src-zarrs> --output-root <src-zarrs>_3d_flow_track4world \
  --dataset-name <name> --image-size 448 --dry-run
```

Drop `--dry-run` to submit the Slurm array. Use the `aria` submitter for
head-pose-compensated (Aria) datasets, `eva` for fixed-camera (EVA / libero).
