# Flow Data Extraction for Egoverse Datasets

Two pipelines, picked by source zarr layout. Both submit a Slurm array via a
thin bulk wrapper; both skip already-completed segments.

## Aria layout (head-pose-compensated)

Source has `images.front_1`, `obs_head_pose`, `obs_eye_gaze`. Multi-segment via
`annotations`.

```bash
bash scripts/submit_bulk_aria_3d_flow_slurm.sh \
  --dataset-root /coc/flash7/scratch/egowm/egoverseS3ZarrDataset/<dataset> \
  --output-root  /coc/flash7/scratch/egowm/egoverseS3ZarrDataset/<dataset>_3d_flow_track4world \
  --dataset-name <dataset> \
  --job-name     aria-3d-flow-<short> \
  --exclude      "qt-1,chitti,sonny"
```

## EVA / libero / static-cam layout (fixed camera)

Source has `images.front_1`; no head pose. One segment per store.

```bash
bash scripts/submit_bulk_eva_3d_flow_slurm.sh \
  --dataset-root <path-to-source-zarrs> \
  --output-root  <path-to-source-zarrs>_3d_flow_track4world \
  --dataset-name <dataset> \
  --job-name     eva-3d-flow-<short> \
  --exclude      "qt-1,chitti,sonny"
```

## Required overrides

- `--image-size` must be ≥ 448 (DINOv2 stride 14 × 5-level pyramid needs ≥32
  patches per side). Native-128 libero data: pass `--image-size 448`.
- All other defaults (`--rate 16`, `--frame-skip 30/45`, `--inference-iters 4`)
  match the trained models.

## Dry-run first

Append `--dry-run` to either command. Prints store/segment counts and writes a
manifest; no jobs submitted.

## Re-running

Re-running the same command picks up only segments whose output dir is
missing the minimal artifact set (`_SUCCESS` + `sampled_{uv,points_t,points_tk,proj_flow_2d}.npy`
+ `metadata.json`). Use `--force` to re-process everything.
