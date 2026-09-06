# 🖥️ Data

- [Access and download](#access-and-download)
- [Storage format](#storage-format)
- [Tasks and co-training regimes](#tasks-and-co-training-regimes)
- [Selecting episodes](#selecting-episodes)
- [Converting your own captures](#converting-your-own-captures)
- [3D-flow targets](#3d-flow-targets)

## Access and download

Training reads **zarr** episodes from `$EGOWAM_DATA_ROOT`. Episodes are selected either from local
directories or from the EgoVerse SQL episode table plus object storage.

### AWS Configure

Install the AWS CLI. Any of these works — the project was run with v1 from pip:

```bash
# into the project's uv environment (recommended -- keeps it with the rest of the deps)
uv pip install awscli

# or with conda / mamba
conda install -c conda-forge awscli

# or the official standalone v2 installer
# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
```

Then:

```
aws configure
AccessKeyId: AKIAYDKH4BNCAYHE5NG2
SecretAccessKey: rGjT6NSh55YiB9MC9EyNGpVy8qcaTn4i19OmkhRW
Default region name: us-east-2
Default output format:
./egowam/utils/aws/setup_secret.sh
```

`setup_secret.sh` will allow your current env to download data from cloudflare. It writes
`~/.egoverse_env`, which the training entrypoint loads via
`egowam/utils/aws/aws_data_utils.py::load_env`. Episodes are then synced on demand by
`S3EpisodeResolver` the first time a config referencing them is trained.

Bringing your own data instead? Use `LocalEpisodeResolver` and skip this entirely.

## Storage format

One zarr directory per episode. Camera streams, per-arm end-effector poses and gripper state, plus
metadata. HDF5 and VRS are capture formats only and are converted before training.

```
$EGOWAM_DATA_ROOT/
  eva_cup_saucer/<episode>/                    # robot
  aria_cup_saucer/<episode>/                   # human, in-domain
  aria_cup_saucer_everse/<episode>/            # human, in-the-wild
  *_3d_flow_track4world/<episode>/             # 3D-flow sidecars
```

The two embodiments publish different keys — the robot has wrist cameras and commanded joints, the
human has hand keypoints and eye gaze:

```
eva_<task>/<episode>/            aria_<task>/<episode>/
  images.front_1                   images.front_1
  images.left_wrist                left.obs_ee_pose
  images.right_wrist               right.obs_ee_pose
  left.obs_ee_pose                 left.obs_keypoints
  left.obs_gripper                 right.obs_keypoints
  left.obs_joints                  left.obs_aria_keypoints
  left.cmd_ee_pose                 right.obs_aria_keypoints
  left.cmd_gripper                 left.obs_wrist_pose
  left.cmd_joints                  right.obs_wrist_pose
  right.*  (same set)              obs_eye_gaze
  annotations                      obs_head_pose
                                   obs_rgb_timestamps_ns
                                   annotations
```

Keymaps and transforms that turn these into model inputs live in `egowam/rldb/embodiment/`.

The human keypoint arrays are `[T, 63]` — **21 hand keypoints** in 3D, flattened as
`(x, y, z) x 21`, per hand. `obs_keypoints` and `obs_aria_keypoints` carry the same layout in
different frames.

### Exploring the data

`egowam/scripts/tutorials/` holds notebooks for inspecting episodes before training:

| notebook | purpose |
|---|---|
| `zarr_data_viz.ipynb` | **data viewer** — render frames and plot per-episode signals |
| `dataset_tutorial.ipynb` | how a dataset is assembled from a resolver |
| `embodiment_tutorial.ipynb` | keymaps and transforms per embodiment |
| `sql_tutorial.ipynb` | querying the episode table directly |

### More data documentation

This repository documents only what EgoWAM trains on. For the wider dataset — the full hand
keypoint specification, capture rigs, and the egocentric industrial split (**EgoVerse-I**) —
see [EgoVerse](https://github.com/GaTech-RL2/EgoVerse), where the data pipeline is maintained.

## Tasks and co-training regimes

Three tasks × three regimes, named `data=<task>/<regime>`:

| Task | Robot dataset | Human dataset |
|---|---|---|
| `cup_saucer` | `eva_cup_saucer` | `aria_cup_saucer`, `aria_cup_saucer_everse` |
| `fold_clothes` | `eva_fold_clothes` | `aria_fold_clothes`, `aria_fold_clothes_everse` |
| `bag_grocery` | `eva_bag_groceries` | `aria_bag_groceries`, `aria_bag_groceries_everse` |

| Regime | Meaning |
|---|---|
| `eva` | robot only, no human data |
| `indomain` | 1:1 robot-to-human, human captured in the same scene |
| `everseA` | the ~10:1 in-the-wild EgoVerse split (flagship) |

Each regime has world-model variants: `<regime>_wm` (Pixel / DINO) and `<regime>_wm3dflow`
(3D Flow). So `data=cup_saucer/everseA_wm3dflow` is the flagship 3D-flow split.

Two 1% slices exist for smoke runs, and are not meaningful to evaluate:

| Config | Mirrors |
|---|---|
| `data=debug` | `fold_clothes/indomain` (BC) |
| `data=debug_wm3dflow` | `cup_saucer/indomain_wm3dflow` (3D Flow) |

The flow slice uses cup-on-saucer because it is the only task with both robot and human
Track4World sidecars.

## Selecting episodes

Data configs choose a resolver per dataset:

- **`LocalEpisodeResolver`** reads directories off disk and filters on the metadata each zarr
  carries. No credentials.
- **`S3EpisodeResolver`** filters the EgoVerse SQL episode table and syncs matches from object
  storage.

### A note on SQL filters

The episode table dropped its `robot_name` column and gained `rig_name`. Configs still written
against the old schema are translated by `remap_legacy_filters` in
`egowam/rldb/zarr/zarr_dataset_multi.py`:

| Legacy filter | Live columns |
|---|---|
| `robot_name: eva_bimanual` | `embodiment: eva_bimanual` |
| `robot_name: aria_bimanual` | `embodiment: human_bimanual` **and** `rig_name: aria_gen1` |

The `rig_name` clause is not cosmetic. `human_bimanual` alone also matches episodes recorded on
other rigs — on cup-on-saucer that is 5963 episodes instead of the intended 552.

## Converting your own captures

```bash
python -m egowam.scripts.eva_process.eva_to_zarr    ...   # robot HDF5 -> zarr
python -m egowam.scripts.aria_process.aria_to_zarr  ...   # Aria VRS -> zarr
```

## 3D-flow targets

Produced by `egowam/scripts/3dflow_process/` (Track4World), stored alongside the dataset:

```
<episode_root>/
  sampled_uv.npy            int32   [N, 2]        pixel anchors
  sampled_points_t.npy      float32 [T, N, 3]     world-frame 3D at t
  sampled_points_tk.npy     float32 [T, N, 3]     world-frame 3D at t+k
  sampled_proj_flow_2d.npy  float32 [T, N, 2]
  metadata.json
```

Flow is read on a fixed 28×40 (1120-point) grid. Anchors whose displacement falls below a movement
threshold are discarded — 2 mm for robot tracks, 10 mm for human tracks, which carry residual head
motion. See `egowam/rldb/zarr/point_flow_loader.py` for the authoritative spec.

Track4World runs in its own environment; the pipeline is not part of the training install.
