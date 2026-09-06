# 🦾 Real-Robot Deployment

The policy was deployed on the lab's bimanual rig. This document describes that setup so the
results are interpretable and the code is adaptable — it is not a turnkey guide, because the code
under `egowam/robot/` is specific to this hardware.

## Rig

| Component | Detail |
|---|---|
| Arms | 2 × ARX5, 6-DoF, upright-mounted, parallel-jaw grippers |
| Ego camera | Project Aria glasses, head-mounted — the stream shared with human demonstrations |
| Wrist cameras | 2 × Intel RealSense D405 |
| Teleoperation | Meta Quest 3 |
| Control rate | 30 Hz |
| Inference | single RTX 4090 |

## Running a policy

```bash
python egowam/robot/rollout.py --action-only-inference --cartesian \
  --policy-path <ckpt> --arms both --frequency 30 --query_frequency 30 \
  --resampled-action-len 45
```

`--action-only-inference` drops the world-model head, which is the deployment configuration for
every variant in the paper: the head shapes the trunk during training and is discarded at test
time, so all variants have identical inference cost.

## What you must change for a different rig

These are hardcoded to our hardware and have no sensible default:

| Location | What |
|---|---|
| `egowam/robot/eva/eva_ws/src/config/configs.yaml` | URDF path, CAN buses, RealSense serial numbers |
| `egowam/utils/egowamUtils.py` | `EXTRINSICS` — camera→base matrices per calibration session |
| `egowam/utils/egowamUtils.py` | `INTRINSICS` — per-camera intrinsics |
| `egowam/robot/rollout.py` | the extrinsics key selecting which calibration to use |

Camera calibration is produced by `egowam/scripts/calibrate_camera/`.

## Frame conventions

Actions are per-arm 6-DoF end-effector poses plus a gripper command, expressed in the **ego-camera
frame** rather than the robot base frame. Human demonstrations are re-expressed in the
*instantaneous* device frame at each timestep, which factors out head motion so the same physical
motion yields comparable numbers across embodiments. Both sides are quantile-normalized, mapping
each dimension's 1st and 99th percentiles to [-1, 1].

The `PoseCoordinateFrameTransform` entries in the EVA transform pipeline perform the base→camera
conversion and are load-bearing: proprioception normalization statistics must be computed *after*
them, or the policy receives poses in the wrong frame.

## Third-party components

`egowam/robot/eva/stanford_repo/` (ARX5 SDK) and `egowam/robot/oculus_reader/` are vendored
third-party code under their own licenses.
