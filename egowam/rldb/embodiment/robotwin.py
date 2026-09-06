"""RoboTwin 2.0 bimanual embodiments.

Data source: RoboTwin 2.0 `data/episodeN.hdf5` converted to EgoVerse zarr by
`egowam/scripts/robotwin_process/robotwin_to_zarr.py`. The zarr key layout is
identical to EVA (`{left,right}.obs_ee_pose` [T,7 xyz+wxyz], `*_gripper` [T,1],
`images.{front_1,left_wrist,right_wrist}`), so the canonical transform pipeline
`_build_eva_bimanual_transform_list` is reused verbatim with the RoboTwin head-camera
extrinsics/intrinsics ("robotwin", registered in egowamUtils). All RoboTwin robots
share one static head camera → one extrinsics key for every robot and both arms.

Poses are stored in the SAPIEN **world** frame (xyz + wxyz quaternion), which is the
same frame the head-camera extrinsic references (verified: projecting endpose with
extrinsic_cv+intrinsic_cv, and wrist-cam↔endpose offset constant at ~0.1 m). The
transform expresses them relative to the head camera at load time, exactly like EVA.
"""

from __future__ import annotations

import os

from egowam.rldb.embodiment.embodiment import Embodiment
from egowam.rldb.embodiment.eva import (
    EVA_POINT_FLOW_3D_KEY,
    _build_eva_bimanual_transform_list,
    _viz_batch_palm_axes,
    _viz_batch_palm_traj,
)
from egowam.rldb.zarr.action_chunk_transforms import Transform

ROBOTWIN_EXTRINSICS_KEY = "robotwin"
ROBOTWIN_INTRINSICS_KEY = "robotwin"

# Track4World 3D point-flow sidecars: parallel dir tree to zarrDataset, keyed by
# the identical <task>/<robot>_<variant>/episodeN.zarr path. point_flow_root is
# the task/variant leaf (the loader globs "*/metadata.json" one level down).
# Sibling tree to the RoboTwin dataset root, not a child of it -- override with
# EGOWAM_ROBOTWIN_FLOW_ROOT so a new machine is configured by environment, not by
# editing code (mirrors _DATA_ROOT in eva.py / human.py).
ROBOTWIN_POINT_FLOW_ROOT = os.environ.get(
    "EGOWAM_ROBOTWIN_FLOW_ROOT",
    "/coc/flash7/scratch/egowm/robotwin_3d_flow_track4world",
)
# Default flow-stats registry key. MUST equal the preflight's
# `{train_datasets-key}.future_flow_3d_points`; override per-domain via config.
ROBOTWIN_POINT_FLOW_STATS_KEY_3D = "robotwin_aloha_bimanual.future_flow_3d_points"


class RobotwinBimanual(Embodiment):
    """Base RoboTwin bimanual embodiment (EE-Cartesian, 14-dim shared action space)."""

    VIZ_IMAGE_KEY = "observations.images.front_img_1"
    EXTRINSICS_KEY = ROBOTWIN_EXTRINSICS_KEY
    INTRINSICS_KEY = ROBOTWIN_INTRINSICS_KEY

    @classmethod
    def get_transform_list(
        cls,
        chunk_length: int = 100,
        stride: int = 1,
        use_future: bool = False,
        include_point_flow_3d: bool = False,
        point_flow_stats_key_3d: str = ROBOTWIN_POINT_FLOW_STATS_KEY_3D,
        **kwargs,
    ) -> list[Transform]:
        # Reuse the canonical EVA bimanual pipeline; only the camera the poses are
        # expressed relative to differs (RoboTwin head cam). include_point_flow_3d
        # adds the Track4World 3D point-flow resample + delta-normalize transforms
        # (NormalizeFlowDelta reads mean/std under point_flow_stats_key_3d, which
        # must match the preflight's dataset-qualified key).
        return _build_eva_bimanual_transform_list(
            extrinsics_key=cls.EXTRINSICS_KEY,
            chunk_length=chunk_length,
            stride=stride,
            is_quat=True,
            use_future=use_future,
            include_point_flow_2d=False,
            include_point_flow_3d=include_point_flow_3d,
            point_flow_stats_key_3d=point_flow_stats_key_3d,
        )

    @classmethod
    def get_keymap(
        cls,
        action_horizon: int = 45,
        use_future: bool = False,
        include_point_flow_3d: bool = False,
        point_flow_root: str = ROBOTWIN_POINT_FLOW_ROOT,
        point_flow_movement_threshold_3d: float = 0.0,
        point_flow_head_skip_frames: int = 0,
        point_flow_tail_skip_frames: int = 0,
        **kwargs,
    ):
        key_map = {
            # Image keys (RoboTwin head cam -> front_1; wrist cams)
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "observations.images.right_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.right_wrist",
            },
            "observations.images.left_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.left_wrist",
            },
            # Proprioception (world-frame EE pose + gripper)
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
            },
            "right.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_gripper",
            },
            "left.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_ee_pose",
            },
            "left.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_gripper",
            },
            # Action keys (future EE-pose chunk = executed trajectory; gripper)
            "right.gripper": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_gripper",
                "horizon": action_horizon,
            },
            "left.gripper": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_gripper",
                "horizon": action_horizon,
            },
            "right.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_ee_pose",
                "horizon": action_horizon,
            },
            "left.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_ee_pose",
                "horizon": action_horizon,
            },
        }
        if use_future:
            key_map.update(
                {
                    "future.observations.images.front_img_1": {
                        "key_type": "camera_keys",
                        "zarr_key": "images.front_1",
                        "future_offset": action_horizon,
                    },
                    "future.observations.images.right_wrist_img": {
                        "key_type": "camera_keys",
                        "zarr_key": "images.right_wrist",
                        "future_offset": action_horizon,
                    },
                    "future.observations.images.left_wrist_img": {
                        "key_type": "camera_keys",
                        "zarr_key": "images.left_wrist",
                        "future_offset": action_horizon,
                    },
                    "_future.left.obs_ee_pose": {
                        "key_type": "proprio_keys",
                        "zarr_key": "left.obs_ee_pose",
                        "future_offset": action_horizon,
                    },
                    "_future.left.obs_gripper": {
                        "key_type": "proprio_keys",
                        "zarr_key": "left.obs_gripper",
                        "future_offset": action_horizon,
                    },
                    "_future.right.obs_ee_pose": {
                        "key_type": "proprio_keys",
                        "zarr_key": "right.obs_ee_pose",
                        "future_offset": action_horizon,
                    },
                    "_future.right.obs_gripper": {
                        "key_type": "proprio_keys",
                        "zarr_key": "right.obs_gripper",
                        "future_offset": action_horizon,
                    },
                }
            )
        if include_point_flow_3d:
            # Track4World 3D point-flow sidecar (source="point_flow" routes the
            # loader to _load_point_flow_target; resolves the flow dir next to
            # each episode's source zarr via metadata.json's zarr_path back-ref).
            key_map[EVA_POINT_FLOW_3D_KEY] = {
                "key_type": "flow_keys",
                "zarr_key": EVA_POINT_FLOW_3D_KEY,
                "source": "point_flow",
                "point_flow_root": point_flow_root,
                "point_dim": 3,
                "movement_threshold": float(point_flow_movement_threshold_3d),
                "head_skip_frames": int(point_flow_head_skip_frames),
                "tail_skip_frames": int(point_flow_tail_skip_frames),
            }
        return key_map

    @classmethod
    def viz_transformed_batch(cls, batch, mode=""):
        """Overlay left/right palm trajectory (or local axes) from actions_cartesian.

        Uses the RoboTwin head-camera intrinsics; actions are already in camera frame
        after the transform pipeline, so extrinsics=None inside the draw helpers.
        """
        image_key = cls.VIZ_IMAGE_KEY
        action_key = "actions_cartesian"
        intrinsics_key = cls.INTRINSICS_KEY
        mode = (mode or "palm_traj").lower()
        if mode == "palm_traj":
            return _viz_batch_palm_traj(
                batch=batch,
                image_key=image_key,
                action_key=action_key,
                intrinsics_key=intrinsics_key,
            )
        if mode == "palm_axes":
            return _viz_batch_palm_axes(
                batch=batch,
                image_key=image_key,
                action_key=action_key,
                intrinsics_key=intrinsics_key,
            )
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected 'palm_traj' or 'palm_axes'."
        )


class RobotwinAloha(RobotwinBimanual):
    """aloha-agilex (6-DoF per arm)."""


class RobotwinArx(RobotwinBimanual):
    """arx-x5 (6-DoF per arm) — the cross-embodiment rollout robot."""


class RobotwinPiper(RobotwinBimanual):
    """piper (6-DoF per arm)."""


class RobotwinUr5(RobotwinBimanual):
    """ur5 (7-DoF per arm; EE-Cartesian action keeps it 14-dim)."""


class RobotwinFranka(RobotwinBimanual):
    """franka (7-DoF per arm; EE-Cartesian action keeps it 14-dim)."""
