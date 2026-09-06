from __future__ import annotations

import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from egowam.rldb.embodiment.embodiment import Embodiment, future_image_key_specs
from egowam.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePointTrajectory,
    InterpolatePose,
    NormalizeFlowDelta,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    Transform,
    XYZWXYZ_to_XYZYPR,
)
from egowam.utils.egowamUtils import (
    EXTRINSICS,
    INTRINSICS,
    cam_frame_to_cam_pixels,
    draw_actions,
)
from egowam.utils.pose_utils import (
    _matrix_to_xyzwxyz,
)

# Stats-registry keys used by NormalizeFlowDelta. These must match the keys the
# preflight populates in ensure_flow_stats. Prefixing with the domain tag avoids
# collisions in the cotrain case where EVA and Aria use the same logical
# flow_key but different per-embodiment mean/std distributions.
EVA_POINT_FLOW_STATS_KEY_3D = "eva_bimanual.future_flow_3d_points"
EVA_POINT_FLOW_STATS_KEY_2D = "eva_bimanual.future_flow_2d_points"

# Point-conditioned flow: default root + canonical key names. Short names used
# as both schematic keys and batch keys.
# Default dataset root; override with EGOWAM_DATA_ROOT (mirrors paths.data_root in the
# Hydra configs) so a new machine is configured by environment rather than by editing code.
_DATA_ROOT = os.environ.get(
    "EGOWAM_DATA_ROOT", "/coc/flash7/scratch/egowm/egoverseS3ZarrDataset"
)
EVA_POINT_FLOW_ROOT = f"{_DATA_ROOT}/eva_cup_saucer_1_3d_flow_track4world"
EVA_POINT_FLOW_2D_KEY = "future_flow_2d_points"
EVA_POINT_FLOW_3D_KEY = "future_flow_3d_points"
FLOW_CHUNK_LENGTH = 100  # shared resample length; matches action chunk_length


class Eva(Embodiment):
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @staticmethod
    def get_transform_list(
        include_point_flow_2d: bool = False,
        include_point_flow_3d: bool = False,
    ) -> list[Transform]:
        return _build_eva_bimanual_transform_list(
            include_point_flow_2d=include_point_flow_2d,
            include_point_flow_3d=include_point_flow_3d,
        )

    @classmethod
    def viz_transformed_batch(cls, batch, mode=""):
        """
        Visualize one transformed EVA batch sample.

        Modes:
            - palm_traj: draw left/right palm trajectories from actions_cartesian.
            - palm_axes: draw local xyz axes at each palm anchor using ypr.
        """
        image_key = cls.VIZ_IMAGE_KEY
        action_key = "actions_cartesian"
        intrinsics_key = "base"
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
            f"Unsupported mode '{mode}'. Expected one of: "
            f"('palm_traj', 'palm_axes', 'keypoints')."
        )

    @classmethod
    def get_keymap(
        cls,
        action_horizon: int = 45,
        use_future: bool = False,
        include_point_flow_2d: bool = False,
        include_point_flow_3d: bool = False,
        point_flow_root: str = EVA_POINT_FLOW_ROOT,
        point_flow_movement_threshold_3d: float = 0.0,
        point_flow_movement_threshold_2d: float = 0.0,
        point_flow_head_skip_frames: int = 0,
        point_flow_tail_skip_frames: int = 0,
        use_wrist_images: bool = True,
        future_image_offsets: list[int] | None = None,
    ):
        """``use_wrist_images=False`` yields a FRONT-CAMERA-ONLY keymap.

        Omitting the keys (rather than letting the model ignore them) also skips their
        JPEG decode, which is ~3x of the per-sample image cost. Aria capture is
        mono-camera, so front-only additionally makes the EVA and Aria branches
        structurally identical.

        ``future_image_offsets`` selects sparse future-video supervision: instead of
        the single frame at ``t + action_horizon``, emit one future image key per
        offset (``..._s1 … _sN``). EVA runs at 30 Hz, so [9,18,27,36,45] is
        0.3 s … 1.5 s ahead. Defaults to None = the historical single frame.
        """
        key_map = {
            # Image keys
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
        }
        if use_wrist_images:
            key_map.update(
                {
                    "observations.images.right_wrist_img": {
                        "key_type": "camera_keys",
                        "zarr_key": "images.right_wrist",
                    },
                    "observations.images.left_wrist_img": {
                        "key_type": "camera_keys",
                        "zarr_key": "images.left_wrist",
                    },
                }
            )
        key_map |= {
            # Proprioception keys
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
            # Action keys
            "right.gripper": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_gripper",  # zarr stores one gripper array per arm
                "horizon": action_horizon,
            },
            "left.gripper": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_gripper",
                "horizon": action_horizon,
            },
            "right.cmd_ee_pose": {
                "key_type": "action_keys",
                # "zarr_key": "right.obs_ee_pose",  # right.cmd_ee_pose
                "zarr_key": "right.cmd_ee_pose",
                "horizon": action_horizon,
            },
            "left.cmd_ee_pose": {
                "key_type": "action_keys",
                # "zarr_key": "left.obs_ee_pose",  # left.cmd_ee_pose
                "zarr_key": "left.cmd_ee_pose",
                "horizon": action_horizon,
            },
        }
        # Extract future states
        if use_future and use_wrist_images:
            # Future wrist targets follow the same flag: with front-only inputs the
            # WM head has no wrist view to predict FROM, so predicting one would be
            # an unpaired target.
            key_map.update(
                future_image_key_specs(
                    "future.observations.images.right_wrist_img",
                    "images.right_wrist",
                    action_horizon,
                    future_image_offsets,
                )
            )
            key_map.update(
                future_image_key_specs(
                    "future.observations.images.left_wrist_img",
                    "images.left_wrist",
                    action_horizon,
                    future_image_offsets,
                )
            )
        if use_future:
            key_map.update(
                future_image_key_specs(
                    "future.observations.images.front_img_1",
                    "images.front_1",
                    action_horizon,
                    future_image_offsets,
                )
            )
            key_map.update(
                {
                    # Internal keys; merged into future.observations.state.ee_pose by transforms
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
        if include_point_flow_2d:
            key_map[EVA_POINT_FLOW_2D_KEY] = {
                "key_type": "flow_keys",
                "zarr_key": EVA_POINT_FLOW_2D_KEY,
                "source": "point_flow",
                "point_flow_root": point_flow_root,
                "point_dim": 2,
                "movement_threshold": float(point_flow_movement_threshold_2d),
                "head_skip_frames": int(point_flow_head_skip_frames),
                "tail_skip_frames": int(point_flow_tail_skip_frames),
            }
        return key_map


def _build_eva_bimanual_transform_list(
    *,
    left_target_world: str = "left_extrinsics_pose",
    right_target_world: str = "right_extrinsics_pose",
    left_cmd_world: str = "left.cmd_ee_pose",
    right_cmd_world: str = "right.cmd_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_gripper: str = "left.gripper",
    right_gripper: str = "right.gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    future_obs_key: str = "future.observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    extrinsics_key: str = "x5Dec13_2",
    is_quat: bool = True,
    use_future: bool = False,
    include_point_flow_2d: bool = False,
    include_point_flow_3d: bool = False,
    # Stats-registry key that NormalizeFlowDelta reads for mean/std. Must match
    # the preflight's `{dataset_name}.future_flow_*_points` key (see
    # point_flow_preflight.run_flow_stats_preflight). Defaults to the EVA domain;
    # other embodiments (e.g. RoboTwin) pass their own dataset-qualified key.
    point_flow_stats_key_3d: str = EVA_POINT_FLOW_STATS_KEY_3D,
    point_flow_stats_key_2d: str = EVA_POINT_FLOW_STATS_KEY_2D,
    **kwargs,
) -> list[Transform]:
    """Canonical EVA bimanual transform pipeline used by tests and notebooks."""
    extrinsics = EXTRINSICS[extrinsics_key]
    left_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["right"][None, :])[0]
    left_extra_batch_key = {"left_extrinsics_pose": left_extrinsics_pose}
    right_extra_batch_key = {"right_extrinsics_pose": right_extrinsics_pose}
    transform_list = [
        ActionChunkCoordinateFrameTransform(
            target_world=left_target_world,
            chunk_world=left_cmd_world,
            transformed_key_name=left_cmd_camframe,
            extra_batch_key=left_extra_batch_key,
            is_quat=is_quat,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_target_world,
            chunk_world=right_cmd_world,
            transformed_key_name=right_cmd_camframe,
            extra_batch_key=right_extra_batch_key,
            is_quat=is_quat,
        ),
        PoseCoordinateFrameTransform(
            target_world=left_target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_pose,
            is_quat=is_quat,
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_pose,
            is_quat=is_quat,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_cmd_camframe,
            output_action_key=left_cmd_camframe,
            stride=stride,
            is_quat=is_quat,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_camframe,
            output_action_key=right_cmd_camframe,
            stride=stride,
            is_quat=is_quat,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=left_gripper,
            output_action_key=left_gripper,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_gripper,
            output_action_key=right_gripper,
            stride=stride,
        ),
    ]

    if is_quat:
        transform_list.append(
            XYZWXYZ_to_XYZYPR(
                keys=[
                    left_cmd_camframe,
                    right_cmd_camframe,
                    left_obs_pose,
                    right_obs_pose,
                ]
            )
        )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_cmd_camframe,
                    left_gripper,
                    right_cmd_camframe,
                    right_gripper,
                ],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[
                    left_obs_pose,
                    left_obs_gripper,
                    right_obs_pose,
                    right_obs_gripper,
                ],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(
                keys_to_delete=[
                    left_cmd_world,
                    right_cmd_world,
                    left_target_world,
                    right_target_world,
                ]
            ),
            NumpyToTensor(
                keys=[
                    actions_key,
                    obs_key,
                ]
            ),
        ]
    )

    if use_future:
        future_left_pose = "_future.left.obs_ee_pose"
        future_left_gripper = "_future.left.obs_gripper"
        future_right_pose = "_future.right.obs_ee_pose"
        future_right_gripper = "_future.right.obs_gripper"
        if is_quat:
            transform_list.append(
                XYZWXYZ_to_XYZYPR(keys=[future_left_pose, future_right_pose])
            )
        transform_list.append(
            ConcatKeys(
                key_list=[
                    future_left_pose,
                    future_left_gripper,
                    future_right_pose,
                    future_right_gripper,
                ],
                new_key_name=future_obs_key,
                delete_old_keys=True,
            )
        )
        transform_list.append(NumpyToTensor(keys=[future_obs_key]))

    # Point-flow trajectories: (a) resample native-k (EVA k=30) to
    # FLOW_CHUNK_LENGTH=100 with stride=1 (inheriting ACTION_STRIDE would drop
    # P_tk on Aria); (b) NormalizeFlowDelta converts absolute trajectory ->
    # delta (relative to step 0) and applies mean/std from the flow_stats
    # registry. This is the single normalization application point.
    if include_point_flow_3d:
        transform_list.append(
            InterpolatePointTrajectory(
                new_chunk_length=FLOW_CHUNK_LENGTH,
                action_key=EVA_POINT_FLOW_3D_KEY,
                output_action_key=EVA_POINT_FLOW_3D_KEY,
                stride=1,
            )
        )
        transform_list.append(
            NormalizeFlowDelta(
                action_key=EVA_POINT_FLOW_3D_KEY,
                output_action_key=EVA_POINT_FLOW_3D_KEY,
                stats_key=point_flow_stats_key_3d,
            )
        )
    if include_point_flow_2d:
        transform_list.append(
            InterpolatePointTrajectory(
                new_chunk_length=FLOW_CHUNK_LENGTH,
                action_key=EVA_POINT_FLOW_2D_KEY,
                output_action_key=EVA_POINT_FLOW_2D_KEY,
                stride=1,
            )
        )
        transform_list.append(
            NormalizeFlowDelta(
                action_key=EVA_POINT_FLOW_2D_KEY,
                output_action_key=EVA_POINT_FLOW_2D_KEY,
                stats_key=point_flow_stats_key_2d,
            )
        )

    return transform_list


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        return arr.numpy()
    return np.asarray(arr)


def _split_action_pose(actions):
    # 14D layout: [L xyz ypr g, R xyz ypr g]
    # 12D layout: [L xyz ypr, R xyz ypr]
    if actions.shape[-1] == 14:
        left_xyz = actions[:, :3]
        left_ypr = actions[:, 3:6]
        right_xyz = actions[:, 7:10]
        right_ypr = actions[:, 10:13]
    elif actions.shape[-1] == 12:
        left_xyz = actions[:, :3]
        left_ypr = actions[:, 3:6]
        right_xyz = actions[:, 6:9]
        right_ypr = actions[:, 9:12]
    else:
        raise ValueError(f"Unsupported action dim {actions.shape[-1]}")
    return left_xyz, left_ypr, right_xyz, right_ypr


def _prepare_viz_image(batch, image_key):
    img = _to_numpy(batch[image_key][0])
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    return img


def _viz_batch_palm_traj(batch, image_key, action_key, intrinsics_key):
    img_np = _prepare_viz_image(batch, image_key)
    intrinsics = INTRINSICS[intrinsics_key]
    actions = _to_numpy(batch[action_key][0])
    left_xyz, _, right_xyz, _ = _split_action_pose(actions)

    vis = draw_actions(
        img_np.copy(),
        type="xyz",
        color="Blues",
        actions=left_xyz,
        extrinsics=None,
        intrinsics=intrinsics,
        arm="left",
    )
    vis = draw_actions(
        vis,
        type="xyz",
        color="Reds",
        actions=right_xyz,
        extrinsics=None,
        intrinsics=intrinsics,
        arm="right",
    )
    return vis


def _viz_batch_palm_axes(batch, image_key, action_key, intrinsics_key, axis_len_m=0.04):
    img_np = _prepare_viz_image(batch, image_key)
    intrinsics = INTRINSICS[intrinsics_key]
    actions = _to_numpy(batch[action_key][0])
    left_xyz, left_ypr, right_xyz, right_ypr = _split_action_pose(actions)
    vis = img_np.copy()

    def _draw_axis_color_legend(frame):
        _, w = frame.shape[:2]
        x_right = w - 12
        y_start = 14
        y_step = 12
        line_len = 24
        axis_legend = [
            ("x", (255, 0, 0)),
            ("y", (0, 255, 0)),
            ("z", (0, 0, 255)),
        ]
        for i, (name, color) in enumerate(axis_legend):
            y = y_start + i * y_step
            x0 = x_right - line_len
            x1 = x_right
            cv2.line(frame, (x0, y), (x1, y), color, 3)
            cv2.putText(
                frame,
                name,
                (x0 - 12, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame

    def _draw_rotation_at_palm(frame, xyz_seq, ypr_seq, label, anchor_color):
        if len(xyz_seq) == 0 or len(ypr_seq) == 0:
            return frame

        palm_xyz = xyz_seq[0]
        palm_ypr = ypr_seq[0]
        rot = R.from_euler("ZYX", palm_ypr, degrees=False).as_matrix()

        axis_points_cam = np.vstack(
            [
                palm_xyz,
                palm_xyz + rot[:, 0] * axis_len_m,
                palm_xyz + rot[:, 1] * axis_len_m,
                palm_xyz + rot[:, 2] * axis_len_m,
            ]
        )

        px = cam_frame_to_cam_pixels(axis_points_cam, intrinsics)[:, :2]
        if not np.isfinite(px).all():
            return frame
        pts = np.round(px).astype(np.int32)

        h, w = frame.shape[:2]
        x0, y0 = pts[0]
        if not (0 <= x0 < w and 0 <= y0 < h):
            return frame

        cv2.circle(frame, (x0, y0), 4, anchor_color, -1)
        axis_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, color in enumerate(axis_colors, start=1):
            x1, y1 = pts[i]
            if 0 <= x1 < w and 0 <= y1 < h:
                cv2.line(frame, (x0, y0), (x1, y1), color, 2)
                cv2.circle(frame, (x1, y1), 2, color, -1)

        cv2.putText(
            frame,
            label,
            (x0 + 6, max(12, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            anchor_color,
            1,
            cv2.LINE_AA,
        )
        return frame

    vis = _draw_rotation_at_palm(vis, left_xyz, left_ypr, "L rot", (255, 180, 80))
    vis = _draw_rotation_at_palm(vis, right_xyz, right_ypr, "R rot", (80, 180, 255))
    vis = _draw_axis_color_legend(vis)
    return vis
