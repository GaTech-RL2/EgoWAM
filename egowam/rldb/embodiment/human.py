from __future__ import annotations

import os

from egowam.rldb.embodiment.embodiment import Embodiment, future_image_key_specs
from egowam.rldb.embodiment.eva import (
    _viz_batch_palm_axes,
    _viz_batch_palm_traj,
)
from egowam.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolatePointTrajectory,
    InterpolatePose,
    NormalizeFlowDelta,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    Transform,
    XYZWXYZ_to_XYZYPR,
)

ARIA_POINT_FLOW_STATS_KEY_3D = "aria_bimanual.future_flow_3d_points"
ARIA_POINT_FLOW_STATS_KEY_2D = "aria_bimanual.future_flow_2d_points"

# Point-conditioned flow: default root + canonical key names for Aria.
# Default dataset root; override with EGOWAM_DATA_ROOT (mirrors paths.data_root in the
# Hydra configs) so a new machine is configured by environment rather than by editing code.
_DATA_ROOT = os.environ.get(
    "EGOWAM_DATA_ROOT", "/coc/flash7/scratch/egowm/egoverseS3ZarrDataset"
)
ARIA_POINT_FLOW_ROOT = f"{_DATA_ROOT}/aria_cup_saucer_3d_flow_track4world"
ARIA_POINT_FLOW_2D_KEY = "future_flow_2d_points"
ARIA_POINT_FLOW_3D_KEY = "future_flow_3d_points"
FLOW_CHUNK_LENGTH = 100  # shared with EVA


class Human(Embodiment):
    VIZ_INTRINSICS_KEY = "base"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"
    ACTION_STRIDE = 3

    @classmethod
    def get_transform_list(
        cls,
        chunk_length: int = 100,
        stride: int | None = None,
        use_future: bool = False,
        include_point_flow_2d: bool = False,
        include_point_flow_3d: bool = False,
    ) -> list[Transform]:
        st = cls.ACTION_STRIDE if stride is None else stride
        return _build_aria_bimanual_transform_list(
            chunk_length=chunk_length,
            stride=st,
            use_future=use_future,
            include_point_flow_2d=include_point_flow_2d,
            include_point_flow_3d=include_point_flow_3d,
        )

    @classmethod
    def viz_transformed_batch(cls, batch, mode=""):
        image_key = cls.VIZ_IMAGE_KEY
        action_key = "actions_cartesian"
        intrinsics_key = cls.VIZ_INTRINSICS_KEY
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
        if mode == "keypoints":
            raise NotImplementedError(
                "mode='keypoints' is reserved and not implemented yet."
            )

        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: "
            f"('palm_traj', 'palm_axes', 'keypoints')."
        )

    @classmethod
    def get_keymap(
        cls,
        action_horizon: int = 30,
        use_future: bool = False,
        include_point_flow_2d: bool = False,
        include_point_flow_3d: bool = False,
        point_flow_root: str = ARIA_POINT_FLOW_ROOT,
        point_flow_movement_threshold_3d: float = 0.0,
        point_flow_movement_threshold_2d: float = 0.0,
        point_flow_head_skip_frames: int = 20,
        point_flow_tail_skip_frames: int = 20,
        future_image_offsets: list[int] | None = None,
    ):
        """``future_image_offsets`` selects sparse future-video supervision.

        Instead of the single frame at ``t + action_horizon``, emit one future image
        key per offset (``..._s1 … _sN``). Aria runs at 20 Hz, so [6,12,18,24,30] is
        0.3 s … 1.5 s ahead — the same wall-clock horizons as EVA's
        [9,18,27,36,45] at 30 Hz, under identical key names. Defaults to None =
        the historical single frame.
        """
        key_map = {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "right.action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.obs_ee_pose",
                "horizon": action_horizon,
            },
            "left.action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.obs_ee_pose",
                "horizon": action_horizon,
            },
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
            },
            "left.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_ee_pose",
            },
            "obs_head_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "obs_head_pose",
            },
        }
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
                    "_future.left.obs_ee_pose": {
                        "key_type": "proprio_keys",
                        "zarr_key": "left.obs_ee_pose",
                        "future_offset": action_horizon,
                    },
                    "_future.right.obs_ee_pose": {
                        "key_type": "proprio_keys",
                        "zarr_key": "right.obs_ee_pose",
                        "future_offset": action_horizon,
                    },
                }
            )
        if include_point_flow_3d:
            key_map[ARIA_POINT_FLOW_3D_KEY] = {
                "key_type": "flow_keys",
                "zarr_key": ARIA_POINT_FLOW_3D_KEY,
                "source": "point_flow",
                "point_flow_root": point_flow_root,
                "point_dim": 3,
                "movement_threshold": float(point_flow_movement_threshold_3d),
                "head_skip_frames": int(point_flow_head_skip_frames),
                "tail_skip_frames": int(point_flow_tail_skip_frames),
            }
        if include_point_flow_2d:
            key_map[ARIA_POINT_FLOW_2D_KEY] = {
                "key_type": "flow_keys",
                "zarr_key": ARIA_POINT_FLOW_2D_KEY,
                "source": "point_flow",
                "point_flow_root": point_flow_root,
                "point_dim": 2,
                "movement_threshold": float(point_flow_movement_threshold_2d),
                "head_skip_frames": int(point_flow_head_skip_frames),
                "tail_skip_frames": int(point_flow_tail_skip_frames),
            }
        return key_map


class Aria(Human):
    VIZ_INTRINSICS_KEY = "base"
    ACTION_STRIDE = 3


class Scale(Human):
    VIZ_INTRINSICS_KEY = "scale"
    ACTION_STRIDE = 1


class Mecka(Human):
    VIZ_INTRINSICS_KEY = "mecka"
    ACTION_STRIDE = 1


def _build_aria_bimanual_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_action_world: str = "left.action_ee_pose",
    right_action_world: str = "right.action_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_action_headframe: str = "left.action_ee_pose_headframe",
    right_action_headframe: str = "right.action_ee_pose_headframe",
    left_obs_headframe: str = "left.obs_ee_pose_headframe",
    right_obs_headframe: str = "right.obs_ee_pose_headframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    future_obs_key: str = "future.observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 3,
    delete_target_world: bool = True,
    use_future: bool = False,
    include_point_flow_2d: bool = False,
    include_point_flow_3d: bool = False,
) -> list[Transform]:
    """Canonical ARIA bimanual transform pipeline used by tests and notebooks.

    Aria human data does not have commanded ee poses; action chunks are built
    from stacked observed ee poses (typically with a horizon on
    ``left/right.action_ee_pose`` mapped from ``left/right.obs_ee_pose``).

    When ``use_future`` is True, future EE poses (world frame, same layout as
    EVA future proprio) are converted to YPR and concatenated into
    ``future_obs_key``; pair with ``Human.get_keymap(..., use_future=True)``.
    """
    keys_to_delete = list(
        {
            left_action_world,
            right_action_world,
            left_obs_pose,
            right_obs_pose,
        }
    )
    target_pose_key = target_world
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)

    transform_list: list[Transform] = [
        ActionChunkCoordinateFrameTransform(
            target_world=target_pose_key,
            chunk_world=left_action_world,
            transformed_key_name=left_action_headframe,
            is_quat=target_world_is_quat,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_pose_key,
            chunk_world=right_action_world,
            transformed_key_name=right_action_headframe,
            is_quat=target_world_is_quat,
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_headframe,
            is_quat=target_world_is_quat,
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_headframe,
            is_quat=target_world_is_quat,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_action_headframe,
            output_action_key=left_action_headframe,
            stride=stride,
            is_quat=target_world_is_quat,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_action_headframe,
            output_action_key=right_action_headframe,
            stride=stride,
            is_quat=target_world_is_quat,
        ),
    ]

    if target_world_is_quat:
        transform_list.append(
            XYZWXYZ_to_XYZYPR(
                keys=[
                    left_action_headframe,
                    right_action_headframe,
                    left_obs_headframe,
                    right_obs_headframe,
                ]
            )
        )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[left_action_headframe, right_action_headframe],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[left_obs_headframe, right_obs_headframe],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=keys_to_delete),
        ]
    )

    if use_future:
        future_left = "_future.left.obs_ee_pose"
        future_right = "_future.right.obs_ee_pose"
        if target_world_is_quat:
            transform_list.append(XYZWXYZ_to_XYZYPR(keys=[future_left, future_right]))
        transform_list.append(
            ConcatKeys(
                key_list=[future_left, future_right],
                new_key_name=future_obs_key,
                delete_old_keys=True,
            )
        )
        transform_list.append(NumpyToTensor(keys=[future_obs_key]))

    # Point-flow trajectories: resample Aria's native k=45 to FLOW_CHUNK_LENGTH=100
    # with stride=1 (NOT the Aria ACTION_STRIDE=3 — that would drop the P_tk
    # endpoint at raw index 44). NormalizeFlowDelta converts the resampled
    # absolute trajectory into a normalized delta using the flow_stats registry.
    if include_point_flow_3d:
        transform_list.append(
            InterpolatePointTrajectory(
                new_chunk_length=FLOW_CHUNK_LENGTH,
                action_key=ARIA_POINT_FLOW_3D_KEY,
                output_action_key=ARIA_POINT_FLOW_3D_KEY,
                stride=1,
            )
        )
        transform_list.append(
            NormalizeFlowDelta(
                action_key=ARIA_POINT_FLOW_3D_KEY,
                output_action_key=ARIA_POINT_FLOW_3D_KEY,
                stats_key=ARIA_POINT_FLOW_STATS_KEY_3D,
            )
        )
    if include_point_flow_2d:
        transform_list.append(
            InterpolatePointTrajectory(
                new_chunk_length=FLOW_CHUNK_LENGTH,
                action_key=ARIA_POINT_FLOW_2D_KEY,
                output_action_key=ARIA_POINT_FLOW_2D_KEY,
                stride=1,
            )
        )
        transform_list.append(
            NormalizeFlowDelta(
                action_key=ARIA_POINT_FLOW_2D_KEY,
                output_action_key=ARIA_POINT_FLOW_2D_KEY,
                stats_key=ARIA_POINT_FLOW_STATS_KEY_2D,
            )
        )

    return transform_list
