"""
EE-Cartesian variant of process_data.py for RoboTwin Diffusion Policy.

Instead of using the 14-dim JOINT vector (/joint_action/vector) as both `state`
and `action` (the native DP behavior), this builds a 14-dim EE-Cartesian vector
for every frame and uses that EE vector as BOTH `state` and `action`.

Per-frame EE vector (14-dim):
    [ L_xyz(3), L_ypr(3), L_grip(1), R_xyz(3), R_ypr(3), R_grip(1) ]
  - xyz  : /endpose/{side}_endpose[:3]   (world frame position)
  - ypr  : scipy Rotation.from_quat([qx,qy,qz,qw]).as_euler("ZYX")
           where /endpose/{side}_endpose[3:7] is a wxyz quaternion (reordered to
           xyzw for scipy). as_euler("ZYX") -> [yaw, pitch, roll].
  - grip : /joint_action/{side}_gripper  (the COMMANDED gripper, same source the
           native DP process_data.py uses via /joint_action/vector).

The zarr layout is identical to process_data.py:
    data/head_camera  (N, 3, H, W) uint8   (NHWC->NCHW)
    data/state        (N, 14)      float32
    data/action       (N, 14)      float32
    meta/episode_ends (n_episodes,) int64

state uses EE[0 .. T-2], action uses EE[1 .. T-1] (action = state shifted by 1),
exactly matching the frame-index bookkeeping of the native process_data.py.
"""

import os
import numpy as np
import zarr
import shutil
import argparse
import cv2
import h5py
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_ypr(quat_wxyz):
    """endpose[3:7] is wxyz; reorder to xyzw for scipy, return ZYX euler = [yaw,pitch,roll]."""
    qw, qx, qy, qz = quat_wxyz
    return R.from_quat([qx, qy, qz, qw]).as_euler("ZYX")


def load_hdf5_ee(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_endpose = root["/endpose/left_endpose"][()]      # (T, 7) xyz + wxyz
        right_endpose = root["/endpose/right_endpose"][()]     # (T, 7)
        left_gripper = root["/joint_action/left_gripper"][()]  # (T,) commanded
        right_gripper = root["/joint_action/right_gripper"][()]
        image_dict = dict()
        for cam_name in root["/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    T = left_endpose.shape[0]
    ee_vector = np.zeros((T, 14), dtype=np.float64)
    for t in range(T):
        l_xyz = left_endpose[t, :3]
        l_ypr = quat_wxyz_to_ypr(left_endpose[t, 3:7])
        r_xyz = right_endpose[t, :3]
        r_ypr = quat_wxyz_to_ypr(right_endpose[t, 3:7])
        ee_vector[t, 0:3] = l_xyz
        ee_vector[t, 3:6] = l_ypr
        ee_vector[t, 6] = left_gripper[t]
        ee_vector[t, 7:10] = r_xyz
        ee_vector[t, 10:13] = r_ypr
        ee_vector[t, 13] = right_gripper[t]

    return left_gripper, ee_vector, image_dict


def main():
    parser = argparse.ArgumentParser(description="Process episodes into an EE-Cartesian zarr for DP.")
    parser.add_argument("task_name", type=str, help="Task name (e.g., stack_bowls_three)")
    parser.add_argument("task_config", type=str, help="Config tag used in zarr name (e.g., demo_clean)")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process (e.g., 50)")
    parser.add_argument(
        "--load_dir",
        type=str,
        default=None,
        help="Directory containing data/episode*.hdf5. "
        "Default: /coc/flash7/scratch/egowm/robotwin/dataset/<task>/aloha-agilex_clean_50",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output zarr path. "
        "Default: /coc/flash7/scratch/egowm/robotwin/_baselines_ee/DP/<task>-<config>-<num>.zarr",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = args.load_dir or (
        f"/coc/flash7/scratch/egowm/robotwin/dataset/{task_name}/aloha-agilex_clean_50"
    )
    save_dir = args.save_dir or (
        f"/coc/flash7/scratch/egowm/robotwin/_baselines_ee/DP/{task_name}-{task_config}-{num}.zarr"
    )

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    total_count = 0
    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    head_camera_arrays = []
    episode_ends_arrays, action_arrays, state_arrays = [], [], []

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        left_gripper_all, ee_vector_all, image_dict_all = load_hdf5_ee(load_path)

        T = left_gripper_all.shape[0]
        for j in range(0, T):
            head_img_bit = image_dict_all["head_camera"][j]
            ee_state = ee_vector_all[j]

            if j != T - 1:
                head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                head_camera_arrays.append(head_img)
                state_arrays.append(ee_state)
            if j != 0:
                action_arrays.append(ee_state)

        current_ep += 1
        total_count += T - 1
        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    state_arrays = np.array(state_arrays)
    action_arrays = np.array(action_arrays)
    head_camera_arrays = np.array(head_camera_arrays)

    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    state_chunk_size = (100, state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=head_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    print(f"Saved EE zarr to {save_dir}")
    print(f"  state  shape: {state_arrays.shape}")
    print(f"  action shape: {action_arrays.shape}")
    print(f"  head_camera shape: {head_camera_arrays.shape}")
    print(f"  n_episodes: {len(episode_ends_arrays)}  total frames: {total_count}")


if __name__ == "__main__":
    main()
