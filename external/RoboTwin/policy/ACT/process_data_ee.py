"""Process RoboTwin hdf5 episodes into ACT training hdf5 with EE-Cartesian
actions/state instead of native joint-qpos.

Per frame, the 14-dim EE vector is:
    [ L_xyz(3), L_ypr(3), L_grip(1),  R_xyz(3), R_ypr(3), R_grip(1) ]
  - xyz  = /endpose/{side}_endpose[:3]                 (world position)
  - ypr  = R.from_quat([qx,qy,qz,qw]).as_euler("ZYX")  (yaw, pitch, roll)
           endpose[3:7] is world quaternion in wxyz order -> reorder to xyzw.
  - grip = /joint_action/{side}_gripper                (commanded gripper,
           same source native ACT uses for the gripper dim)

This 14-dim EE vector replaces the native joint 14-vector for BOTH
observations/qpos and action. Everything else (image handling, per-episode
hdf5 layout, qpos/action indexing) is kept identical to process_data.py so the
dataloader (utils.py) and downstream consumers see the exact same schema.

Output layout (matches process_data.py's per-episode hdf5):
    <output_root>/sim-<task>-<task_config>-<num>/episode_<i>.hdf5
      action                          (T, 14)  float
      observations/qpos               (T, 14)  float
      observations/left_arm_dim       (T,)     int
      observations/right_arm_dim      (T,)     int
      observations/images/cam_high        (T, 480, 640, 3) uint8
      observations/images/cam_right_wrist (T, 480, 640, 3) uint8
      observations/images/cam_left_wrist  (T, 480, 640, 3) uint8
"""

import sys

sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import cv2
import argparse
from scipy.spatial.transform import Rotation


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # commanded gripper (same source native ACT uses for the gripper dim)
        left_gripper = root["/joint_action/left_gripper"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        # native joint arms (only used to record left_arm_dim / right_arm_dim,
        # matching the original layout: 6 per side)
        left_arm = root["/joint_action/left_arm"][()]
        right_arm = root["/joint_action/right_arm"][()]
        # EE-Cartesian endpose: [x, y, z, qw, qx, qy, qz] (world)
        left_endpose = root["/endpose/left_endpose"][()]
        right_endpose = root["/endpose/right_endpose"][()]

        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return (
        left_gripper,
        left_arm,
        right_gripper,
        right_arm,
        left_endpose,
        right_endpose,
        image_dict,
    )


def endpose_to_ee(endpose_row, grip):
    """Convert one endpose row [x,y,z,qw,qx,qy,qz] + scalar grip -> 7-dim
    [x, y, z, yaw, pitch, roll, grip]."""
    xyz = endpose_row[:3]
    qw, qx, qy, qz = endpose_row[3], endpose_row[4], endpose_row[5], endpose_row[6]
    # scipy expects xyzw order
    ypr = Rotation.from_quat([qx, qy, qz, qw]).as_euler("ZYX")  # [yaw, pitch, roll]
    return np.concatenate([xyz, ypr, [grip]], axis=0)


def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            left_endpose_all,
            right_endpose_all,
            image_dict,
        ) = load_hdf5(os.path.join(path, f"episode{i}.hdf5"))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )
            left_endpose, right_endpose = left_endpose_all[j], right_endpose_all[j]

            if j != left_gripper_all.shape[0] - 1:
                # 14-dim EE-Cartesian vector (replaces native joint 14-vector)
                left_ee = endpose_to_ee(left_endpose, left_gripper)  # 7
                right_ee = endpose_to_ee(right_endpose, right_gripper)  # 7
                state = np.concatenate((left_ee, right_ee), axis=0)  # 14: EE

                state = state.astype(np.float32)
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            image.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
            image.create_dataset("cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8)
            image.create_dataset("cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RoboTwin episodes into EE-Cartesian ACT hdf5.")
    parser.add_argument("task_name", type=str, help="e.g. stack_bowls_three")
    parser.add_argument("task_config", type=str, help="output config label, e.g. demo_clean")
    parser.add_argument("expert_data_num", type=int, help="number of episodes, e.g. 50")
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default=None,
        help="dir containing episode<i>.hdf5. "
        "Default: /coc/flash7/scratch/egowm/robotwin/dataset/<task>/aloha-agilex_clean_50/data",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/coc/flash7/scratch/egowm/robotwin/_baselines_ee/ACT",
        help="output root; episodes go to <root>/sim-<task>-<config>-<num>/",
    )

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    raw_data_dir = args.raw_data_dir
    if raw_data_dir is None:
        raw_data_dir = f"/coc/flash7/scratch/egowm/robotwin/dataset/{task_name}/aloha-agilex_clean_50/data"

    save_path = os.path.join(args.output_root, f"sim-{task_name}-{task_config}-{expert_data_num}")

    print(f"[process_data_ee] raw_data_dir = {raw_data_dir}")
    print(f"[process_data_ee] save_path    = {save_path}")

    begin = data_transform(raw_data_dir, expert_data_num, save_path)
    print(f"[process_data_ee] done: {begin} episodes -> {save_path}")
