# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Regenerates a RoboCasa dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 224x224px resolution.
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - Successful demonstrations are saved to the main regenerated dataset.
    - All rollouts (successes and failures) are optionally saved to a separate rollout dataset.

Usage:
    # Inside robocasa directory and robocasa conda environment:
    uv run -m cosmos_policy.experiments.robot.robocasa.regenerate_robocasa_dataset \
        --dataset_path <PATH TO HDF5 DATASET DIR> \
        --target_dir <PATH TO TARGET DIR> \
        [--data_collection] [--jpeg_compress] [--local_log_dir <PATH>]

    Examples:
        # With data collection mode enabled (saves rollout episodes)
        # kitchen_coffee  kitchen_doors  kitchen_drawer  kitchen_microwave  kitchen_pnp  kitchen_sink  kitchen_stove
        # Specify --num_demos to limit the number of demos to process (default is None, which means use all demos). Episodes skipped due to a buggy timeout on environment reset do not count towards this limit.
        uv run -m cosmos_policy.experiments.robot.robocasa.regenerate_robocasa_dataset \
            --dataset_path cosmos_policy/robocasa/datasets/v0.1/single_stage/kitchen_coffee/ \
            --target_dir cosmos_policy/robocasa/datasets/v0.1/single_stage_regen/kitchen_coffee/ \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True
        uv run -m cosmos_policy.experiments.robot.robocasa.regenerate_robocasa_dataset \
            --dataset_path cosmos_policy/robocasa/datasets/v0.1/single_stage/kitchen_doors/ \
            --target_dir cosmos_policy/robocasa/datasets/v0.1/single_stage_regen/kitchen_doors/ \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True
        uv run -m cosmos_policy.experiments.robot.robocasa.regenerate_robocasa_dataset \
            --dataset_path cosmos_policy/robocasa/datasets/v0.1/single_stage/kitchen_drawer/ \
            --target_dir cosmos_policy/robocasa/datasets/v0.1/single_stage_regen/kitchen_drawer/ \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True
        uv run -m cosmos_policy.experiments.robot.robocasa.regenerate_robocasa_dataset \
            --dataset_path cosmos_policy/robocasa/datasets/v0.1/single_stage/kitchen_microwave/ \
            --target_dir cosmos_policy/robocasa/datasets/v0.1/single_stage_regen/kitchen_microwave/ \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True
        uv run -m cosmos_policy.experiments.robot.robocasa.regenerate_robocasa_dataset \
            --dataset_path cosmos_policy/robocasa/datasets/v0.1/single_stage/kitchen_pnp/ \
            --target_dir cosmos_policy/robocasa/datasets/v0.1/single_stage_regen/kitchen_pnp/ \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True
        uv run -m cosmos_policy.experiments.robot.robocasa.regenerate_robocasa_dataset \
            --dataset_path cosmos_policy/robocasa/datasets/v0.1/single_stage/kitchen_sink/ \
            --target_dir cosmos_policy/robocasa/datasets/v0.1/single_stage_regen/kitchen_sink/ \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True
        uv run -m cosmos_policy.experiments.robot.robocasa.regenerate_robocasa_dataset \
            --dataset_path cosmos_policy/robocasa/datasets/v0.1/single_stage/kitchen_stove/ \
            --target_dir cosmos_policy/robocasa/datasets/v0.1/single_stage_regen/kitchen_stove/ \
            --data_collection True --jpeg_compress True --local_log_dir ./regen_data/ --deterministic True


"""

import argparse
import datetime
import json
import os
import random
import signal
import time

import h5py
import imageio
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from termcolor import colored

from cosmos_policy.robocasa.robocasa.scripts.playback_dataset import (
    get_env_metadata_from_dataset,
    reset_to,
)

IMAGE_RESOLUTION = 224
DATE_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class TimeoutException(Exception):
    """Exception raised when an operation times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Operation timed out")


def str2bool(v):
    """Convert string representation of truth to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def jpeg_encode_image(image, quality=95):
    """Encode image as JPEG bytes."""
    import io

    from PIL import Image

    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def set_seed_everywhere(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def print_section_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_summary_stats(stats):
    """Print formatted summary statistics."""
    print_section_header("REGENERATION SUMMARY")

    total_episodes = stats["total_episodes_processed"]
    successful = stats["total_successful_episodes"]
    failed = stats["total_failed_episodes"]
    skipped = stats["total_skipped_episodes"]

    print("\nEpisode Statistics:")
    print(f"  - Total episodes processed: {total_episodes}")
    print(
        f"  - Successful episodes: {successful} ({successful / total_episodes * 100:.1f}%)"
        if total_episodes > 0
        else "  - Successful episodes: 0"
    )
    print(
        f"  - Failed episodes: {failed} ({failed / total_episodes * 100:.1f}%)"
        if total_episodes > 0
        else "  - Failed episodes: 0"
    )
    print(f"  - Skipped episodes (too short): {skipped}")

    print("\nAction Statistics:")
    print(f"  - Total actions in original demos: {stats['total_original_actions']}")
    print(f"  - Total actions replayed: {stats['total_actions_replayed']}")
    print(f"  - Total no-op actions filtered: {stats['total_noop_actions']}")
    print(f"  - Total timesteps trimmed: {stats['total_trimmed_timesteps']}")

    avg_noop = stats["total_noop_actions"] / total_episodes if total_episodes > 0 else 0
    avg_trimmed = stats["total_trimmed_timesteps"] / successful if successful > 0 else 0
    print("\nAverages:")
    print(f"  - Avg no-op actions per episode: {avg_noop:.1f}")
    print(f"  - Avg trimmed timesteps per success: {avg_trimmed:.1f}")

    print("\nFiles Processed:")
    print(f"  - Total HDF5 files: {stats['total_hdf5_files']}")

    print("\n" + "=" * 80 + "\n")


def main(args):
    if args.deterministic:
        set_seed_everywhere(seed=0)

    print_section_header(f"ROBOCASA DATASET REGENERATION - {DATE_TIME}")
    print(f"\nSource: {args.dataset_path}")
    print(f"Target: {args.target_dir}")
    print(f"Cameras: {args.camera_names}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Data collection: {args.data_collection}")
    if args.data_collection:
        print(f"JPEG compression: {args.jpeg_compress}")
        print(f"Rollout data dir: {args.local_log_dir}")
    print(f"Deterministic: {args.deterministic}")

    # Initialize statistics tracking
    stats = {
        "total_episodes_processed": 0,
        "total_successful_episodes": 0,
        "total_failed_episodes": 0,
        "total_skipped_episodes": 0,
        "total_noop_actions": 0,
        "total_trimmed_timesteps": 0,
        "total_original_actions": 0,
        "total_actions_replayed": 0,
        "total_hdf5_files": 0,
        "start_time": time.time(),
    }

    # Episode-level logging
    episode_logs = []

    # Create target directory
    if os.path.isdir(args.target_dir):
        user_input = input(
            f"\nTarget directory already exists at path: {args.target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: "
        )
        if user_input != "y":
            exit()
    os.makedirs(args.target_dir, exist_ok=True)

    # Create rollout data directory if data collection is enabled
    # Place it in the same outer directory as the regenerated demos dataset
    if args.data_collection:
        # Get the parent directory of target_dir
        target_parent = os.path.dirname(os.path.normpath(args.target_dir))
        # Get grandparent and parent basename
        target_grandparent = os.path.dirname(target_parent)
        target_parent_basename = os.path.basename(target_parent)
        # Create rollout data directory name
        rollout_parent_name = target_parent_basename + "_rollout_data"
        rollout_data_dir = os.path.join(target_grandparent, rollout_parent_name)
        os.makedirs(rollout_data_dir, exist_ok=True)
        print(f"\nData collection enabled. Rollout data will be saved to: {rollout_data_dir}")

    # Create video output directory in the same outer directory as the regenerated demos dataset
    # Get the parent directory of target_dir
    target_parent = os.path.dirname(os.path.normpath(args.target_dir))
    # Get grandparent and parent basename
    target_grandparent = os.path.dirname(target_parent)
    target_parent_basename = os.path.basename(target_parent)
    # Create video output directory name
    video_parent_name = target_parent_basename + "_temp_MP4_videos"
    video_out_dir = os.path.join(target_grandparent, video_parent_name)
    os.makedirs(video_out_dir, exist_ok=True)
    print(f"Video outputs will be saved to: {video_out_dir}")

    # Find all HDF5 files in the dataset path (recursively)
    hdf5_files = set()
    for root, dirs, files in os.walk(args.dataset_path, followlinks=True):
        for file in files:
            if "kitchen_navigate" in root or "kitchen_navigate" in dirs:
                continue  # Skip kitchen_navigate tasks
            if file.lower().endswith(("demo_gentex_im128_randcams.hdf5")):
                filepath = os.path.join(root, file)
                hdf5_files.add(filepath)
    hdf5_files = list(hdf5_files)
    hdf5_files.sort()

    stats["total_hdf5_files"] = len(hdf5_files)
    print(f"\nFound {len(hdf5_files)} HDF5 files to process")

    # Setup counters
    global_episode_counter = 0  # Global episode counter for data collection

    for file_idx, hdf5_file in enumerate(hdf5_files):
        print(f"\n{'─' * 80}")
        print(f"Processing file {file_idx + 1}/{len(hdf5_files)}: {os.path.basename(hdf5_file)}")
        print(f"{'─' * 80}")

        replayed_demos_counter = 0  # Counter for demos that were replayed (excludes only reset timeouts)

        # Get environment metadata
        env_meta = get_env_metadata_from_dataset(dataset_path=hdf5_file)
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_names"] = args.camera_names
        env_kwargs["camera_widths"] = args.image_size
        env_kwargs["camera_heights"] = args.image_size

        # # Uncomment below to save the controller configs (might be needed for policy evaluation)
        # with open("robocasa_controller_configs.pkl", "wb") as f:
        #     pickle.dump(env_meta['env_kwargs']['controller_configs'], f)

        if args.verbose:
            print(colored(f"Initializing environment for {env_kwargs['env_name']}...", "yellow"))

        env = robosuite.make(**env_kwargs)

        # Open original HDF5 file
        orig_data_file = h5py.File(hdf5_file, "r")

        # Create new HDF5 file for regenerated demos
        # Get relative path structure
        rel_path = os.path.relpath(hdf5_file, args.dataset_path)
        new_data_path = os.path.join(args.target_dir, rel_path)
        os.makedirs(os.path.dirname(new_data_path), exist_ok=True)

        # Create corresponding rollout data directory structure (if data collection is enabled)
        if args.data_collection:
            rollout_data_subdir = os.path.join(rollout_data_dir, os.path.dirname(rel_path))
            os.makedirs(rollout_data_subdir, exist_ok=True)

        # Create corresponding video output subdirectory structure
        video_out_subdir = os.path.join(video_out_dir, os.path.dirname(rel_path))
        os.makedirs(video_out_subdir, exist_ok=True)

        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        # List of all demonstration episodes (sorted in increasing number order)
        demos = list(orig_data_file["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        for ind in range(len(demos)):
            ep = demos[ind]
            demo_data = orig_data_file[f"data/{ep}"]

            # Get original states and actions
            orig_states = demo_data["states"][()]
            orig_actions = demo_data["actions"][()]

            # Prepare initial state
            initial_state = dict(states=orig_states[0])
            initial_state["model"] = demo_data.attrs["model_file"]
            initial_state["ep_meta"] = demo_data.attrs.get("ep_meta", None)

            # Parse episode metadata
            ep_meta = json.loads(initial_state["ep_meta"])
            lang = ep_meta.get("lang", "Unknown task")

            # Initialize episode log
            episode_log = {
                "episode_id": global_episode_counter + 1,
                "hdf5_file": os.path.basename(hdf5_file),
                "demo_name": ep,
                "task_description": lang,
                "original_num_actions": len(orig_actions),
                "initial_state": orig_states[0].tolist(),
                "ep_meta": ep_meta,
            }

            if args.verbose:
                print(colored(f"\nEpisode {ep}: {lang}", "green"))

            # Reset environment, set initial state, and wait a few steps for environment to settle
            try:
                # Set up timeout for reset operations
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(25)

                env.reset()
                reset_to(env, initial_state)

                # Disable the alarm
                signal.alarm(0)
            except (TimeoutException, ValueError) as e:
                # Timeout occurred (or ValueError from robosuite after timeout), skip this episode
                signal.alarm(0)  # Disable the alarm
                print(colored(f"Timeout during reset for episode {ep}, skipping to next episode", "yellow"))
                episode_log["status"] = "timeout"
                episode_log["reason"] = f"Reset operation timed out after 5 seconds: {str(e)}"
                episode_logs.append(episode_log)

                # Environment is now in a corrupted state, need to recreate it
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    del env
                except Exception:
                    pass

                # Recreate the environment (with no timeout during creation)
                if args.verbose:
                    print(colored("Recreating environment after timeout...", "yellow"))
                try:
                    # Ensure no alarm is active during environment creation
                    signal.alarm(0)
                    env = robosuite.make(**env_kwargs)
                except Exception as env_create_error:
                    print(colored(f"Failed to recreate environment: {env_create_error}", "red"))
                    print(colored("Skipping remaining episodes in this file.", "red"))
                    break

                continue

            for _ in range(10):
                # Deterministic seeding for reproducibility
                if args.deterministic:
                    set_seed_everywhere(seed=0)
                obs, reward, done, info = env.step(np.zeros(env.action_dim))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_left_images = []
            agentview_right_images = []
            eye_in_hand_images = []

            # Data collection buffers
            if args.data_collection:
                primary_images_list = []  # For left camera
                secondary_images_list = []  # For right camera
                wrist_images_list = []  # For wrist camera
                proprio_list = []  # For proprioceptive state
                actions_list = []  # For actions

            # Episode-specific counters
            ep_noop_count = 0
            ep_trimmed_count = 0

            # Replay original demo actions in environment and record observations
            success = False
            num_actions_replayed = 0
            for action_idx, action in enumerate(orig_actions):
                # Deterministic seeding for reproducibility
                if args.deterministic:
                    set_seed_everywhere(seed=0)

                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    if args.verbose:
                        print(f"\tSkipping no-op action: {action}")
                    ep_noop_count += 1
                    stats["total_noop_actions"] += 1
                    continue

                # Get current observation (before stepping)
                obs = env._get_observations()

                if states == []:
                    # In the first timestep, copy the initial state over
                    states.append(orig_states[0])
                else:
                    # For all other timesteps, get state from environment
                    states.append(env.sim.get_state().flatten())

                # Record original action (from demo)
                actions.append(action)

                # Record proprioceptive data
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                if "robot0_joint_pos" in obs:
                    joint_states.append(obs["robot0_joint_pos"])

                # End-effector state
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )

                # Robot state for compatibility
                robot_states.append(
                    np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                )

                # Get camera images and flip vertically (environment renders upside down)
                agentview_left_image = obs["robot0_agentview_left_image"][::-1]
                agentview_right_image = obs["robot0_agentview_right_image"][::-1]
                eye_in_hand_image = obs["robot0_eye_in_hand_image"][::-1]

                agentview_left_images.append(agentview_left_image)
                agentview_right_images.append(agentview_right_image)
                eye_in_hand_images.append(eye_in_hand_image)

                # Collect data for rollout dataset (if data collection mode is enabled)
                if args.data_collection:
                    primary_images_list.append(agentview_left_image)
                    secondary_images_list.append(agentview_right_image)
                    wrist_images_list.append(eye_in_hand_image)
                    # Combine proprioceptive data (gripper position + end-effector position + end-effector quaternion)
                    proprio = np.concatenate(
                        [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
                    )
                    proprio_list.append(proprio)
                    actions_list.append(action)

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())
                num_actions_replayed += 1

                # Check for success after executing action
                success = env._check_success()
                if success:
                    # Calculate how many steps were trimmed
                    ep_trimmed_count = len(orig_actions) - num_actions_replayed
                    stats["total_trimmed_timesteps"] += ep_trimmed_count
                    if ep_trimmed_count > 0:
                        print(
                            colored(
                                f"  [SUCCESS] Success detected! Trimmed {ep_trimmed_count} steps from end of demo.",
                                "cyan",
                            )
                        )
                    break

            # Update statistics
            stats["total_original_actions"] += len(orig_actions)
            stats["total_actions_replayed"] += num_actions_replayed

            # Update episode log
            episode_log["num_actions_replayed"] = num_actions_replayed
            episode_log["num_noop_actions_filtered"] = ep_noop_count
            episode_log["num_timesteps_trimmed"] = ep_trimmed_count
            episode_log["success"] = bool(success)

            # Skip dummy demos where there were less than 5 actions to replay
            if num_actions_replayed < 5:
                print(
                    colored(
                        f"  Skipping episode: less than 5 actions to replay (only {num_actions_replayed}).", "yellow"
                    )
                )
                stats["total_skipped_episodes"] += 1
                episode_log["skipped"] = True
                episode_log["skip_reason"] = "too_short"
                episode_logs.append(episode_log)
                continue

            episode_log["skipped"] = False
            stats["total_episodes_processed"] += 1

            # Save episodic data for rollout dataset (if data collection mode is enabled)
            # This captures ALL episodes regardless of success/failure
            if args.data_collection and len(primary_images_list) > 0:
                # Increment total episode counter
                global_episode_counter += 1

                # Get task name from episode metadata
                task_name = lang.replace(" ", "_")

                # Prepare collected data dictionary
                collected_data = dict(
                    primary_images=np.stack(primary_images_list, axis=0),  # (T, H, W, C) - left camera
                    secondary_images=np.stack(secondary_images_list, axis=0),  # (T, H, W, C) - right camera
                    wrist_images=np.stack(wrist_images_list, axis=0),  # (T, H, W, C)
                    proprio=np.stack(proprio_list, axis=0),  # (T, D)
                    actions=np.stack(actions_list, axis=0),  # (T, action_dim)
                    success=bool(success),  # True for successful episodes, False for unsuccessful
                )

                def _save_episode_data():
                    """Save collected episode data to HDF5 file."""
                    ep_filename = f"episode_data--task={task_name}--{DATE_TIME}--ep={global_episode_counter}--success={success}--regen_demo.hdf5"
                    ep_filepath = os.path.join(rollout_data_subdir, ep_filename)
                    with h5py.File(ep_filepath, "w") as f:
                        for k, v in collected_data.items():
                            if isinstance(v, np.ndarray):
                                is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                                if is_image and args.jpeg_compress:
                                    jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                                    dt = h5py.vlen_dtype(np.dtype("uint8"))
                                    f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                                else:
                                    f.create_dataset(k, data=v)
                            else:
                                f.attrs[k] = v
                        f.attrs["task_description"] = lang

                _save_episode_data()
                if args.verbose:
                    print(f"  Saved rollout episode data: episode={global_episode_counter}, success={success}")

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if success:
                dones = np.zeros(len(actions)).astype(np.uint8)
                dones[-1] = 1
                rewards = np.zeros(len(actions)).astype(np.uint8)
                rewards[-1] = 1
                assert len(actions) == len(agentview_left_images)

                ep_data_grp = grp.create_group(ep)
                obs_grp = ep_data_grp.create_group("obs")
                obs_grp.create_dataset("gripper_states", data=np.stack(gripper_states, axis=0))
                obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
                obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
                obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
                obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

                # Save image data (no JPEG compression for main dataset)
                obs_grp.create_dataset("robot0_agentview_left_rgb", data=np.stack(agentview_left_images, axis=0))
                obs_grp.create_dataset("robot0_agentview_right_rgb", data=np.stack(agentview_right_images, axis=0))
                obs_grp.create_dataset("robot0_eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0))

                ep_data_grp.create_dataset("actions", data=actions)
                ep_data_grp.create_dataset("states", data=np.stack(states))
                ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
                ep_data_grp.create_dataset("rewards", data=rewards)
                ep_data_grp.create_dataset("dones", data=dones)

                # Save task description as attribute
                ep_data_grp.attrs["task_description"] = lang

                stats["total_successful_episodes"] += 1
                print(colored("  Episode saved successfully", "green"))
            else:
                stats["total_failed_episodes"] += 1
                print(colored("  Episode failed", "red"))

            # Add to episode logs
            episode_logs.append(episode_log)

            # Save a video of the episode (for visualization)
            if len(agentview_left_images) > 0:
                # Sanitize task name for filesystem
                task_name_sanitized = lang.replace(" ", "_")
                mp4_path = os.path.join(
                    video_out_subdir,
                    f"ep={global_episode_counter}--task={task_name_sanitized}--success={bool(success)}.mp4",
                )
                video_writer = imageio.get_writer(mp4_path, fps=30)
                for img_left, img_right, img_wrist in zip(
                    agentview_left_images, agentview_right_images, eye_in_hand_images
                ):
                    # Concatenate images horizontally for visualization
                    combined_img = np.concatenate([img_left, img_right, img_wrist], axis=1)
                    video_writer.append_data(combined_img)
                video_writer.close()
                if args.verbose:
                    print(f"  Saved rollout MP4 at path {mp4_path}")

            # Progress update
            processed = stats["total_episodes_processed"]
            success_count = stats["total_successful_episodes"]
            if processed > 0:
                print(
                    f"\n  Progress: {processed} episodes processed | {success_count} successful ({success_count / processed * 100:.1f}%) | {stats['total_noop_actions']} no-ops filtered"
                )

            # Exit early if in debug mode (after 1 demo)
            if args.debug:
                print(colored("\nDebug mode: Stopping after 1 demo.", "cyan"))
                # Close HDF5 files and environment
                orig_data_file.close()
                new_data_file.close()
                env.close()
                print(f"Saved regenerated demos at: {new_data_path}")

                # Calculate elapsed time
                stats["end_time"] = time.time()
                stats["elapsed_time_seconds"] = stats["end_time"] - stats["start_time"]

                # Save summary statistics to separate file
                summary_log_path = os.path.join(args.target_dir, "regeneration_summary.json")
                with open(summary_log_path, "w") as f:
                    json.dump(
                        {
                            "summary_statistics": stats,
                            "arguments": vars(args),
                            "timestamp": DATE_TIME,
                        },
                        f,
                        indent=2,
                    )
                print(f"Saved summary statistics to: {summary_log_path}")

                # Save per-episode logs to separate file
                episodes_log_path = os.path.join(args.target_dir, "regeneration_episodes.json")
                with open(episodes_log_path, "w") as f:
                    json.dump(
                        {
                            "episode_logs": episode_logs,
                            "timestamp": DATE_TIME,
                        },
                        f,
                        indent=2,
                    )
                print(f"Saved per-episode logs to: {episodes_log_path}")

                # Print summary
                print_summary_stats(stats)

                if args.data_collection:
                    print(
                        f"Data collection complete! Saved {global_episode_counter} rollout episode(s) to: {rollout_data_dir}"
                    )
                    print(f"JPEG compression: {'Enabled' if args.jpeg_compress else 'Disabled'}")

                print(f"[LOG] Summary log saved at: {summary_log_path}")
                print(f"[LOG] Episodes log saved at: {episodes_log_path}")
                print(f"[VIDEO] MP4 videos saved at: {video_out_dir}")
                return

            # Check if we've reached the max number of demos to process
            replayed_demos_counter += 1
            if args.num_demos is not None and replayed_demos_counter >= args.num_demos:
                print(colored(f"\nReached max number of demos ({args.num_demos}), stopping processing.", "cyan"))
                break

        # Close HDF5 files
        orig_data_file.close()
        new_data_file.close()
        env.close()
        print(f"Saved regenerated demos at: {new_data_path}")

    # Calculate elapsed time
    stats["end_time"] = time.time()
    stats["elapsed_time_seconds"] = stats["end_time"] - stats["start_time"]
    stats["elapsed_time_formatted"] = str(datetime.timedelta(seconds=int(stats["elapsed_time_seconds"])))

    # Save summary statistics to separate file
    summary_log_path = os.path.join(args.target_dir, "regeneration_summary.json")
    with open(summary_log_path, "w") as f:
        json.dump(
            {
                "summary_statistics": stats,
                "arguments": vars(args),
                "timestamp": DATE_TIME,
            },
            f,
            indent=2,
        )
    print(f"\n[SUCCESS] Saved summary statistics to: {summary_log_path}")

    # Save per-episode logs to separate file
    episodes_log_path = os.path.join(args.target_dir, "regeneration_episodes.json")
    with open(episodes_log_path, "w") as f:
        json.dump(
            {
                "episode_logs": episode_logs,
                "timestamp": DATE_TIME,
            },
            f,
            indent=2,
        )
    print(f"[SUCCESS] Saved per-episode logs to: {episodes_log_path}")

    # Print summary statistics
    print_summary_stats(stats)
    print(f"[TIME] Total time elapsed: {stats['elapsed_time_formatted']}")

    # Report data collection summary if enabled
    if args.data_collection:
        print(
            f"\n[DATA] Data collection complete! Saved {global_episode_counter} rollout episodes to: {rollout_data_dir}"
        )
        print(f"[COMPRESSION] JPEG compression: {'Enabled' if args.jpeg_compress else 'Disabled'}")

    print(f"\n[COMPLETE] Dataset regeneration complete! Saved new dataset at: {args.target_dir}")
    print(f"[LOG] Summary log saved at: {summary_log_path}")
    print(f"[LOG] Episodes log saved at: {episodes_log_path}")
    print(f"[VIDEO] MP4 videos saved at: {video_out_dir}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to directory containing raw HDF5 dataset. Example: ./robocasa/datasets/v0.1/single_stage/",
        required=True,
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        help="Path to regenerated dataset directory. Example: ./robocasa/datasets/v0.1/single_stage_regen",
        required=True,
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="Camera names to use for observations",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size to use for rendering (square images)",
    )
    parser.add_argument(
        "--data_collection",
        type=str2bool,
        default=False,
        help="If True, save replayed demonstration episodes as rollout episodes. Accepts: yes/no, true/false, t/f, y/n, 1/0",
    )
    parser.add_argument(
        "--jpeg_compress",
        type=str2bool,
        default=False,
        help="If True, apply JPEG compression to rollout images (default: False). Accepts: yes/no, true/false, t/f, y/n, 1/0",
    )
    parser.add_argument(
        "--local_log_dir",
        type=str,
        default="./experiments/logs",
        help="Local directory for rollout data logs (used when data_collection=True)",
    )
    parser.add_argument(
        "--deterministic",
        type=str2bool,
        default=True,
        help="If True, use deterministic seed for environment (default: True). Accepts: yes/no, true/false, t/f, y/n, 1/0",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If True, print additional information",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If True, stop after replaying 1 demo (for testing)",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=None,
        help="Maximum number of demos to replay (default: None, meaning all demos). Episodes skipped due to reset timeout do not count towards this limit.",
    )
    args = parser.parse_args()

    # Start data regeneration
    main(args)
