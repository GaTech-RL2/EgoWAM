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
run_libero_eval_overfit.py

Evaluates a trained policy on the training dataset episodes (overfitting evaluation).
This script uses the dataset configuration directly from the training config, including
its dataset statistics and T5 embeddings, to perform evaluation on the training data.

Usage example:
    uv run -m cosmos_policy.experiments.robot.libero.run_libero_eval_overfit \
        --config cosmos_predict2_2b_480p_libero_one_demo_one_episode \
        --ckpt_path /path/to/checkpoint \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --seed 195 \
        --deterministic True \
        --run_id_note overfit_eval \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1
"""

import json
import logging
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Optional

import draccus
import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
import wandb
from libero.libero import benchmark

from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy.experiments.robot.cosmos_utils import (
    WorkerPoolManager,
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_qvalue_prediction,
    get_t5_embedding_from_cache,
    get_value_prediction,
    query_model_parallel,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    save_rollout_video,
    save_rollout_video_with_future_image_predictions,
)
from cosmos_policy.experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    log_message,
    setup_logging,
)
from cosmos_policy.utils.utils import jpeg_encode_image, set_seed_everywhere

# Cosmos Policy latent sequence indices
# 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: action, 5: future proprio, 6: future wrist img, 7: future primary img, 8: value
CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 3
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = 5, 7

# Define max steps for each task suite (same as regular eval)
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class PolicyEvalOverfitConfig:
    # fmt: off
    suite: str = "libero"                                                # Evaluation suite name

    #################################################################################################################
    # Cosmos Policy-specific parameters
    #################################################################################################################
    model_family: str = "cosmos"                                         # Model family
    config: str = ""                                                     # Training config name (should match the checkpoint)
    ckpt_path: str = ""                                                  # Pretrained checkpoint path
    planning_model_config_name: str = ""                                 # Planning model config name
    planning_model_ckpt_path: str = ""                                   # Planning model checkpoint path
    config_file: str = "cosmos_policy/config/config.py"  # Cosmos default config file path

    use_third_person_image: bool = True                                  # Whether to include primary (third-person) image in input
    num_third_person_images: int = 1                                     # Number of third-person images to include in input (LIBERO: 1 agentview image)
    use_wrist_image: bool = True                                         # Whether to include wrist image in input
    num_wrist_images: int = 1                                            # Number of wrist images to include in input (LIBERO: 1 wrist image)
    use_proprio: bool = True                                             # Whether to include proprio state in input
    flip_images: bool = True                                             # Whether to flip images vertically across x-axis
    use_variance_scale: bool = False                                     # Whether to scale variance used to sample sigma max for denoising for increased diversity in generations
    use_jpeg_compression: bool = True                                    # Whether to use JPEG compression on images before querying policy
    ar_future_prediction: bool = False                                   # Whether to predict future state autoregressively
    ar_value_prediction: bool = False                                    # Whether to predict future state value autoregressively
    ar_qvalue_prediction: bool = False                                   # Whether to predict Q-value autoregressively
    num_denoising_steps_action: int = 5                                  # Number of denoising steps to take for action prediction
    num_denoising_steps_future_state: int = 1                            # Number of denoising steps to take for future state prediction (only applicable if ar_future_prediction is True; otherwise equal to num_denoising_steps_action)
    num_denoising_steps_value: int = 1                                   # Number of denoising steps to take for value prediction (only applicable if ar_value_prediction is True; otherwise equal to num_denoising_steps_action)
    unnormalize_actions: bool = True                                     # Unnormalize actions if trained with normalized actions
    normalize_proprio: bool = True                                       # Normalize proprio input if trained with normalized proprio
    trained_with_image_aug: bool = True                                  # Whether the model was trained with image augmentations (needed for test-time image transformations)
    chunk_size: int = 16                                                 # Number of actions to predict in chunk
    num_open_loop_steps: int = 16                                        # Number of actions in predicted chunk to execute open-loop before requerying policy

    deterministic: bool = True                                           # Whether to run in deterministic mode
    deterministic_reset: bool = False                                    # Whether to run in deterministic reset mode (sets global random seed right before env reset)
    deterministic_reset_seed: int = None                                 # (Only applicable if deterministic_reset==True) The seed to set before deterministic reset; if not provided, defaults to the base seed

    #################################################################################################################
    # Planning model and best-of-N search parameters
    #################################################################################################################
    use_ensemble_future_state_predictions: bool = False                  # Whether to use ensemble of future state predictions
    num_future_state_predictions_in_ensemble: int = 3                    # Number of future state predictions in ensemble
    future_state_ensemble_aggregation_scheme: str = "average"            # How to aggregate future state predictions in an ensemble of future state predictions (options: "average", "first")
    use_ensemble_value_predictions: bool = False                         # Whether to use ensemble of value predictions
    num_value_predictions_in_ensemble: int = 5                           # Number of value predictions in ensemble
    value_ensemble_aggregation_scheme: str = "average"                   # How to aggregate values in an ensemble of value predictions (options: "average", "gamma_weighted_average", "lcb", "success_vote", "majority_mean")
    search_depth: int = 1                                                # Number of levels to search through in the best-of-N search tree
    mask_current_state_action_for_value_prediction: bool = False         # Whether to use input masking to mask out certain inputs (current state and action) during value prediction
    mask_future_state_for_qvalue_prediction: bool = False                # Whether to use input masking to mask out certain inputs (future state) during Q(s, a) value prediction

    num_queries_best_of_n: int = 1                                       # Number of queries to make to the model (this is the N in best-of-N search)
    use_parallel_inference: bool = False                                 # Whether to use parallel inference across multiple GPUs
    available_gpus: str = "0,1,2,3,4,5,6,7"                              # Comma-separated list of GPU IDs available for use for parallel inference (defaults to all 8 GPUs on a node)
    parallel_timeout: int = 15                                           # Timeout in seconds for each parallel query

    #################################################################################################################
    # Overfitting evaluation parameters
    #################################################################################################################
    num_trials_per_episode: int = 1                                      # Number of rollouts per training episode
    env_img_res: int = 256                                               # Resolution for rendering environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"                            # Local directory for eval logs
    run_id_note: Optional[str] = None                                    # Extra note to add to end of run ID for logging

    use_wandb: bool = False                                              # Whether to also log results in Weights & Biases
    wandb_entity: str = "YOUR_ENTITY"                                    # Name of WandB entity
    wandb_project: str = "YOUR_PROJECT"                                  # Name of WandB project

    seed: int = 7                                                        # Random seed (for reproducibility)
    randomize_seed: bool = False                                         # Whether to randomize the seed for sampling

    #################################################################################################################
    # Data collection parameters
    #################################################################################################################
    data_collection: bool = False                                        # If True, save episodic data for later offline use
    jpeg_compress: bool = True                                           # If True, apply JPEG compression to images before saving

    # fmt: on


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def validate_config(cfg: PolicyEvalOverfitConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.ckpt_path is not None, "ckpt_path must not be None!"
    assert cfg.config is not None, "config must not be None!"

    if "image_aug" in str(cfg.ckpt_path):
        assert cfg.trained_with_image_aug, (
            "Expecting `trained_with_image_aug==True` because model was trained with image augmentations!"
        )


def get_task_suite_from_dataset_suite(dataset_suite: str) -> str:
    """Map dataset suite name to LIBERO task suite name."""
    # Dataset suite names might have suffixes like "_no_noops_rerendered"
    # Extract the base suite name
    if "libero_spatial" in dataset_suite:
        return "libero_spatial"
    elif "libero_object" in dataset_suite:
        return "libero_object"
    elif "libero_goal" in dataset_suite:
        return "libero_goal"
    elif "libero_10" in dataset_suite or "libero_long" in dataset_suite:
        return "libero_10"
    elif "libero_90" in dataset_suite:
        return "libero_90"
    else:
        # Default to libero_spatial if we can't determine
        return "libero_spatial"


def prepare_observation(obs, resize_size, flip_images: bool = False):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs, flip_images)
    wrist_img = get_libero_wrist_image(obs, flip_images)

    # Prepare observations dict
    observation = {
        "primary_image": img,
        "wrist_image": wrist_img,
        "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
    }

    return observation  # Return processed observation


def get_initial_state_from_episode_data(episode_data, env):
    """Extract initial state from episode data by setting environment to first observation."""
    # For overfitting evaluation, we'll use the default initial state from the environment
    # The training data's initial state might not be easily reproducible
    # Return None to use default initial state (environment will be reset in run_episode)
    return None


def run_episode(
    cfg: PolicyEvalOverfitConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    if cfg.deterministic_reset:
        reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
        set_seed_everywhere(reset_seed)
    
    # Reset environment - reset() may return the observation directly
    reset_result = env.reset()
    if isinstance(reset_result, dict):
        obs = reset_result
    elif reset_result is not None:
        # If reset() returns something but not a dict, it might be the observation
        obs = reset_result
    else:
        # If reset() returns None, take a dummy step to get observation
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)

    # Initialize action queue
    if cfg.num_open_loop_steps != cfg.chunk_size:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match cfg.chunk_size "
            f"{cfg.chunk_size}! For best performance (in terms of both speed and success rate), we "
            "recommend executing the full action chunk."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    replay_wrist_images = [] if cfg.use_wrist_image else None
    future_image_predictions_list = []
    # Use a default max_steps (will be determined from task suite if available)
    max_steps = 520  # Default to libero_10 max steps

    # Best-of-N search variables
    base_seed = cfg.seed  # Used for seed switching (if applicable)

    # Data collection buffers
    if cfg.data_collection:
        primary_images_list = []
        wrist_images_list = []
        proprio_list = []
        actions_list = []

    # Run episode
    success = False
    try:
        NUM_STEPS_WAIT = 10
        while t < max_steps + NUM_STEPS_WAIT:
            # If the deterministic flag is set, reset the random state with the same seed in every step
            if os.environ.get("DETERMINISTIC", "").lower() == "true":
                seed = 0
                set_seed_everywhere(seed)

            # Do nothing for the first few timesteps to let objects stabilize
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation = prepare_observation(obs, resize_size, cfg.flip_images)
            replay_images.append(observation["primary_image"])
            if replay_wrist_images is not None:
                replay_wrist_images.append(observation["wrist_image"])

            if cfg.data_collection:
                primary_images_list.append(observation["primary_image"])
                wrist_images_list.append(observation["wrist_image"])
                proprio_list.append(observation["proprio"])

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                best_actions = None
                best_future_predictions = None

                # Query model multiple times if value functions are available
                num_queries = cfg.num_queries_best_of_n

                # Use parallel inference if enabled and multiple queries are needed
                if cfg.use_parallel_inference and num_queries > 1 and worker_pool and worker_pool.initialized:
                    # Query model in parallel
                    start_time = time.time()
                    query_results = query_model_parallel(
                        cfg, observation, task_description, worker_pool, cfg.parallel_timeout
                    )
                    total_query_time = time.time() - start_time

                    log_message(
                        f"Parallel queries completed: {len(query_results)} results in {total_query_time:.3f}s", log_file
                    )

                else:
                    # Serial execution (original behavior)
                    query_results = []
                    for query_idx in range(num_queries):
                        actions_by_depth = []  # Action chunks across all depths of the search
                        future_image_predictions_by_depth = []  # Future image predictions across all depths of the search
                        value_predictions_by_depth = []  # Value predictions across all depths of the search
                        return_dict = {}
                        # Query model to get action
                        start_time = time.time()
                        action_return_dict = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            observation,
                            task_description,
                            seed=cfg.seed + query_idx,
                            randomize_seed=cfg.randomize_seed,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=not (
                                cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
                            ),
                        )
                        query_time = time.time() - start_time
                        log_message(
                            f"Query {query_idx + 1}/{num_queries}: Action query time = {query_time:.3f} sec", log_file
                        )
                        return_dict["actions"] = action_return_dict["actions"]
                        actions_by_depth.append(return_dict["actions"])

                        if cfg.ar_future_prediction:
                            # Autoregressively query model to get future state prediction
                            start_time = time.time()
                            future_state_return_dict = get_future_state_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                generated_latent_with_action=action_return_dict["generated_latent"],
                                orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
                                future_proprio_latent_idx=action_return_dict["latent_indices"][
                                    "future_proprio_latent_idx"
                                ],
                                future_wrist_image_latent_idx=action_return_dict["latent_indices"][
                                    "future_wrist_image_latent_idx"
                                ],
                                future_wrist_image2_latent_idx=action_return_dict["latent_indices"][
                                    "future_wrist_image2_latent_idx"
                                ],
                                future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
                                future_image2_latent_idx=action_return_dict["latent_indices"][
                                    "future_image2_latent_idx"
                                ],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
                                use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
                                num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
                                future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Future state prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["future_image_predictions"] = future_state_return_dict[
                                "future_image_predictions"
                            ]
                            future_image_predictions_by_depth.append(return_dict["future_image_predictions"])

                        else:
                            return_dict["future_image_predictions"] = action_return_dict["future_image_predictions"]

                        if cfg.ar_value_prediction:
                            # Autoregressively query model to get value prediction
                            start_time = time.time()
                            value_return_dict = get_value_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                future_state_samples_list=future_state_return_dict["future_state_samples_list"],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_value=cfg.num_denoising_steps_value,
                                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}",
                                log_file,
                            )
                        elif cfg.ar_qvalue_prediction:
                            # Autoregressively query model to get Q-value prediction
                            start_time = time.time()
                            value_return_dict = get_qvalue_prediction(
                                cfg,
                                model=planning_model if planning_model is not None else model,
                                data_batch=action_return_dict["data_batch"],
                                action_sample=action_return_dict["generated_latent"],
                                seed=cfg.seed + query_idx,
                                randomize_seed=cfg.randomize_seed,
                                num_denoising_steps_value=cfg.num_denoising_steps_value,
                                use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
                                num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
                            )
                            query_time = time.time() - start_time
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction query time = {query_time:.3f} sec",
                                log_file,
                            )
                            return_dict["value_prediction"] = value_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])
                            log_message(
                                f"Query {query_idx + 1}/{num_queries}: Value prediction: {return_dict['value_prediction']:.4f}",
                                log_file,
                            )
                        else:
                            return_dict["value_prediction"] = action_return_dict["value_prediction"]
                            value_predictions_by_depth.append(return_dict["value_prediction"])

                        return_dict["future_image_predictions_by_depth"] = future_image_predictions_by_depth
                        return_dict["value_predictions_by_depth"] = value_predictions_by_depth
                        return_dict["actions_by_depth"] = actions_by_depth
                        query_results.append(return_dict)

                # Print all value predictions
                log_message(f"t={t}: Current base seed: {base_seed}", log_file)
                for query_idx, return_dict in enumerate(query_results):
                    predicted_value = return_dict["value_prediction"]
                    log_message(
                        f"Query {query_idx + 1}/{num_queries} (seed {cfg.seed + query_idx}): Predicted value = {predicted_value:.4f}",
                        log_file,
                    )
                # Get dict: seed number -> (action chunk, future state, value)
                seed_to_return_dict = {
                    cfg.seed + query_idx: (
                        return_dict["actions"],
                        return_dict["future_image_predictions"],
                        return_dict["value_prediction"],
                    )
                    for query_idx, return_dict in enumerate(query_results)
                }
                # Get seed with highest value
                best_seed, best_return_dict = max(seed_to_return_dict.items(), key=lambda x: x[1][2])
                best_actions = best_return_dict[0]
                best_future_predictions = best_return_dict[1]
                best_value_predictions = best_return_dict[2]
                # Use the best actions, future predictions, and value predictions found
                action_queue.extend(best_actions)
                future_image_predictions_list.append(best_future_predictions)
                log_message(f"t={t}: Selected seed {best_seed} with value = {best_value_predictions:.4f}", log_file)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            print(f"t: {t}\t action: {action}")

            if cfg.data_collection:
                actions_list.append(action.copy())

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        error_msg = f"Episode error: {e}"
        traceback_str = traceback.format_exc()
        log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    # Fill data collection buffers
    if cfg.data_collection:
        collected_data = dict(
            primary_images=np.stack(primary_images_list, axis=0),  # (T, H, W, C)
            wrist_images=np.stack(wrist_images_list, axis=0),  # (T, H, W, C)
            proprio=np.stack(proprio_list, axis=0),  # (T, D)
            actions=np.stack(actions_list, axis=0),  # (T, action_dim)
            success=success,
        )
        # Add future image predictions if available
        if len(future_image_predictions_list) > 0:
            if cfg.use_third_person_image:
                future_primary_images = [
                    x["future_image"] for x in future_image_predictions_list if x["future_image"] is not None
                ]
                if len(future_primary_images) > 0:
                    collected_data["future_primary_images"] = np.stack(future_primary_images, axis=0)
            # Wrist image predictions (may be None depending on config)
            if (
                cfg.use_wrist_image
                and "future_wrist_image" in future_image_predictions_list[0]
                and future_image_predictions_list[0]["future_wrist_image"] is not None
            ):
                future_wrist_images = [x["future_wrist_image"] for x in future_image_predictions_list]
                collected_data["future_wrist_images"] = np.stack(future_wrist_images, axis=0)
    else:
        collected_data = None

    return success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data


def run_training_episode_eval(
    cfg: PolicyEvalOverfitConfig,
    episode_idx: int,
    episode_data: dict,
    model,
    planning_model,
    dataset_stats,
    worker_pool,
    resize_size,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single training episode."""
    # Extract episode information
    task_description = episode_data["command"]
    dataset_suite = episode_data.get("suite", "libero_spatial")
    task_suite_name = get_task_suite_from_dataset_suite(dataset_suite)
    
    # Get max steps for this task suite
    max_steps = TASK_MAX_STEPS.get(task_suite_name, 520)

    log_message(f"\nTraining Episode {episode_idx}: {task_description}", log_file)
    log_message(f"Dataset suite: {dataset_suite}, Task suite: {task_suite_name}", log_file)

    # Initialize LIBERO environment
    # We need to find a task from the benchmark that matches the task description
    # For now, we'll use the first task from the appropriate suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    # Use the first task from the suite (we'll match by description if possible)
    # For simplicity, use task 0
    task = task_suite.get_task(0)
    env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    
    # Override task description with the one from training data
    task_description = episode_data["command"]

    # Start episodes
    episode_successes = 0
    for trial_idx in tqdm.tqdm(range(cfg.num_trials_per_episode)):
        log_message(f"Starting trial {trial_idx + 1}/{cfg.num_trials_per_episode} for episode {episode_idx}...", log_file)

        # Get initial state (for now, use None to use default reset)
        initial_state = get_initial_state_from_episode_data(episode_data, env)

        # Run episode
        success, replay_images, replay_wrist_images, future_image_predictions_list, collected_data = run_episode(
            cfg,
            env,
            task_description,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            resize_size,
            initial_state,
            log_file,
        )

        # Update counters
        total_episodes += 1
        if success:
            episode_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            log_file=log_file,
        )

        # Save replay video with future image predictions included
        future_primary_image_predictions = None
        if cfg.use_third_person_image:
            future_primary_image_predictions = [x["future_image"] for x in future_image_predictions_list]
        future_wrist_image_predictions = None
        if cfg.use_wrist_image:
            future_wrist_image_predictions = [x["future_wrist_image"] for x in future_image_predictions_list]
        save_rollout_video_with_future_image_predictions(
            replay_images,
            total_episodes,
            success=success,
            task_description=task_description,
            chunk_size=cfg.chunk_size,
            num_open_loop_steps=cfg.num_open_loop_steps,
            rollout_wrist_images=replay_wrist_images,
            future_primary_image_predictions=future_primary_image_predictions,
            future_wrist_image_predictions=future_wrist_image_predictions,
            log_file=log_file,
            show_diff=False,
        )

        # Save episodic data (in data collection mode)
        if cfg.data_collection and collected_data is not None:

            def _save_episode_data():
                """Save collected episode data to HDF5 file."""
                ep_filename = f"episode_data--overfit--{DATE_TIME}--ep={episode_idx}--trial={trial_idx}--total_ep={total_episodes}--success={success}--{cfg.run_id_note}.hdf5"
                rollout_data_dir = os.path.join(cfg.local_log_dir, "rollout_data")
                os.makedirs(rollout_data_dir, exist_ok=True)
                ep_filepath = os.path.join(rollout_data_dir, ep_filename)
                with h5py.File(ep_filepath, "w") as f:
                    for k, v in collected_data.items():
                        if isinstance(v, np.ndarray):
                            is_image = v.ndim == 4 and v.shape[-1] == 3 and v.dtype == np.uint8
                            if is_image and cfg.jpeg_compress:
                                jpeg_list = [jpeg_encode_image(frame, quality=95) for frame in v]
                                dt = h5py.vlen_dtype(np.dtype("uint8"))
                                f.create_dataset(k + "_jpeg", data=jpeg_list, dtype=dt)
                            else:
                                f.create_dataset(k, data=v)
                        else:
                            f.attrs[k] = v
                    f.attrs["task_description"] = task_description
                    f.attrs["training_episode_idx"] = episode_idx

            _save_episode_data()

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log episode results
    episode_success_rate = float(episode_successes) / float(cfg.num_trials_per_episode) if cfg.num_trials_per_episode > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Episode {episode_idx} success rate: {episode_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/overfit/episode_{episode_idx}": episode_success_rate,
                f"num_episodes/overfit/episode_{episode_idx}": cfg.num_trials_per_episode,
                f"num_successes/overfit/episode_{episode_idx}": episode_successes,
            },
        )

    return (
        total_episodes,
        total_successes,
    )


@draccus.wrap()
def eval_libero_overfit(cfg: PolicyEvalOverfitConfig):
    """Main function to evaluate a trained policy on training dataset episodes (overfitting evaluation)."""

    # Set DETERMINISTIC environment variable if on deterministic mode (makes some model operations deterministic)
    assert not (cfg.deterministic and cfg.randomize_seed), (
        "Cannot enable both deterministic mode and randomize seed mode!"
    )
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    # Set multiprocessing start method if using parallel inference
    if cfg.use_parallel_inference:
        mp.set_start_method("spawn", force=True)

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Load model and config
    model, cosmos_config = get_model(cfg)
    
    # Extract dataset from config
    log_message("Instantiating dataset from config...", None)
    dataset = instantiate(cosmos_config.dataloader_train.dataset)
    
    # Extract dataset statistics and T5 embeddings from dataset
    dataset_stats = dataset.dataset_stats
    t5_text_embeddings = getattr(dataset, 't5_text_embeddings', None)
    
    log_message(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_steps} steps", None)
    log_message(f"Dataset statistics keys: {list(dataset_stats.keys())}", None)
    if t5_text_embeddings is not None:
        log_message(f"T5 embeddings loaded: {len(t5_text_embeddings)} embeddings", None)
    else:
        log_message("Warning: T5 embeddings not found in dataset", None)

    # Initialize T5 text embeddings cache from dataset
    # The T5 embeddings are already loaded in the dataset, but we need to initialize the cache
    # for the evaluation functions to use them
    if t5_text_embeddings is not None:
        from cosmos_policy.experiments.robot.cosmos_utils import t5_text_embeddings_cache, DEVICE
        # Move embeddings to the appropriate device if they're tensors
        device = DEVICE
        for key, value in t5_text_embeddings.items():
            if isinstance(value, torch.Tensor):
                t5_text_embeddings_cache[key] = value.to(device)
            else:
                t5_text_embeddings_cache[key] = value
        log_message(f"T5 embeddings cache initialized from dataset ({len(t5_text_embeddings_cache)} embeddings)", None)

    # Verify chunk size matches
    assert cfg.chunk_size == dataset.chunk_size, (
        f"Mismatch found between train and test chunk sizes! Train: {dataset.chunk_size}, Test: {cfg.chunk_size}"
    )

    # If using parallel inference, initialize worker pool
    worker_pool = None
    if cfg.use_parallel_inference:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        available_gpus = available_gpus[: cfg.num_queries_best_of_n]  # Only need N parallel workers
        worker_pool = WorkerPoolManager(cfg, dataset_stats, available_gpus)
        model = None
        planning_model = None
    else:
        worker_pool = None

        # Initialize model for world model and value function
        if cfg.planning_model_ckpt_path != "":
            planning_model, _ = get_planning_model(cfg)
        else:
            planning_model = None

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg.model_family)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(
        cfg=cfg,
        task_identifier="overfit",
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Eval config: {cfg}", log_file)
    log_message(f"Evaluating on {dataset.num_episodes} training episodes", log_file)

    # Log parallel inference configuration and start worker pool
    if cfg.use_parallel_inference and worker_pool:
        available_gpus = [int(gpu.strip()) for gpu in cfg.available_gpus.split(",")]
        available_gpus = available_gpus[: cfg.num_queries_best_of_n]
        log_message(f"Parallel inference enabled on GPUs: {available_gpus}", log_file)
        log_message(f"Parallel timeout: {cfg.parallel_timeout}s", log_file)
        log_message(f"Multiprocessing start method: {mp.get_start_method()}", log_file)

        # Verify GPUs are available
        for gpu_id in available_gpus:
            if gpu_id >= torch.cuda.device_count():
                log_message(
                    f"Warning: GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs found)", log_file
                )

        # Start worker pool
        try:
            log_message("Starting worker pool...", log_file)
            worker_pool.start_workers()
            log_message("Worker pool started successfully", log_file)
        except Exception as e:
            error_msg = f"Failed to start worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)
            log_message("Disabling parallel inference for this run", log_file)
            worker_pool = None
    else:
        log_message("Using serial inference (parallel inference disabled)", log_file)

    # Start evaluation on training episodes
    total_episodes, total_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(dataset.num_episodes)):
        episode_data = dataset.data[episode_idx]
        (
            total_episodes,
            total_successes,
        ) = run_training_episode_eval(
            cfg,
            episode_idx,
            episode_data,
            model,
            planning_model,
            dataset_stats,
            worker_pool,
            resize_size,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    
    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/overfit/total": final_success_rate,
                f"num_episodes/overfit/total": total_episodes,
                f"num_successes/overfit/total": total_successes,
            },
        )
        wandb.save(local_log_filepath)

    # Cleanup worker pool
    if worker_pool:
        try:
            worker_pool.shutdown()
        except Exception as e:
            error_msg = f"Error shutting down worker pool: {e}"
            traceback_str = traceback.format_exc()
            log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero_overfit()
