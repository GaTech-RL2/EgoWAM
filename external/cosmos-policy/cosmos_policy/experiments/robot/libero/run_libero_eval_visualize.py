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
run_libero_eval_visualize.py

Visualizes model predictions on training dataset episodes without running simulation rollouts.
Shows side-by-side comparisons of ground truth videos vs predicted future images, with action
trajectories and value predictions overlaid on the videos.

Usage example:
    uv run -m cosmos_policy.experiments.robot.libero.run_libero_eval_visualize \
        --config cosmos_predict2_2b_480p_libero_one_demo_one_episode \
        --ckpt_path /path/to/checkpoint \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --seed 195 \
        --deterministic True \
        --run_id_note visualize \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --episode_indices 0 \
        --save_video_path ./visualizations
"""

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional

import draccus
import imageio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image, ImageDraw, ImageFont

from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    get_value_prediction,
    unnormalize_actions,
)
from cosmos_policy.experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    log_message,
    setup_logging,
)
from cosmos_policy.utils.utils import set_seed_everywhere

# Cosmos Policy latent sequence indices
# 0: blank, 1: curr proprio, 2: curr wrist img, 3: curr primary img, 4: action, 5: future proprio, 6: future wrist img, 7: future primary img, 8: value
CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 3
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = 5, 7


@dataclass
class PolicyEvalVisualizeConfig:
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

    deterministic: bool = True                                           # Whether to run in deterministic mode

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
    available_gpus: str = "0"                                             # Comma-separated list of GPU IDs available for use

    #################################################################################################################
    # Visualization parameters
    #################################################################################################################
    episode_indices: Optional[str] = None                                # Comma-separated episode indices to visualize (e.g., "0,1,2") or None for all
    max_episodes: Optional[int] = None                                    # Maximum number of episodes to process (None = all)
    save_video_path: str = "./visualizations"                            # Output directory for visualization videos
    action_trajectory_style: str = "overlay"                             # How to display actions: "overlay" (on video) or "plot" (separate plot)
    chunk_visualization: bool = True                                     # Whether to show action chunks or individual actions

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"                            # Local directory for eval logs
    run_id_note: Optional[str] = None                                    # Extra note to add to end of run ID for logging

    seed: int = 7                                                        # Random seed (for reproducibility)
    randomize_seed: bool = False                                         # Whether to randomize the seed for sampling

    # fmt: on


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def validate_config(cfg: PolicyEvalVisualizeConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.ckpt_path is not None, "ckpt_path must not be None!"
    assert cfg.config is not None, "config must not be None!"

    if "image_aug" in str(cfg.ckpt_path):
        assert cfg.trained_with_image_aug, (
            "Expecting `trained_with_image_aug==True` because model was trained with image augmentations!"
        )


def prepare_observation_from_data(primary_image, wrist_image, proprio, flip_images: bool = False):
    """Prepare observation dictionary from training data."""
    # Flip images if needed
    if flip_images:
        primary_image = np.flipud(primary_image)
        wrist_image = np.flipud(wrist_image)
    
    observation = {
        "primary_image": primary_image,
        "wrist_image": wrist_image,
        "proprio": proprio,
    }
    return observation


def add_text_overlays(img: np.ndarray, timestep: int, gt_value: Optional[float], pred_value: Optional[float], 
                     font_size: int = 20) -> np.ndarray:
    """Add text overlays for timestep, GT value, and predicted value to image."""
    # Convert to PIL Image for text drawing
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to use a standard font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("Verdana", font_size)
            except IOError:
                font = ImageFont.load_default()
    
    # Prepare text
    text_parts = [f"t={timestep}"]
    if gt_value is not None:
        text_parts.append(f"GT:{gt_value:.3f}")
    if pred_value is not None:
        text_parts.append(f"Pred:{pred_value:.3f}")
    
    text = " | ".join(text_parts)
    
    # Draw text with background for readability
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw semi-transparent background
    padding = 5
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [10, 10, 10 + text_width + 2 * padding, 10 + text_height + 2 * padding],
        fill=(0, 0, 0, 180)  # Semi-transparent black
    )
    pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(pil_img)
    
    # Draw text in white
    draw.text((10 + padding, 10 + padding), text, font=font, fill=(255, 255, 255))
    
    return np.array(pil_img)


def overlay_action_trajectories(img: np.ndarray, gt_actions: np.ndarray, pred_actions: np.ndarray,
                                current_timestep: int, history_length: int = 50) -> np.ndarray:
    """Overlay action trajectories on video frame."""
    h, w = img.shape[:2]
    
    # Create a plot area at the bottom of the image
    plot_height = min(150, h // 4)
    
    # Get action history (last history_length steps)
    start_idx = max(0, current_timestep - history_length)
    end_idx = min(current_timestep + 1, len(gt_actions), len(pred_actions))
    
    if end_idx <= start_idx:
        return img
    
    gt_history = gt_actions[start_idx:end_idx]
    pred_history = pred_actions[start_idx:end_idx]
    
    if len(gt_history) == 0 or len(pred_history) == 0:
        return img
    
    # Create matplotlib figure
    action_dim = gt_actions.shape[1]
    fig, axes = plt.subplots(1, action_dim, figsize=(w/100, plot_height/100), dpi=100)
    if action_dim == 1:
        axes = [axes]
    
    time_steps = np.arange(start_idx, end_idx)
    
    # Use a single legend for all subplots (placed at the top)
    handles = None
    labels = None
    
    for i, ax in enumerate(axes):
        ax.clear()
        
        if len(time_steps) > 0:
            line1 = ax.plot(time_steps, gt_history[:, i], 'b-', linewidth=1.5)
            line2 = ax.plot(time_steps, pred_history[:, i], 'r--', linewidth=1.5)
            line3 = ax.axvline(x=current_timestep, color='g', linestyle=':', linewidth=1)
            
            # Store handles and labels for legend (only once)
            if handles is None:
                handles = [line1[0], line2[0], line3]
                labels = ['GT', 'Pred', 'Current']
        
        # Set y-axis limits based on data range
        all_values = np.concatenate([gt_history[:, i], pred_history[:, i]])
        y_min, y_max = all_values.min(), all_values.max()
        y_range = y_max - y_min
        if y_range < 0.1:
            y_range = 0.1
        ax.set_ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])
        ax.set_title(f'A{i}', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    
    # Add a single legend at the top of the figure
    if handles is not None:
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=6, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for legend
    
    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    # Use modern matplotlib API: buffer_rgba() returns RGBA, convert to RGB
    buf = fig.canvas.buffer_rgba()
    plot_array = np.asarray(buf)
    # Convert RGBA to RGB by dropping alpha channel
    plot_array = plot_array[:, :, :3]
    plt.close(fig)
    
    # Resize plot to match desired height
    plot_pil = Image.fromarray(plot_array)
    plot_pil = plot_pil.resize((w, plot_height), Image.Resampling.LANCZOS)
    plot_array = np.array(plot_pil)
    
    # Combine image and plot (replace bottom portion of image)
    if h > plot_height:
        combined_img = np.vstack([img[:-plot_height], plot_array])
    else:
        combined_img = plot_array
    
    return combined_img


def create_side_by_side_frame(gt_primary: np.ndarray, gt_wrist: np.ndarray,
                             pred_primary: np.ndarray, pred_wrist: np.ndarray,
                             timestep: int, gt_value: Optional[float], pred_value: Optional[float],
                             gt_actions: Optional[np.ndarray] = None,
                             pred_actions: Optional[np.ndarray] = None,
                             current_timestep: int = 0,
                             target_size: int = 256) -> np.ndarray:
    """Create a side-by-side frame showing GT vs predicted images."""
    # Resize images to target size
    def resize_img(img, size):
        pil_img = Image.fromarray(img)
        return np.array(pil_img.resize((size, size), Image.Resampling.LANCZOS))
    
    gt_primary_resized = resize_img(gt_primary, target_size)
    gt_wrist_resized = resize_img(gt_wrist, target_size)
    pred_primary_resized = resize_img(pred_primary, target_size)
    pred_wrist_resized = resize_img(pred_wrist, target_size)
    
    # Add text overlays
    gt_primary_with_text = add_text_overlays(gt_primary_resized, timestep, gt_value, None)
    gt_wrist_with_text = add_text_overlays(gt_wrist_resized, timestep, gt_value, None)
    pred_primary_with_text = add_text_overlays(pred_primary_resized, timestep, None, pred_value)
    pred_wrist_with_text = add_text_overlays(pred_wrist_resized, timestep, None, pred_value)
    
    # Create side-by-side layout
    # Top row: GT Primary | Predicted Primary
    # Bottom row: GT Wrist | Predicted Wrist
    top_row = np.hstack([gt_primary_with_text, pred_primary_with_text])
    bottom_row = np.hstack([gt_wrist_with_text, pred_wrist_with_text])
    
    # Add action trajectories if provided
    if gt_actions is not None and pred_actions is not None and len(gt_actions) > 0 and len(pred_actions) > 0:
        # Overlay trajectories on bottom row
        bottom_row = overlay_action_trajectories(bottom_row, gt_actions, pred_actions, current_timestep)
    
    frame = np.vstack([top_row, bottom_row])
    
    return frame


def get_model_predictions(cfg: PolicyEvalVisualizeConfig, model, planning_model, dataset_stats,
                          observation: dict, task_description: str, timestep: int) -> dict:
    """Run model inference to get predictions for a single timestep."""
    # Get action prediction
    action_return_dict = get_action(
        cfg,
        model,
        dataset_stats,
        observation,
        task_description,
        seed=cfg.seed + timestep,
        randomize_seed=cfg.randomize_seed,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=not (
            cfg.ar_future_prediction or cfg.ar_value_prediction or cfg.ar_qvalue_prediction
        ),
    )
    
    predictions = {
        "actions": action_return_dict["actions"],
        "future_image_predictions": action_return_dict.get("future_image_predictions", {}),
        "value_prediction": action_return_dict.get("value_prediction", None),
    }
    
    # Get future state prediction if needed
    if cfg.ar_future_prediction:
        future_state_return_dict = get_future_state_prediction(
            cfg,
            model=planning_model if planning_model is not None else model,
            data_batch=action_return_dict["data_batch"],
            generated_latent_with_action=action_return_dict["generated_latent"],
            orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
            future_proprio_latent_idx=action_return_dict["latent_indices"]["future_proprio_latent_idx"],
            future_wrist_image_latent_idx=action_return_dict["latent_indices"]["future_wrist_image_latent_idx"],
            future_wrist_image2_latent_idx=action_return_dict["latent_indices"]["future_wrist_image2_latent_idx"],
            future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
            future_image2_latent_idx=action_return_dict["latent_indices"]["future_image2_latent_idx"],
            seed=cfg.seed + timestep,
            randomize_seed=cfg.randomize_seed,
            num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
            use_ensemble_future_state_predictions=cfg.use_ensemble_future_state_predictions,
            num_future_state_predictions_in_ensemble=cfg.num_future_state_predictions_in_ensemble,
            future_state_ensemble_aggregation_scheme=cfg.future_state_ensemble_aggregation_scheme,
        )
        predictions["future_image_predictions"] = future_state_return_dict["future_image_predictions"]
    
    # Get value prediction if needed
    if cfg.ar_value_prediction:
        value_return_dict = get_value_prediction(
            cfg,
            model=planning_model if planning_model is not None else model,
            data_batch=action_return_dict["data_batch"],
            future_state_samples_list=action_return_dict.get("future_state_samples_list", []),
            seed=cfg.seed + timestep,
            randomize_seed=cfg.randomize_seed,
            num_denoising_steps_value=cfg.num_denoising_steps_value,
            use_ensemble_value_predictions=cfg.use_ensemble_value_predictions,
            num_value_predictions_in_ensemble=cfg.num_value_predictions_in_ensemble,
        )
        predictions["value_prediction"] = value_return_dict["value_prediction"]
    
    return predictions


def visualize_episode_predictions(
    cfg: PolicyEvalVisualizeConfig,
    episode_idx: int,
    episode_data: dict,
    model,
    planning_model,
    dataset_stats,
    resize_size: int,
    log_file=None,
) -> str:
    """Visualize predictions for a single training episode."""
    # Extract episode information
    task_description = episode_data["command"]
    gt_images = episode_data["images"]  # (T, H, W, 3)
    gt_wrist_images = episode_data["wrist_images"]  # (T, H, W, 3)
    gt_actions = episode_data["actions"].copy()  # (T, action_dim) - may be normalized
    gt_proprio = episode_data["proprio"]  # (T, proprio_dim)
    gt_returns = episode_data.get("returns", None)  # (T,) - GT values
    
    num_steps = len(gt_images)
    
    # Unnormalize GT actions if they are normalized (for comparison with predictions)
    # The dataset may have normalized actions, but we want to compare in original scale
    if cfg.unnormalize_actions:
        # Check if actions are in normalized range [-1, 1]
        if np.all(gt_actions >= -1.1) and np.all(gt_actions <= 1.1):
            # Actions appear to be normalized, unnormalize them
            gt_actions = unnormalize_actions(gt_actions, dataset_stats)
    
    log_message(f"\nVisualizing Episode {episode_idx}: {task_description}", log_file)
    log_message(f"Episode length: {num_steps} steps", log_file)
    
    # Prepare output directory
    os.makedirs(cfg.save_video_path, exist_ok=True)
    
    # Initialize video writer
    video_filename = f"episode_{episode_idx}_comparison_{cfg.run_id_note or ''}.mp4"
    video_path = os.path.join(cfg.save_video_path, video_filename)
    video_writer = imageio.get_writer(video_path, fps=10)
    
    # Initialize prediction arrays
    pred_actions_all = []
    pred_values_all = []
    
    # Process each timestep
    for t in tqdm.tqdm(range(num_steps)):
        # Prepare observation from GT data
        observation = prepare_observation_from_data(
            gt_images[t],
            gt_wrist_images[t],
            gt_proprio[t],
            cfg.flip_images
        )
        
        # Get model predictions
        predictions = get_model_predictions(
            cfg, model, planning_model, dataset_stats,
            observation, task_description, t
        )
        
        # Extract predictions
        pred_actions_chunk = predictions["actions"]  # List of actions (chunk_size elements)
        pred_future_images = predictions.get("future_image_predictions", {})
        pred_value = predictions.get("value_prediction", None)
        
        # Store predictions
        # get_action returns actions as a list of lists: [[action_0], [action_1], ..., [action_chunk_size-1]]
        if isinstance(pred_actions_chunk, list) and len(pred_actions_chunk) > 0:
            # Use first action in chunk
            first_action = pred_actions_chunk[0]
            if isinstance(first_action, list):
                pred_actions_all.append(np.array(first_action))
            else:
                pred_actions_all.append(np.array(first_action))
        else:
            pred_actions_all.append(np.zeros_like(gt_actions[0]))
        
        # Handle value prediction (already converted to float by get_action)
        if pred_value is not None:
            if isinstance(pred_value, torch.Tensor):
                pred_value = float(pred_value.item())
            elif isinstance(pred_value, np.ndarray):
                pred_value = float(pred_value.item() if pred_value.size == 1 else pred_value[0])
            else:
                pred_value = float(pred_value)
            pred_values_all.append(pred_value)
        else:
            pred_values_all.append(None)
        
        # Get GT value for this timestep
        gt_value = None
        if gt_returns is not None and t < len(gt_returns):
            gt_value = float(gt_returns[t])
        
        # Get predicted future images (use first frame of chunk if available)
        pred_primary = pred_future_images.get("future_image", None) if pred_future_images else None
        pred_wrist = pred_future_images.get("future_wrist_image", None) if pred_future_images else None
        
        # Convert to numpy arrays and handle batch dimension if present
        if pred_primary is not None:
            if isinstance(pred_primary, torch.Tensor):
                pred_primary = pred_primary.cpu().numpy()
            pred_primary = np.array(pred_primary)
            # Remove batch dimension if present (B, H, W, C) -> (H, W, C)
            if pred_primary.ndim == 4:
                pred_primary = pred_primary[0]
        
        if pred_wrist is not None:
            if isinstance(pred_wrist, torch.Tensor):
                pred_wrist = pred_wrist.cpu().numpy()
            pred_wrist = np.array(pred_wrist)
            # Remove batch dimension if present (B, H, W, C) -> (H, W, C)
            if pred_wrist.ndim == 4:
                pred_wrist = pred_wrist[0]
        
        # If no future prediction, use current GT image as placeholder
        if pred_primary is None:
            pred_primary = gt_images[min(t + 1, num_steps - 1)]
        if pred_wrist is None:
            pred_wrist = gt_wrist_images[min(t + 1, num_steps - 1)]
        
        # Flip predicted images back if they were flipped during model input
        # The model was trained with flipped images, so predictions are in flipped coordinate system
        # We need to flip them back to match GT images for visualization
        if cfg.flip_images:
            if pred_primary is not None:
                pred_primary = np.flipud(pred_primary)
            if pred_wrist is not None:
                pred_wrist = np.flipud(pred_wrist)
        
        # Convert predictions to numpy arrays
        pred_actions_array = np.array(pred_actions_all) if len(pred_actions_all) > 0 else np.zeros((t + 1, gt_actions.shape[1]))
        pred_actions_padded = np.zeros_like(gt_actions)
        pred_actions_padded[:len(pred_actions_array)] = pred_actions_array
        
        # Create side-by-side frame
        frame = create_side_by_side_frame(
            gt_primary=gt_images[t],
            gt_wrist=gt_wrist_images[t],
            pred_primary=pred_primary,
            pred_wrist=pred_wrist,
            timestep=t,
            gt_value=gt_value,
            pred_value=pred_values_all[-1] if pred_values_all else None,
            gt_actions=gt_actions,
            pred_actions=pred_actions_padded,
            current_timestep=t,
        )
        
        video_writer.append_data(frame)
    
    video_writer.close()
    log_message(f"Saved visualization video: {video_path}", log_file)
    
    return video_path


@draccus.wrap()
def eval_libero_visualize(cfg: PolicyEvalVisualizeConfig):
    """Main function to visualize model predictions on training dataset episodes."""

    # Set DETERMINISTIC environment variable if on deterministic mode
    assert not (cfg.deterministic and cfg.randomize_seed), (
        "Cannot enable both deterministic mode and randomize seed mode!"
    )
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

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
    
    # Initialize T5 text embeddings cache from dataset
    if t5_text_embeddings is not None:
        from cosmos_policy.experiments.robot.cosmos_utils import t5_text_embeddings_cache, DEVICE
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

    # Initialize planning model if needed
    planning_model = None
    if cfg.planning_model_ckpt_path != "":
        planning_model, _ = get_planning_model(cfg)
    else:
        planning_model = None

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg.model_family)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(
        cfg=cfg,
        task_identifier="visualize",
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=False,
        wandb_entity="",
        wandb_project="",
    )
    log_message(f"Visualization config: {cfg}", log_file)
    log_message(f"Visualizing {dataset.num_episodes} training episodes", log_file)

    # Determine which episodes to visualize
    if cfg.episode_indices is not None and cfg.episode_indices != "":
        # Parse comma-separated indices
        episode_indices = [int(x.strip()) for x in cfg.episode_indices.split(",")]
    elif cfg.max_episodes is not None:
        episode_indices = list(range(min(cfg.max_episodes, dataset.num_episodes)))
    else:
        episode_indices = list(range(dataset.num_episodes))
    
    log_message(f"Processing {len(episode_indices)} episodes: {episode_indices}", log_file)

    # Visualize each episode
    for episode_idx in tqdm.tqdm(episode_indices):
        if episode_idx >= dataset.num_episodes:
            log_message(f"Skipping episode {episode_idx} (out of range)", log_file)
            continue
        
        episode_data = dataset.data[episode_idx]
        video_path = visualize_episode_predictions(
            cfg,
            episode_idx,
            episode_data,
            model,
            planning_model,
            dataset_stats,
            resize_size,
            log_file,
        )
        log_message(f"Completed visualization for episode {episode_idx}: {video_path}", log_file)

    log_message("Visualization complete!", log_file)

    # Close log file
    if log_file:
        log_file.close()


if __name__ == "__main__":
    eval_libero_visualize()
