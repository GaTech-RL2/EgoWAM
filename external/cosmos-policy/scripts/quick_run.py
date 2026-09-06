import pickle
import torch
from PIL import Image
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
    get_t5_embedding_from_cache,
)

# Instantiate config (see PolicyEvalConfig in cosmos_policy/experiments/robot/libero/run_libero_eval.py for definitions)
cfg = PolicyEvalConfig(
    config="cosmos_predict2_2b_480p_libero__inference_only",
    ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    config_file="cosmos_policy/config/config.py",
    dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
    t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
    use_wrist_image=True,
    use_proprio=True,
    normalize_proprio=True,
    unnormalize_actions=True,
    chunk_size=16,
    num_open_loop_steps=16,
    trained_with_image_aug=True,
    use_jpeg_compression=True,
    flip_images=True,  # Only for LIBERO; images render upside-down
    num_denoising_steps_action=5,
    num_denoising_steps_future_state=1,
    num_denoising_steps_value=1,
)
# Load dataset stats for action/proprio scaling
dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
# Initialize T5 text embeddings cache
init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
# Load model
model, cosmos_config = get_model(cfg)
# Load sample observation:
#   observation (dict): {
#     "primary_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "proprio": robot proprioceptive state,
#   }
with open("cosmos_policy/experiments/robot/libero/sample_libero_10_observation.pkl", "rb") as file:
    observation = pickle.load(file)
    task_description = "put both the alphabet soup and the tomato sauce in the basket"
# Generate robot actions, future state (proprio + images), and value
action_return_dict = get_action(
    cfg,
    model,
    dataset_stats,
    observation,
    task_description,
    num_denoising_steps_action=cfg.num_denoising_steps_action,
    generate_future_state_and_value_in_parallel=True,
)
# Print actions
print(f"Generated action chunk: {action_return_dict['actions']}")
# Save future image predictions (third-person image and wrist image)
img_path1, img_path2 = "future_image.png", "future_wrist_image.png"
Image.fromarray(action_return_dict['future_image_predictions']['future_image']).save(img_path1)
Image.fromarray(action_return_dict['future_image_predictions']['future_wrist_image']).save(img_path2)
print(f"Saved future image predictions to:\n\t{img_path1}\n\t{img_path2}")
# Print value
print(f"Generated value: {action_return_dict['value_prediction']}")


