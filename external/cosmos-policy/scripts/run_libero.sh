uv run --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path /coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/logs/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_libero_one_demo_one_episode/checkpoints/iter_000050000 \
    --config_file cosmos_policy/config/config.py \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --use_proprio True \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 16 \
    --num_open_loop_steps 16 \
    --task_suite_name libero_10 \
    --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
    --randomize_seed False \
    --data_collection False \
    --available_gpus "0" \
    --seed 195 \
    --use_variance_scale False \
    --deterministic True \
    --run_id_note chkpt50K--5stepAct--seed195--deterministic \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --num_trials_per_task 1
  
# --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B