#!/bin/bash
# EgoWAM closed-loop eval in RoboTwin SAPIEN sim (plan §5).
# Usage (run from this dir):
#   bash eval.sh <task_name> <task_config> <ckpt_setting> <ckpt_path> [robot] [seed] [gpu_id]
# e.g.
#   bash eval.sh pick_diverse_bottles demo_clean rt_cross_pick_vae /path/last.ckpt arx-x5 0 0
# task_config: demo_clean (Easy) | demo_randomized (Hard). robot defaults to arx-x5.

policy_name=EgoWAM
task_name=${1}
task_config=${2}
ckpt_setting=${3}
ckpt_path=${4}
robot=${5:-aloha-agilex}   # bridge/model side name (selects the hpt_wm embodiment)
seed=${6:-0}
gpu_id=${7:-0}

# Map our robot name -> RoboTwin sim embodiment key (_embodiment_config.yml).
case "$robot" in
  arx-x5) sim_emb=ARX-X5;;
  ur5) sim_emb=ur5-wsg;;
  franka) sim_emb=franka-panda;;
  aloha-agilex) sim_emb=aloha-agilex;;
  piper) sim_emb=piper;;
  *) sim_emb="$robot";;
esac

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}  robot: ${robot} -> sim_emb: ${sim_emb}\033[0m"

cd ../..  # move to RoboTwin root

# Override the task_config embodiment so the rollout runs on ${robot} (plan §5 uses arx-x5).
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_path ${ckpt_path} \
    --robot ${robot} \
    --embodiment "['${sim_emb}']" \
    --seed ${seed} \
    --policy_name ${policy_name}
