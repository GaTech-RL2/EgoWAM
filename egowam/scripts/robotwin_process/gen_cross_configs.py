"""Generate RoboTwin cross-embodiment (cotrain shared-head) hydra configs.

Emits, for each robot-set × variant, a model config; and per task a data + train config.
Mirrors model/hpt_wm_cotrain.yaml but for RoboTwin robots (all bimanual, 14-dim
EE-Cartesian, extrinsics/intrinsics "robotwin"), reusing the exact single-embodiment blocks
(model/hpt_wm_rt_aloha_*.yaml) already validated. Plan §4.

Variants:
  vae = Cosmos VAE WM head (shared_future_image_latent: 4096 / lat_channels 16)
  rae = RAEVAEWrapper / DINOv2-768 (shared_future_image_latent: 196608 / lat_channels 768)
  hpt = no image WM: drop shared_future_image_latent head, cosmos_vae: null, aux=[future_ee_pose]
All variants drop future_flow. ot left unset (false). kinematics_solver removed.

Run: python -m egowam.scripts.robotwin_process.gen_cross_configs
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parents[2] / "hydra_configs"
ZARR_ROOT = (
    os.environ.get("EGOWAM_ROBOTWIN_ROOT", "/coc/flash7/scratch/egowm/robotwin")
    + "/zarrDataset"
)
# Track4World 3D point-flow sidecars (parallel tree to ZARR_ROOT). N=560 pts/ep.
FLOW_ROOT = os.environ.get(
    "EGOWAM_ROBOTWIN_FLOW_ROOT", "$EGOWAM_ROBOTWIN_ROOT_3d_flow_track4world"
)
FLOW_NUM_POINTS = 560

ROBOTS = {
    "aloha": {
        "folder": "aloha-agilex",
        "cls": "RobotwinAloha",
        "emb": "robotwin_aloha_bimanual",
    },
    "arx": {"folder": "arx-x5", "cls": "RobotwinArx", "emb": "robotwin_arx_bimanual"},
    "piper": {
        "folder": "piper",
        "cls": "RobotwinPiper",
        "emb": "robotwin_piper_bimanual",
    },
    "ur5": {"folder": "ur5", "cls": "RobotwinUr5", "emb": "robotwin_ur5_bimanual"},
    "franka": {
        "folder": "franka",
        "cls": "RobotwinFranka",
        "emb": "robotwin_franka_bimanual",
    },
}
# Cross model configs are named by their EMBODIMENT set (the robots cotrained), not by task.
# Only the pick task ships in the release, so only its embodiment set is generated.
# stack/hang used aloha_arx_franka_ur5; re-add both entries together to restore them,
# otherwise the generator would recreate model configs no shipped data config can pair with.
SETS = {
    "aloha_arx_piper_ur5": ["aloha", "arx", "piper", "ur5"],
}
TASKS = {
    "pick": "pick_diverse_bottles",
}
TASK_SET = {
    "pick": "aloha_arx_piper_ur5",
}
VARIANT = (
    "clean_50"  # default data variant (fast; override folder_path for randomized_500)
)

# ---- reusable spec fragments (identical to the validated single configs) --------------------
_XATTN = {
    "random_horizon_masking": False,
    "cross_attn": {
        "crossattn_latent": 16,
        "crossattn_heads": 8,
        "crossattn_dim_head": 64,
        "crossattn_modality_dropout": 0.1,
        "modality_embed_dim": 256,
    },
}


def _mlp_stem(input_dim):
    return {
        "_target_": "egowam.models.hpt_nets.MLPPolicyStem",
        "input_dim": input_dim,
        "output_dim": 256,
        "widths": [256],
        "specs": copy.deepcopy(_XATTN),
    }


def _resnet():
    return {"_target_": "egowam.models.hpt_nets.ResNet", "output_dim": 256}


def _fm_head(act_dim, act_seq, nblocks, infer_dims, pooling=None, decoder=None):
    h = {
        "_target_": "egowam.models.fm_policy.FMPolicy",
        "prediction_type": "v_prediction",
        "action_horizon": act_seq if act_seq > 1 else 1,
        "num_inference_steps": 50,
        "pooling": pooling,
        "time_dist": "beta",
        "infer_ac_dims": dict(infer_dims),
    }
    if decoder is None:
        h["model"] = {
            "_target_": "egowam.models.denoising_nets.CrossTransformer",
            "nblocks": nblocks,
            "cond_dim": 256,
            "hidden_dim": 128,
            "act_dim": act_dim,
            "act_seq": act_seq,
            "n_heads": 4,
            "dropout": 0.1,
            "mlp_layers": 4,
            "mlp_ratio": 4,
        }
    else:
        h["model"] = decoder
    return h


def _dit(lat_channels):
    return {
        "_target_": "egowam.models.dit_image_decoder.DiTImageDecoder",
        "lat_channels": lat_channels,
        "lat_h": 16,
        "lat_w": 16,
        "patch_size": 2,
        "hidden_dim": 384,
        "n_blocks": 6,
        "n_heads": 6,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "obs_dim": 256,
        "time_dim": 256,
    }


def _pointflow_head(embs, num_points=FLOW_NUM_POINTS, point_dim=3):
    """Shared 3D point-flow WM head (PointFMPolicy/PointFlowDecoder).

    Native k=45 trajectories are resampled to FLOW_CHUNK_LENGTH=100 by the
    embodiment transform, so action_horizon=100. infer_ac_dims enumerates every
    cotrained domain (shared head), each [100, N, point_dim].
    """
    return {
        "_target_": "egowam.models.fm_policy.PointFMPolicy",
        "prediction_type": "v_prediction",
        "action_horizon": 100,
        "num_inference_steps": 50,
        "pooling": None,
        "time_dist": "beta",
        "infer_ac_dims": {e: [100, num_points, point_dim] for e in embs},
        "model": {
            "_target_": "egowam.models.point_flow_decoder.PointFlowDecoder",
            "num_points": num_points,
            "point_dim": point_dim,
            "action_horizon": 100,
            "cond_dim": 256,
            "hidden_dim": 256,
            "n_blocks": 4,
            "n_heads": 4,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "time_dim": 256,
        },
    }


_COSMOS = {
    "_target_": "egowam.models.cosmos_vae.CosmosVAEWrapper",
    "vae_pth": "hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
    "chunk_duration": 1,
    "freeze": True,
}
_RAE = {
    "_target_": "egowam.models.rae_vae.RAEVAEWrapper",
    "rae_root": os.environ.get(
        "EGOWAM_RAE_ROOT", "/coc/flash7/mlin365/EgoVerse-main/external/RAE"
    ),
    "config_path": "/coc/flash7/mlin365/EgoVerse-main/external/RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml",
    "ckpt_path": "/coc/flash7/mlin365/RAE/ckpts/egowm_20260318_161148/stage1_egowm_20260318_161148/checkpoints/ep-0000008.pt",
    "use_stage2_features": True,
    "freeze": True,
}
_TRUNK = {
    "embed_dim": 256,
    "num_blocks": 16,
    "num_heads": 8,
    "token_postprocessing": "joint_token",
    "observation_horizon": 1,
    "action_horizon": 64,
    "future_horizon": 16,
    "no_trunk": False,
    "use_domain_embedding": True,
    "drop_path": 0.1,
    "weight_init_style": "pytorch",
}
_AUGS_TRAIN = {
    "_target_": "torchvision.transforms.Compose",
    "transforms": [
        {
            "_target_": "torchvision.transforms.ColorJitter",
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.05,
        },
        {
            "_target_": "torchvision.transforms.Normalize",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    ],
}
_AUGS_EVAL = {
    "_target_": "torchvision.transforms.Compose",
    "transforms": [
        {
            "_target_": "torchvision.transforms.Normalize",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    ],
}


def build_model(robots, variant):
    embs = [ROBOTS[r]["emb"] for r in robots]
    dims14 = {e: 14 for e in embs}
    is_hpt = variant == "hpt"
    is_flow = variant == "3dflow"  # 3D point-flow WM head instead of image latent
    if variant == "rae":
        img_dim, lat_ch, wm = 196608, 768, _RAE
    else:
        img_dim, lat_ch, wm = 4096, 16, _COSMOS

    if is_flow:
        aux = ["future_ee_pose", "future_flow_3d_points"]
        aux_weights = {"future_ee_pose": 1.0, "future_flow_3d_points": 1.0}
    elif is_hpt:
        aux = ["future_ee_pose"]
        aux_weights = {"future_ee_pose": 1.0}
    else:
        aux = ["future_ee_pose", "future_image_latent"]
        aux_weights = {"future_ee_pose": 1.0, "future_image_latent": 1.0}

    head_specs = {e: None for e in embs}
    head_specs["shared"] = _fm_head(14, 100, 6, dims14)
    head_specs["shared_future_ee_pose"] = _fm_head(14, 1, 4, dims14)
    if is_flow:
        head_specs["shared_future_flow_3d_points"] = _pointflow_head(embs)
    elif not is_hpt:
        h = _fm_head(
            None,
            1,
            None,
            {e: img_dim for e in embs},
            pooling="mean",
            decoder=_dit(lat_ch),
        )
        head_specs["shared_future_image_latent"] = h

    robomimic = {
        "_target_": "egowam.algo.hpt_wm.HPT",
        "data_schematic": "_${data.dataset.data_schematic}",
        "camera_transforms": {
            e: {
                "_target_": "egowam.utils.egowamUtils.CameraTransforms",
                "intrinsics_key": "robotwin",
                "extrinsics_key": "robotwin",
            }
            for e in embs
        },
        "diffusion": True,
        "6dof": True,
        "ac_keys": {e: "actions_cartesian" for e in embs},
        "shared_ac_key": "actions_cartesian",
        "auxiliary_ac_keys": {e: list(aux) for e in embs},
        "shared_auxiliary_ac_keys": list(aux),
        "auxiliary_loss_weights": aux_weights,
        "future_proprio_key": "future_ee_pose",
        "future_image_keys": [] if (is_hpt or is_flow) else ["future_front_img_1"],
        "wm_prediction_mode": "joint",
        "teacher_forcing_ratio": 1.0,
        "training_stage": 3,
        "use_shared_vae_stem": False,
        "vae_stem_projection": "linear",
        "cosmos_vae": None if (is_hpt or is_flow) else wm,
        "trunk": copy.deepcopy(_TRUNK),
        "multitask": False,
        "pretrained": False,
        "pretrained_checkpoint": "",
        "reverse_kl_samples": 8,
        "domains": list(embs),
        "shared_obs_keys": ["front_img_1"],
        "shared_stem_specs": {"front_img_1": _mlp_stem(256)},
        "stem_specs": {
            e: {
                "left_wrist_img": _mlp_stem(256),
                "right_wrist_img": _mlp_stem(256),
                "state_ee_pose": _mlp_stem(14),
            }
            for e in embs
        },
        "head_specs": head_specs,
        "encoder_specs": {
            "front_img_1": _resnet(),
            "right_wrist_img": _resnet(),
            "left_wrist_img": _resnet(),
        },
        "train_image_augs": copy.deepcopy(_AUGS_TRAIN),
        "eval_image_augs": copy.deepcopy(_AUGS_EVAL),
    }
    if not is_hpt and not is_flow:
        robomimic["future_image_latent_key"] = "future_image_latent"
        robomimic["future_image_resize"] = 128

    return {
        "_target_": "egowam.pl_utils.pl_model.ModelWrapper",
        "robomimic_model": robomimic,
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "_partial_": True,
            "lr": 1e-4,
            "weight_decay": 0.0001,
        },
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "_partial_": True,
            "T_max": 1400,
            "eta_min": 1e-5,
        },
    }


def _robot_data_variant(robot, data_variant):
    """Per-robot zarr folder suffix.

    'clean'    -> every robot uses clean_50 (default cross).
    'mixed500' -> the eval robot (aloha) keeps clean_50 to match the single-aloha
                  baseline; all OTHER embodiments use randomized_500 for richer
                  cross-embodiment transfer (the 'cross_500' data variant).
    """
    if data_variant == "mixed500" and robot != "aloha":
        return "randomized_500"
    return VARIANT  # clean_50


def _dataset_entry(robot, task, mode, flow=False, data_variant="clean"):
    r = ROBOTS[robot]
    fv = _robot_data_variant(robot, data_variant)
    folder = f"{ZARR_ROOT}/{TASKS[task]}/{r['folder']}_{fv}"
    key_map = {
        "_target_": f"egowam.rldb.embodiment.robotwin.{r['cls']}.get_keymap",
        "action_horizon": 45,
        "use_future": True,
    }
    transform_list = {
        "_target_": f"egowam.rldb.embodiment.robotwin.{r['cls']}.get_transform_list",
        "chunk_length": 100,
        "use_future": True,
    }
    if flow:
        # Track4World sidecars parallel to the main zarr (same task/robot leaf).
        # stats_key must equal the preflight's `{train_datasets-key}.future_flow_3d_points`.
        flow_root = f"{FLOW_ROOT}/{TASKS[task]}/{r['folder']}_{fv}"
        key_map.update(
            {
                "include_point_flow_3d": True,
                "point_flow_root": flow_root,
                "point_flow_movement_threshold_3d": 0.0,
                "point_flow_head_skip_frames": 0,
                "point_flow_tail_skip_frames": 0,
            }
        )
        transform_list.update(
            {
                "include_point_flow_3d": True,
                "point_flow_stats_key_3d": f"{r['emb']}.future_flow_3d_points",
            }
        )
    return {
        "_target_": "egowam.rldb.zarr.zarr_dataset_multi.MultiDataset._from_resolver",
        "resolver": {
            "_target_": "egowam.rldb.zarr.zarr_dataset_multi.LocalEpisodeResolver",
            "folder_path": folder,
            "key_map": key_map,
            "transform_list": transform_list,
        },
        "mode": mode,
    }


def build_data(task, flow=False, data_variant="clean"):
    robots = SETS[TASK_SET[task]]
    embs = [ROBOTS[r]["emb"] for r in robots]
    return {
        "_target_": "egowam.pl_utils.pl_data_utils.MultiDataModuleWrapper",
        "train_datasets": {
            ROBOTS[r]["emb"]: _dataset_entry(r, task, "train", flow, data_variant)
            for r in robots
        },
        "valid_datasets": {
            ROBOTS[r]["emb"]: _dataset_entry(r, task, "valid", flow, data_variant)
            for r in robots
        },
        "train_dataloader_params": {
            e: {"batch_size": 16, "num_workers": 8} for e in embs
        },
        "valid_dataloader_params": {
            e: {"batch_size": 16, "num_workers": 8} for e in embs
        },
    }


def _flow_schematic_keys():
    # The 4 Track4World point-flow sidecar keys (flow_keys bypass quantile norm;
    # NormalizeFlowDelta handles them). Names match _load_point_flow_target output.
    keys = {}
    base = "future_flow_3d_points"
    for suffix in ("", "__current_points", "__validity_mask", "__sampled_uv"):
        keys[base + suffix] = {"key_type": "flow_keys", "zarr_key": base + suffix}
    return keys


def _schematic_block(flow=False):
    block = {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.front_img_1",
        },
        "right_wrist_img": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.right_wrist_img",
        },
        "left_wrist_img": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.left_wrist_img",
        },
        "future_front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "future.observations.images.front_img_1",
        },
        "future_right_wrist_img": {
            "key_type": "camera_keys",
            "zarr_key": "future.observations.images.right_wrist_img",
        },
        "future_left_wrist_img": {
            "key_type": "camera_keys",
            "zarr_key": "future.observations.images.left_wrist_img",
        },
        "ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "observations.state.ee_pose",
        },
        "future_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "future.observations.state.ee_pose",
        },
        "actions_cartesian": {
            "key_type": "action_keys",
            "zarr_key": "actions_cartesian",
        },
        "embodiment": {"key_type": "metadata_keys", "zarr_key": "metadata.embodiment"},
    }
    if flow:
        block.update(_flow_schematic_keys())
    return block


def build_train(flow=False):
    """One task-agnostic cross train config.

    The task is selected purely by CLI overrides ``model=robotwin/hpt_wm_rt_cross_<set>_<variant>``
    and ``data=robotwin/<task>_cross``. The top-level ``data_schematic`` is a UNION over all five
    robots; trainHydra filters datasets to the model's ``domains`` (``_active_dataset_names_for_model``)
    and the shape/norm passes only touch loaded datasets, so the un-used embodiment block stays inert
    (the model never queries a domain outside its own set). Shapes are keyed by zarr_key and identical
    across robots, so the union costs nothing.
    """
    embs = [ROBOTS[r]["emb"] for r in ROBOTS]  # union: all 5 robots
    suffix = "_3dflow" if flow else ""  # data/name suffix
    model_variant = "3dflow" if flow else "vae"  # default model variant to compose
    return {
        "defaults": [
            {"model": f"robotwin/hpt_wm_rt_cross_aloha_arx_piper_ur5_{model_variant}"},
            {"paths": "default"},
            {"trainer": "ddp"},
            {"debug": None},
            {"logger": "wandb"},
            {"data": f"robotwin/pick_cross{suffix}"},
            {"callbacks": "checkpoints"},
            "_self_",
            {"flow3d": "none"},
            {"override hydra/launcher": "submitit"},
        ],
        "name": f"rt_cross{suffix}",
        "description": f"rt_cross{suffix}",
        "ckpt_path": None,
        "train": True,
        "eval": False,
        "validate": False,
        "eval_class": {
            "_target_": "egowam.scripts.evaluation.Eve",
            "mode": "real",
            "arm": "both",
            "eval_path": "./logs/eval/${name}_${now:%Y-%m-%d_%H-%M-%S}",
        },
        "hydra": {
            "run": {"dir": "./logs/${name}/${description}_${now:%Y-%m-%d_%H-%M-%S}"},
            "sweep": {"dir": "./logs/${name}/${description}_${now:%Y-%m-%d_%H-%M-%S}"},
        },
        "launch_params": {"gpus_per_node": 1, "nodes": 1},
        "trainer": {"limit_val_batches": 300},
        "data_schematic": {
            "_target_": "egowam.rldb.zarr.utils.DataSchematic",
            "norm_mode": "quantile",
            "schematic_dict": {e: _schematic_block(flow) for e in embs},
            "viz_img_key": {e: "front_img_1" for e in embs},
        },
        "seed": 42,
        "norm_stat_fraction": 1.0,
    }


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    type(None), lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", "null")
)


def _dump(obj, path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(
            obj,
            f,
            Dumper=_Dumper,
            sort_keys=False,
            default_flow_style=False,
            width=120,
            allow_unicode=True,
        )
    print(f"wrote {path}")


def main():
    for s in SETS:
        for v in ("vae", "rae", "hpt", "3dflow"):
            hdr = f"# AUTO-GENERATED by gen_cross_configs.py — RoboTwin cross-embodiment ({s} set, {v}).\n"
            _dump(
                build_model(SETS[s], v),
                CFG / "model" / "robotwin" / f"hpt_wm_rt_cross_{s}_{v}.yaml",
                hdr,
            )
    for t in TASKS:
        _dump(
            build_data(t),
            CFG / "data" / "robotwin" / f"{t}_cross.yaml",
            f"# AUTO-GENERATED — RoboTwin cross-embodiment data ({t}: {SETS[TASK_SET[t]]}); folder_path per-robot knob.\n",
        )
        _dump(
            build_data(t, flow=True),
            CFG / "data" / "robotwin" / f"{t}_cross_3dflow.yaml",
            f"# AUTO-GENERATED — RoboTwin cross-embodiment 3D-flow data ({t}: {SETS[TASK_SET[t]]}); flow sidecars parallel to zarr.\n",
        )
        # The cross_500 variant (aloha clean_50 + other robots randomized_500) is not part
        # of the release; build_data(..., data_variant="mixed500") still supports it if you
        # want to regenerate those configs locally.
    # ONE task-agnostic train config (union schematic); select task via model=/data= overrides:
    #   pick  -> model=robotwin/hpt_wm_rt_cross_aloha_arx_piper_ur5_<v>  data=robotwin/pick_cross
    _dump(
        build_train(),
        CFG / "train_zarr_wm_robotwin.yaml",
        "# AUTO-GENERATED — RoboTwin cross train (task-agnostic; union schematic).\n"
        "# Select the task by overriding both model= and data= (see gen_cross_configs.main).\n",
    )
    # The flow variant is not a second entrypoint: its extra schematic keys ship as the
    # `flow3d=point3d_rt` config group, so one train config covers both. A flow key on a
    # dataset with no point_flow_root raises KeyError, hence the keys stay conditional.
    flow_sd = build_train(flow=True)["data_schematic"]["schematic_dict"]
    _dump(
        {
            emb: {k: v for k, v in keys.items() if "flow" in k}
            for emb, keys in flow_sd.items()
        },
        CFG / "flow3d" / "point3d_rt.yaml",
        "# @package data_schematic.schematic_dict\n"
        "#\n"
        "# AUTO-GENERATED — Track4World point-flow targets for the RoboTwin embodiments.\n"
        "# Select with flow3d=point3d_rt on train_zarr_wm_robotwin.\n",
    )


if __name__ == "__main__":
    main()
