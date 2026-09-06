"""RoboTwin deploy shim for EgoVerse ``hpt_wm`` world-model policies (plan §5).

RoboTwin's ``script/eval_policy.py`` imports this module (as ``EgoWAM`` on ``sys.path=./policy``)
and drives it via ``get_model`` / ``eval`` / ``reset_model``. All model logic lives in the EgoVerse
bridge ``egowam.scripts.robotwin_process.egowam_deploy.EgoWAMPolicy``; this file only:
  * maps a RoboTwin observation dict -> the bridge's obs dict (``encode_obs``),
  * builds the policy from ``usr_args`` (``get_model``),
  * runs a receding-horizon chunk via ``TASK_ENV.take_action(a, action_type="ee")`` (``eval``),
  * resets the per-episode step counter (``reset_model``).

Env requirement: this process must have BOTH RoboTwin's SAPIEN stack (sapien, mplib) AND EgoVerse's
torch/egowam importable. See egowam/scripts/robotwin_process/run_sim_eval.py for the launcher and
the env note (emimic2 lacks sapien/mplib today).
"""

import os
import sys
from pathlib import Path

import numpy as np

# Make egowam importable when eval runs from the RoboTwin root.
# deploy_policy.py -> EgoWAM -> policy -> RoboTwin -> external -> <EgoVerse repo root>
_EGOVERSE_ROOT = os.environ.get("EGOVERSE_ROOT") or str(Path(__file__).resolve().parents[4])
if _EGOVERSE_ROOT not in sys.path:
    sys.path.insert(0, _EGOVERSE_ROOT)

import cv2  # noqa: E402

from egowam.scripts.robotwin_process.egowam_deploy import EgoWAMPolicy  # noqa: E402

# Training resolution (robotwin_to_zarr kept native 240x320). Resize eval frames to match so the
# visual encoders see the training distribution regardless of the eval head_camera_type.
_TRAIN_H, _TRAIN_W = 240, 320

# Optional deploy-time visual-domain correction (see agent/robotwin_crossembodiment_experiment.md
# §7h): this checkout's sim renders a DIFFERENT bowl asset than the training data was collected with
# (asset-version drift; e.g. teal vs olive base3.glb), so eval images are OOD -> the diffusion policy
# goes bimodal and the closed loop collapses. Setting EGOWAM_COLOR_MATCH_STATS=<train_img_stats.json>
# (per-cam per-channel RGB mean/std from the training zarr) remaps each eval frame's per-channel
# mean/std toward the training distribution, which removes the collapse (arm tracks the demo). NB:
# this corrects COLOR only, not asset GEOMETRY — sufficient to un-break the rollout, not necessarily
# to complete precision grasps. Off by default (no env var) => behaviour unchanged.
_CM_STATS = None
_cm_path = os.environ.get("EGOWAM_COLOR_MATCH_STATS")
if _cm_path and os.path.exists(_cm_path):
    import json as _json

    _CM_STATS = {
        k: (np.asarray(v["mean"], np.float32), np.asarray(v["std"], np.float32) + 1e-6)
        for k, v in _json.load(open(_cm_path)).items()
    }
    print(f"[EgoWAM.encode_obs] color-match ON <- {_cm_path}", flush=True)


def _resize_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.shape[0] == _TRAIN_H and img.shape[1] == _TRAIN_W:
        return img
    # cv2.resize takes (W, H); INTER_AREA for downscaling. RoboTwin RGB is already RGB uint8.
    return cv2.resize(img, (_TRAIN_W, _TRAIN_H), interpolation=cv2.INTER_AREA)


def _color_match(img: np.ndarray, key: str) -> np.ndarray:
    """Remap img's per-channel mean/std toward the training distribution for `key` (no-op if the
    stats file / this cam key is absent)."""
    if _CM_STATS is None or key not in _CM_STATS:
        return img
    im = np.asarray(img, np.float32)
    mu = im.reshape(-1, 3).mean(0)
    sd = im.reshape(-1, 3).std(0) + 1e-6
    rmu, rsd = _CM_STATS[key]
    return np.clip((im - mu) / sd * rsd + rmu, 0, 255).astype(np.uint8)


def encode_obs(observation):
    """RoboTwin obs dict -> EgoWAMPolicy obs dict.

    Camera map matches robotwin_to_zarr: head->front_img_1, left->left_wrist, right->right_wrist.
    endpose is xyz+wxyz world frame; gripper is normalized [0,1] (1.0=open) — no inversion.
    Images are optionally color-matched toward the training distribution (see _CM_STATS above).
    """
    o = observation["observation"]
    ep = observation["endpose"]
    return {
        "front_img_1": _color_match(_resize_rgb(o["head_camera"]["rgb"]), "front_img_1"),
        "left_wrist_img": _color_match(_resize_rgb(o["left_camera"]["rgb"]), "left_wrist_img"),
        "right_wrist_img": _color_match(_resize_rgb(o["right_camera"]["rgb"]), "right_wrist_img"),
        "left_endpose": np.asarray(ep["left_endpose"], dtype=np.float64).reshape(7),
        "right_endpose": np.asarray(ep["right_endpose"], dtype=np.float64).reshape(7),
        "left_gripper": float(np.asarray(ep["left_gripper"]).reshape(-1)[0]),
        "right_gripper": float(np.asarray(ep["right_gripper"]).reshape(-1)[0]),
    }


def get_model(usr_args):
    ckpt_path = usr_args.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("EgoWAM deploy needs --ckpt_path (path to an hpt_wm last.ckpt)")
    return EgoWAMPolicy(
        ckpt_path=ckpt_path,
        robot=usr_args.get("robot", "arx-x5"),
        num_inference_steps=int(usr_args.get("num_inference_steps", 10)),
        action_horizon=int(usr_args.get("action_horizon", 45)),
        chunk_length=int(usr_args.get("chunk_length", 100)),
        exec_horizon=int(usr_args.get("exec_horizon", 30)),
    )


def _install_ik_counter(TASK_ENV, model):
    """Wrap robot.{left,right}_plan_path so silent IK failures are printed as parseable markers.

    RoboTwin's take_action('ee') falls back to a fixed 50-step interpolation when plan_path
    status != 'Success' (no exception raised). run_sim_eval greps these markers to separate
    IK-infeasibility from genuine policy failures (plan §5 / §7.4)."""
    if getattr(model, "_ik_patched", False):
        return
    robot = getattr(TASK_ENV, "robot", None)
    if robot is None:
        return

    def _wrap(orig, side):
        def wrapped(*a, **k):
            res = orig(*a, **k)
            try:
                if isinstance(res, dict) and res.get("status") != "Success":
                    print(f"EGOWAM_IK_FAIL:{side}", flush=True)
            except Exception:
                pass
            return res

        return wrapped

    for side in ("left", "right"):
        name = f"{side}_plan_path"
        if hasattr(robot, name):
            setattr(robot, name, _wrap(getattr(robot, name), side))
    model._ik_patched = True


def eval(TASK_ENV, model, observation):
    """One receding-horizon prediction: encode obs -> predict exec_horizon waypoints -> execute.

    The outer loop in eval_policy.py re-invokes this until success or step_lim, so each call
    replans from the freshest observation (observation_horizon=1)."""
    _install_ik_counter(TASK_ENV, model)
    obs = encode_obs(observation)
    actions = model.get_action(obs)  # (exec_horizon, 16) ee: [xyz+quat_wxyz+grip]x2
    for action in actions:
        TASK_ENV.take_action(action, action_type="ee")
        if TASK_ENV.eval_success:
            break
        observation = TASK_ENV.get_obs()  # refresh now_obs (drives the eval video)


def reset_model(model):
    model.reset()
