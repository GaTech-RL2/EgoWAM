#!/usr/bin/env python3
"""Held-out success-rate eval harness for the EE-Cartesian ACT (and DP) baselines.

Mirrors the EgoWAM 100-seed sweep (``sim_smoke.run_eval_seeds``): the SAME deterministic
held-out seed iteration (``seed_base=100000, +1, ...``, disjoint from the training seeds),
the SAME ``UnStableError`` skip, the SAME per-seed resumable JSON (written after every seed),
the SAME production ``step_lim`` budget, and the SAME ``_eval_summary`` — but driving a RoboTwin
EE-Cartesian policy instead of ``EgoWAMPolicy``:

    obs -> EE-14 world state + 3 cam images -> ACT model -> (chunk,14) world EE-Cartesian chunk
        -> world14_to_ee16 -> take first ``exec_horizon`` -> execute_ee_dense (mink IK + vcap).

The ACT model is loaded via RoboTwin's ``ACT`` class and queried DIRECTLY (``model.policy(...)``)
rather than through ``ACT.get_action`` — get_action stacks cameras in ``[head, left, right]`` order,
which is SWAPPED relative to the training order ``camera_names = [cam_high(head), cam_right_wrist,
cam_left_wrist]`` (i.e. ``[head, right, left]``). We reproduce the training order exactly.

Image channel order: RoboTwin collection stores the live sapien RGB frame via ``cv2.imencode`` and
training reads it back with ``cv2.imdecode`` (a channel-order-preserving round trip), so the model
trained on RGB-ordered frames == the live sim obs ``["rgb"]``. NO BGR swap (matches process_data_ee
/ RoboTwin's own ACT deploy). Frames are resized to 640x480 (cv2 INTER_LINEAR), CHW, /255.

DP (Diffusion Policy) EE-Cartesian path (--policy dp): loads RoboTwin's ``DP`` (DiffusionUnetImage-
Policy, robot_dp_14) via hydra+dill under emimic2_sim (a diffusers-0.37 typing shim re-injects the
``Union``/``Optional`` names diffusion_policy expects from ``diffusers.optimization``). The DP-EE ckpt
conditions ONLY on ``head_cam`` (native 240x320) + ``agent_pos`` = ``endpose_to_world14(le,re,lg,rg)``
(== the training zarr ``state``; ``left_cam``/``right_cam`` are commented out of shape_meta but still
required as deque keys by RoboTwin's DPRunner, so we populate & they're ignored). It emits an
``n_action_steps``(6) x 14 WORLD EE-Cartesian chunk fed to ``world14_to_ee16`` exactly like ACT, and
keeps an ``n_obs_steps``(3) history deque updated after each executed chunk (RoboTwin DP-deploy order).

Run on a GPU node (sapien Vulkan + the mink executor). Examples:
  cd external/RoboTwin && \
  TMPDIR=$EGOWAM_ROBOTWIN_ROOT/_sim_rollout/_eval50/_logs \
  ../../emimic2_sim/bin/python -m egowam.scripts.robotwin_process.eval_baseline_ee \
      --policy act --task stack_bowls_three \
      --ckpt-dir $EGOWAM_ROBOTWIN_ROOT/_baselines_ee/ACT/_ckpt/act-stack_bowls_three/demo_clean-50 \
      --n-target 100

  cd external/RoboTwin && \
  TMPDIR=$EGOWAM_ROBOTWIN_ROOT/_sim_rollout/_eval50/_logs \
  ../../emimic2_sim/bin/python -m egowam.scripts.robotwin_process.eval_baseline_ee \
      --policy dp --task hanging_mug \
      --ckpt-dir policy/DP/checkpoints/hanging_mug-demo_clean-50-0/50.ckpt \
      --results-path $EGOWAM_ROBOTWIN_ROOT/_baselines_ee/_eval/dp_hanging_mug.json \
      --n-target 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from egowam.scripts.robotwin_process.egowam_deploy import (
    endpose_to_world14,
    world14_to_ee16,
)
from egowam.scripts.robotwin_process.sim_smoke import (
    _class_decorator,
    _eval_summary,
    _load_args,
    _setup_robotwin_path,
    execute_ee_dense,
    patch_mink_ik,
)

RT_ROOT = Path(__file__).resolve().parents[3] / "external" / "RoboTwin"

# Training image geometry (process_data_ee.py): head/right/left cameras resized to (W=640,H=480).
_IMG_W, _IMG_H = 640, 480
# Training camera stack order (constants.py SIM_TASK_CONFIGS[...]["camera_names"]).
#   cam_high        = head_camera
#   cam_right_wrist = right_camera
#   cam_left_wrist  = left_camera
_CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
_CAMERA_OBS_KEYS = ["head_camera", "right_camera", "left_camera"]


# ----------------------------------------------------------------------------------------------
# ACT EE-Cartesian policy wrapper
# ----------------------------------------------------------------------------------------------
class ACTEEPolicy:
    """RoboTwin ``ACT`` (EE-Cartesian, 14-dim) -> (chunk_size, 14) WORLD EE-Cartesian action chunk.

    Bypasses ``ACT.get_action`` (swapped camera order) and calls ``model.policy`` directly with the
    training-order camera stack and the EE-14 world state as ``qpos``."""

    def __init__(self, ckpt_dir: str, device: str = "cuda:0", chunk_size: int = 50):
        from argparse import Namespace

        import cv2  # noqa: F401
        import torch

        # RoboTwin's ACT package: `import act_policy` + `from detr.main import ...`.
        act_dir = str(RT_ROOT / "policy" / "ACT")
        if act_dir not in sys.path:
            sys.path.insert(0, act_dir)
        from act_policy import ACT

        # usr_args == RoboTwin's deploy args (deploy_policy.yml / bring-up notes) with the two extra
        # keys the Namespace path of build_ACT_model_and_optimizer needs (parser defaults are NOT
        # applied when a RoboTwin_Config Namespace is passed):
        #   action_dim  -> ACT.__init__ sets self.state_dim = RoboTwin_Config.action_dim
        #   weight_decay-> optimizer AdamW(weight_decay=args.weight_decay)
        usr_args = dict(
            ckpt_dir=ckpt_dir,
            policy_class="ACT",
            kl_weight=10,
            chunk_size=chunk_size,
            hidden_dim=512,
            dim_feedforward=3200,
            temporal_agg=False,
            device=device,
            state_dim=14,
            action_dim=14,
            camera_names=list(_CAMERA_NAMES),
            lr=1e-5,
            lr_backbone=1e-5,
            weight_decay=1e-4,
            seed=0,
            backbone="resnet18",
            enc_layers=4,
            dec_layers=7,
            nheads=8,
            num_queries=chunk_size,
            position_embedding="sine",
            masks=False,
            dilation=False,
            dropout=0.1,
            pre_norm=False,
        )
        self.model = ACT(
            usr_args, Namespace(**usr_args)
        )  # auto-loads policy_last.ckpt + stats
        if self.model.stats is None:
            raise SystemExit(
                f"[eval_baseline_ee] no dataset_stats.pkl in {ckpt_dir} — cannot normalize"
            )
        self._cv2 = cv2
        self._torch = torch
        self.device = device

    def reset(self):
        # Direct model.policy() calls are stateless (latent = zeros at inference); nothing to reset.
        self.model.t = 0

    def post_exec(self, TASK_ENV):
        # ACT is history-free (single-frame qpos + images per query) -> no post-execution obs update.
        pass

    def _img(self, rgb: np.ndarray) -> np.ndarray:
        """sim obs rgb (H,W,3 uint8 RGB) -> (3,H',W') float32 [0,1], resized to 640x480 (CHW)."""
        r = self._cv2.resize(
            np.ascontiguousarray(rgb),
            (_IMG_W, _IMG_H),
            interpolation=self._cv2.INTER_LINEAR,
        )
        return np.moveaxis(r, -1, 0).astype(np.float32) / 255.0

    def predict_chunk_ee14(self, obs: dict) -> np.ndarray:
        """RoboTwin obs (get_obs) -> (chunk_size, 14) WORLD EE-Cartesian [L xyz+ypr+grip, R ...]."""
        torch = self._torch
        endp = obs["endpose"]
        le = np.asarray(endp["left_endpose"], np.float64).reshape(7)  # xyz+wxyz world
        re = np.asarray(endp["right_endpose"], np.float64).reshape(7)
        lg = float(np.asarray(endp["left_gripper"]).reshape(-1)[0])
        rg = float(np.asarray(endp["right_gripper"]).reshape(-1)[0])
        ee14 = endpose_to_world14(le, re, lg, rg)  # (14,) world xyz+ypr+grip x2

        imgs = np.stack(
            [self._img(obs["observation"][k]["rgb"]) for k in _CAMERA_OBS_KEYS], axis=0
        )  # (3, 3, 480, 640) in [head, right, left] training order

        qpos = np.asarray(self.model.pre_process(ee14), np.float32)  # normalize
        qpos_t = torch.from_numpy(qpos)[None].float().to(self.device)  # (1,14)
        img_t = torch.from_numpy(imgs)[None].float().to(self.device)  # (1,3,3,480,640)
        with torch.no_grad():
            a_hat = self.model.policy(qpos_t, img_t)  # (1, chunk_size, 14) normalized
        chunk = self.model.post_process(a_hat.detach().cpu().numpy())[
            0
        ]  # (chunk_size,14) world EE
        return np.asarray(chunk, np.float64)


# ----------------------------------------------------------------------------------------------
# DP (Diffusion Policy) EE-Cartesian policy wrapper
# ----------------------------------------------------------------------------------------------
class DPEEPolicy:
    """RoboTwin ``DP`` (DiffusionUnetImagePolicy) EE-Cartesian variant -> (n_action_steps, 14) WORLD
    EE-Cartesian action chunk.

    The DP-EE ckpt was trained (policy/DP/process_data_ee.py + robot_dp_14.yaml) on a 14-dim EE state
    ``agent_pos = endpose_to_world14(le, re, lg, rg)`` (== training zarr ``state``) and the native
    240x320 ``head_camera`` frame; ``left_cam``/``right_cam`` are commented out of the shape_meta /
    dataset / obs_encoder, so ONLY ``head_cam`` + ``agent_pos`` condition the model. RoboTwin's
    ``DPRunner.get_action`` nonetheless hard-codes reading ``left_cam``/``right_cam`` off the stacked
    obs deque, so we still populate those keys (the encoder ignores them). Images are RGB (no BGR
    swap — RoboTwin's imencode/imdecode round trip preserves channel order == the live sim rgb) and
    fed at native resolution (no resize; the encoder asserts the 240x320 training shape). The DP
    policy normalizes ``agent_pos``/``head_cam`` internally, so we pass raw [0,1] float32 images and
    the raw EE-14 state.

    Closed-loop obs history mirrors RoboTwin's DP deploy (deploy_policy.py ``eval`` + DPRunner):
    ``reset()`` clears the deque; ``predict_chunk_ee14`` builds the EE obs and calls ``get_action``
    (which appends it and returns the last-``n_obs_steps`` stacked history, padded on the first call);
    ``post_exec`` appends the post-execution obs (``update_obs``) exactly as RoboTwin does after taking
    the chunk's actions -> the 3-step history stays correct across replans."""

    def __init__(
        self,
        ckpt: str,
        device: str = "cuda:0",
        n_obs_steps: int = 3,
        n_action_steps: int = 6,
    ):
        from typing import Optional as _Optional
        from typing import Union as _Union

        import cv2  # noqa: F401

        # diffusers 0.37 (emimic2_sim) dropped the typing re-exports that
        # diffusion_policy/model/common/lr_scheduler.py imports from diffusers.optimization
        # (`from diffusers.optimization import Union, ..., Optional, ...`). Inject them from typing
        # into the module namespace BEFORE importing anything from diffusion_policy that pulls it in.
        import diffusers.optimization as _dopt

        for _name, _val in (("Union", _Union), ("Optional", _Optional)):
            if not hasattr(_dopt, _name):
                setattr(_dopt, _name, _val)

        # diffusion_policy is a LOCAL package under policy/DP; add that dir so `from dp_model import DP`
        # (and dp_model's own `from diffusion_policy...` imports) resolve. RobotWorkspace.__init__ only
        # builds model+ema+optimizer (the dataset/zarr is instantiated lazily in run()), so no cwd or
        # training-zarr dependency at load time.
        dp_dir = str(RT_ROOT / "policy" / "DP")
        if dp_dir not in sys.path:
            sys.path.insert(0, dp_dir)

        ckpt_file = self._resolve_ckpt(ckpt)
        from dp_model import DP

        self.model = DP(
            ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps
        )
        self._cv2 = cv2
        self.device = device
        self.n_action_steps = n_action_steps
        print(
            f"[eval_baseline_ee] DP loaded: {ckpt_file} "
            f"(n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps})",
            flush=True,
        )

    @staticmethod
    def _resolve_ckpt(ckpt: str) -> str:
        """DP takes a ckpt FILE (e.g. .../50.ckpt). If a directory is given, pick the highest-numbered
        ``<epoch>.ckpt`` in it (e.g. 50.ckpt over 25.ckpt) for convenience."""
        if os.path.isfile(ckpt):
            return ckpt
        if os.path.isdir(ckpt):
            cands = []
            for f in os.listdir(ckpt):
                if f.endswith(".ckpt"):
                    stem = f[:-5]
                    cands.append((int(stem) if stem.isdigit() else -1, f))
            if not cands:
                raise SystemExit(f"[eval_baseline_ee] no *.ckpt in DP ckpt dir {ckpt}")
            return os.path.join(ckpt, max(cands)[1])
        raise SystemExit(f"[eval_baseline_ee] DP ckpt path not found: {ckpt}")

    def _chw(self, rgb: np.ndarray) -> np.ndarray:
        """sim obs rgb (H,W,3 uint8 RGB) -> (3,H,W) float32 [0,1]. NO BGR swap, native resolution
        (resized to the 240x320 training shape only if the sim ever hands back a different size)."""
        r = np.ascontiguousarray(rgb)
        if r.shape[:2] != (240, 320):
            r = self._cv2.resize(
                r, (320, 240), interpolation=self._cv2.INTER_LINEAR
            )  # (W,H)
        return np.moveaxis(r, -1, 0).astype(np.float32) / 255.0

    def _encode(self, obs: dict) -> dict:
        """RoboTwin get_obs -> DP obs dict {head_cam, left_cam, right_cam, agent_pos} (RoboTwin's
        DPRunner.get_action requires all three cam keys; only head_cam is used by the encoder)."""
        o = obs["observation"]
        endp = obs["endpose"]
        le = np.asarray(endp["left_endpose"], np.float64).reshape(7)
        re = np.asarray(endp["right_endpose"], np.float64).reshape(7)
        lg = float(np.asarray(endp["left_gripper"]).reshape(-1)[0])
        rg = float(np.asarray(endp["right_gripper"]).reshape(-1)[0])
        agent_pos = endpose_to_world14(le, re, lg, rg).astype(
            np.float32
        )  # (14,) == training state
        return dict(
            head_cam=self._chw(o["head_camera"]["rgb"]),
            left_cam=self._chw(o["left_camera"]["rgb"]),
            right_cam=self._chw(o["right_camera"]["rgb"]),
            agent_pos=agent_pos,
        )

    def reset(self):
        self.model.reset_obs()

    def predict_chunk_ee14(self, obs: dict) -> np.ndarray:
        """RoboTwin obs -> (n_action_steps, 14) WORLD EE-Cartesian chunk. ``get_action`` appends the
        obs to the deque and stacks the last n_obs_steps (padding on the first call)."""
        actions = self.model.get_action(
            self._encode(obs)
        )  # (n_action_steps, 14) EE-14 world
        return np.asarray(actions, np.float64)

    def post_exec(self, TASK_ENV):
        # Mirror RoboTwin's DP deploy: append the post-execution obs to the history deque.
        self.model.update_obs(self._encode(TASK_ENV.get_obs()))


def _load_policy(
    policy: str,
    ckpt_dir: str,
    chunk_size: int,
    n_obs_steps: int = 3,
    n_action_steps: int = 6,
):
    if policy == "act":
        return ACTEEPolicy(ckpt_dir, chunk_size=chunk_size)
    if policy == "dp":
        # RoboTwin DP (policy/DP/dp_model.py): loads a diffusion_policy workspace ckpt via hydra+dill
        # under emimic2_sim (diffusers-shim'd) and returns an n_action_steps x 14 WORLD EE-Cartesian
        # chunk that feeds world14_to_ee16 exactly like ACT. `ckpt_dir` here is the DP ckpt FILE
        # (e.g. .../50.ckpt); a directory is also accepted (highest-epoch .ckpt picked).
        return DPEEPolicy(
            ckpt_dir, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps
        )
    raise SystemExit(f"[eval_baseline_ee] unknown policy '{policy}'")


# ----------------------------------------------------------------------------------------------
# Held-out eval loop (mirrors sim_smoke.run_eval_seeds)
# ----------------------------------------------------------------------------------------------
def run_eval(
    policy: str,
    task_name: str,
    task_config: str,
    robot: str,
    ckpt_dir: str,
    results_path: str,
    n_target: int = 100,
    seed_base: int = 100000,
    max_candidates: int = 600,
    exec_horizon: int = 30,
    steps_per_wp: int = 15,
    max_joint_vel: float = 1.0,
    chunk_size: int = 50,
    n_obs_steps: int = 3,
    n_action_steps: int = 6,
):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results = []
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                results = json.load(f)
        except Exception:
            results = []
    done = {r["seed"] for r in results}

    variant_tag = f"{policy}_ee"

    def n_valid():
        return sum(1 for r in results if r["status"] in ("success", "fail", "error"))

    def n_succ():
        return sum(1 for r in results if r["status"] == "success")

    def save():
        tmp = results_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1)
        os.replace(tmp, results_path)

    print(
        f"[eval] {variant_tag} {task_name}: resume with {n_valid()}/{n_target} valid "
        f"({n_succ()} success) already logged -> {results_path}",
        flush=True,
    )
    if n_valid() >= n_target:
        _eval_summary(variant_tag, task_name, results, n_target)
        return

    model = _load_policy(
        policy,
        ckpt_dir,
        chunk_size,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )
    args = _load_args(task_name, task_config, robot)
    args.setdefault("data_type", {})
    args["data_type"]["endpose"] = True
    TASK_ENV = _class_decorator(task_name)

    seed = seed_base
    tried = 0
    while n_valid() < n_target and tried < max_candidates:
        s = seed
        seed += 1
        tried += 1
        if s in done:
            continue
        # ---- setup (deterministic per seed; unstable scenes skipped, same for every variant) ----
        try:
            TASK_ENV.setup_demo(now_ep_num=0, seed=s, is_test=True, **args)
        except Exception as e:
            results.append({"seed": s, "status": "skip_setup", "err": type(e).__name__})
            done.add(s)
            save()
            try:
                TASK_ENV.close_env()
            except Exception:
                pass
            continue
        # ---- closed-loop rollout (mink IK + vcap dense executor) to the production step_lim ----
        restore_ik = patch_mink_ik(TASK_ENV, verbose=False)
        model.reset()
        rec = {"seed": s}
        try:
            replans, ikf = 0, 0
            while (
                TASK_ENV.take_action_cnt < TASK_ENV.step_lim
                and not TASK_ENV.eval_success
            ):
                obs = TASK_ENV.get_obs()
                ee14 = model.predict_chunk_ee14(
                    obs
                )  # (H,14) world EE (ACT: chunk_size; DP: n_action_steps)
                ee16 = world14_to_ee16(ee14)  # (H,16) [xyz+quat_wxyz+grip]x2
                st = execute_ee_dense(
                    TASK_ENV,
                    ee16[
                        :exec_horizon
                    ],  # DP: H=n_action_steps(6) <= exec_horizon -> whole chunk executed
                    steps_per_wp=steps_per_wp,
                    max_joint_vel=max_joint_vel,
                )
                ikf += st["ik_fails"]
                replans += 1
                # DP keeps an n_obs_steps history deque -> append the post-execution obs (RoboTwin's
                # DP deploy does update_obs after taking the chunk). No-op for the history-free ACT.
                model.post_exec(TASK_ENV)
            ok = bool(TASK_ENV.eval_success or TASK_ENV.check_success())
            rec.update(
                status="success" if ok else "fail",
                steps=int(TASK_ENV.take_action_cnt),
                replans=replans,
                ik_fails=ikf,
            )
        except Exception as e:
            traceback.print_exc()
            rec.update(status="error", err=type(e).__name__)
        finally:
            restore_ik()
            try:
                TASK_ENV.close_env()
            except Exception:
                pass
        results.append(rec)
        done.add(s)
        save()
        print(
            f"[eval] {variant_tag} {task_name} seed={s} -> {rec['status']}  "
            f"valid={n_valid()}/{n_target} success={n_succ()}",
            flush=True,
        )
    _eval_summary(variant_tag, task_name, results, n_target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=["act", "dp"], default="act")
    p.add_argument("--task", required=True, help="e.g. stack_bowls_three | hanging_mug")
    p.add_argument("--task-config", default="demo_clean")
    p.add_argument("--robot", default="aloha-agilex")
    p.add_argument(
        "--ckpt-dir",
        required=True,
        help="ACT: dir with policy_last.ckpt + dataset_stats.pkl. "
        "DP: the ckpt FILE (e.g. .../hanging_mug-demo_clean-50-0/50.ckpt); a dir is also accepted "
        "(highest-epoch *.ckpt picked).",
    )
    p.add_argument(
        "--results-path",
        default=None,
        help="resumable per-seed JSON. Default: "
        "$EGOWAM_ROBOTWIN_ROOT/_baselines_ee/_eval/<policy>_<task>.json",
    )
    p.add_argument(
        "--n-target", type=int, default=100, help="number of valid held-out seeds"
    )
    p.add_argument(
        "--seed-base", type=int, default=100000, help="first candidate held-out seed"
    )
    p.add_argument("--max-candidates", type=int, default=600)
    p.add_argument(
        "--exec-horizon", type=int, default=30, help="waypoints executed per replan"
    )
    p.add_argument(
        "--steps-per-wp", type=int, default=15, help="sim steps per native waypoint"
    )
    p.add_argument(
        "--max-joint-vel", type=float, default=1.0, help="dense executor joint-vel cap"
    )
    p.add_argument(
        "--chunk-size", type=int, default=50, help="ACT num_queries / predicted chunk"
    )
    p.add_argument(
        "--n-obs-steps", type=int, default=3, help="DP obs history length (robot_dp_14)"
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=6,
        help="DP executed action steps per query",
    )
    args = p.parse_args()

    results_path = args.results_path or (
        f"$EGOWAM_ROBOTWIN_ROOT/_baselines_ee/_eval/{args.policy}_{args.task}.json"
    )

    _setup_robotwin_path()
    from test_render import Sapien_TEST

    Sapien_TEST()  # render/Vulkan check (raises if the GPU can't render)
    print(
        f"[eval_baseline_ee] Sapien render check passed; policy={args.policy} task={args.task}"
    )

    run_eval(
        args.policy,
        args.task,
        args.task_config,
        args.robot,
        args.ckpt_dir,
        results_path,
        n_target=args.n_target,
        seed_base=args.seed_base,
        max_candidates=args.max_candidates,
        exec_horizon=args.exec_horizon,
        steps_per_wp=args.steps_per_wp,
        max_joint_vel=args.max_joint_vel,
        chunk_size=args.chunk_size,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )


if __name__ == "__main__":
    main()
