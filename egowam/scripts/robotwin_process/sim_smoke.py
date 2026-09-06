#!/usr/bin/env python3
"""RoboTwin SAPIEN sim smoke / GT sanity gate for the EgoWAM bring-up (plan §5, §9).

Validates that the sim actually runs on THIS cluster for a given robot BEFORE trusting any learned
policy — the gate the plan calls for. Modes:

  --mode expert   : run RoboTwin's scripted expert (setup_demo -> play_once -> check_success) for N
                    seeds on <robot>. If the expert succeeds, the sim + planner (mplib/curobo) +
                    success check + robot config all work; any later policy failure is the policy's.
  --mode ee_path  : after setup_demo, read the achieved endpose via get_obs and feed it back through
                    TASK_ENV.take_action(a, action_type="ee") as a 16-dim [xyz+quat_wxyz+grip]x2 action.
                    Confirms our exact ee action contract executes (IK plan runs) end-to-end. Note:
                    this feeds a raw WORLD-frame pose and does NOT exercise the cam<->base / ypr->quat
                    conversion — use --mode gt_replay to validate that.
  --mode gt_replay: THE proper bridge-conversion gate (plan §5/§9). Capture the scripted expert's
                    achieved world-frame endpose trajectory (a known-solvable GT), then replay it in
                    the SAME seeded scene THROUGH the exact policy conversion
                    (world -> cam14 -> egowam_deploy.cam14_to_ee16 -> 16-dim ee) into take_action("ee"),
                    counting EGOWAM_IK_FAIL, and check_success. High replay success + ~0 IK fails =>
                    the deploy bridge's frame/quaternion/gripper conversion is sound.

Train-episode debug gates (sim-rollout debug plan; default seeds come from the dataset's seed.txt
so `--episode-idx 0` reruns the EXACT scene of training episode0):
  --mode obs_parity     : setup the sim on a training episode's seed and diff the initial sim obs
                          (images + endpose) against frame 0 of the converted zarr episode. Proves
                          the training scene is exactly reproducible (Gate B).
  --mode traj_replay    : replay the dataset's shipped collection joint paths
                          (_traj_data/episodeN.pkl) via set_path_lst + play_once — the exact
                          collect_data.py path: native joint-space execution, no IK, no conversion.
                          Success => seed->scene parity AND scene solvability (Gate B').
  --mode gt_dense_replay: replay ALL recorded endposes of a training episode (world -> cam14 ->
                          ee16) through the native-rate dense streaming executor. Validates the
                          executor on dense targets with known-good values (Gate C).
  --mode policy_train_ep: closed-loop policy rollout on the training episode's seed using the
                          native-rate dense executor (Gate D). --compare-old also runs the original
                          take_action("ee")-per-upsampled-waypoint path for an A/B.

Must run under the combined env (emimic2_sim: sapien+mplib + egowam) on a GPU node (Vulkan).
Arg construction mirrors external/RoboTwin/script/eval_policy.py:main.

Example:
  cd external/RoboTwin && \
  ../../emimic2_sim/bin/python -m egowam.scripts.robotwin_process.sim_smoke \
      --task pick_diverse_bottles --task-config demo_clean --robot arx-x5 --seeds 3 --mode expert
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

# RoboTwin root — this file is egowam/scripts/robotwin_process/sim_smoke.py.
RT_ROOT = Path(__file__).resolve().parents[3] / "external" / "RoboTwin"

# Our robot names (dataset/model side) -> RoboTwin sim embodiment keys (_embodiment_config.yml).
SIM_EMB = {
    "aloha-agilex": "aloha-agilex",
    "arx-x5": "ARX-X5",
    "piper": "piper",
    "ur5": "ur5-wsg",
    "franka": "franka-panda",
}
# aloha-agilex is natively bimanual (dual_arm: True) -> len-1 embodiment. Every other robot is a
# single-arm URDF that RoboTwin duplicates into a bimanual pair via the len-3 form
# [robot, robot, arm_distance]. arx-x5 clean_50 data is symmetric about x=0 with EE homes at
# x=+/-0.30 -> arm separation ~= 0.6m.
NATIVE_BIMANUAL = {"aloha-agilex"}
DEFAULT_DUAL_DIS = 0.6


def _setup_robotwin_path():
    os.chdir(RT_ROOT)  # RoboTwin uses relative "./assets/..." paths
    # Mirror eval_policy.py's path setup; ./script holds test_render + generate_episode_instructions.
    for p in ("./", "./policy", "./description/utils", "./script"):
        if p not in sys.path:
            sys.path.append(p)


def _start_video(TASK_ENV, out_mp4, cam="front_camera"):
    """Wire an ffmpeg writer to TASK_ENV so take_action() streams frames from camera `cam` to out_mp4.

    RoboTwin's take_action writes now_obs["observation"][self._eval_video_cam]["rgb"] each step
    (patched to honor _eval_video_cam; default head_camera). video_size is read from that camera's
    live rgb so it matches whatever resolution the sim renders."""
    TASK_ENV._eval_video_cam = cam
    obs = (
        TASK_ENV.get_obs()
    )  # also primes now_obs for the first take_action frame write
    if cam == "third_view":
        fr = obs.get("third_view_rgb")
        if fr is None:
            raise SystemExit(
                "[sim_smoke] third_view_rgb missing — set data_type.third_view=True"
            )
        fr = fr["rgb"] if isinstance(fr, dict) else fr
    else:
        if cam not in obs["observation"]:
            raise SystemExit(
                f"[sim_smoke] camera '{cam}' not in obs (have: "
                f"{[k for k in obs['observation'] if isinstance(obs['observation'][k], dict)]})"
            )
        fr = obs["observation"][cam]["rgb"]
    h, w = np.asarray(fr).shape[:2]
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    # System ffmpeg isn't on the compute-node PATH; use the binary bundled with imageio-ffmpeg.
    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = "ffmpeg"
    ff = subprocess.Popen(
        [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{w}x{h}",
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            out_mp4,
        ],
        stdin=subprocess.PIPE,
    )
    TASK_ENV._set_eval_video_ffmpeg(ff)
    return out_mp4


def _load_args(
    task_name, task_config, robot, dual_dis=DEFAULT_DUAL_DIS, video_dir=None
):
    """Replicates eval_policy.main's config assembly.

    aloha-agilex is native bimanual -> len-1 embodiment. All other robots (arx-x5, piper, ur5,
    franka) are single-arm URDFs -> len-3 embodiment [emb, emb, dist] giving two separate arms at
    +/- dist/2 in x (eval_policy.main len==3 branch)."""
    from envs import CONFIGS_PATH

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    sim_emb = SIM_EMB.get(
        robot, robot
    )  # our name -> RoboTwin sim key (arx-x5 -> ARX-X5)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = None

    with open(
        os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8"
    ) as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    if sim_emb not in emb_types:
        raise SystemExit(
            f"[sim_smoke] sim embodiment '{sim_emb}' (robot '{robot}') not in "
            f"_embodiment_config.yml (have: {sorted(emb_types)})"
        )
    robot_file = emb_types[sim_emb]["file_path"]

    with open(
        os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8"
    ) as f:
        cam_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cam_cfg[head_type]["h"]
    args["head_camera_w"] = cam_cfg[head_type]["w"]

    def _emb_cfg(rf):
        with open(os.path.join(rf, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    if robot in NATIVE_BIMANUAL:
        args["embodiment"] = [sim_emb]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = sim_emb
    else:
        # Single-arm robot -> two separate arms (len-3 form, eval_policy.main len==3 branch).
        args["embodiment"] = [sim_emb, sim_emb, dual_dis]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["embodiment_dis"] = dual_dis
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{sim_emb}+{sim_emb}"
    args["left_embodiment_config"] = _emb_cfg(robot_file)
    args["right_embodiment_config"] = _emb_cfg(robot_file)
    # collect_data/eval_policy quiet a few flags for headless runs:
    args["render_freq"] = 0
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_mode"] = True
    # Video: setting eval_video_save_dir -> TASK_ENV.eval_video_path, so take_action streams frames.
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        args["eval_video_log"] = True
        args["eval_video_save_dir"] = video_dir
        # Enable the third-person observer render so now_obs['third_view_rgb'] is populated each step
        # (the default presentation-video camera). Small extra render cost, only during video runs.
        args.setdefault("data_type", {})
        args["data_type"]["third_view"] = True
    else:
        args["eval_video_log"] = False
    return args


def _class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    return getattr(envs_module, task_name)()


def run_expert(
    task_name, task_config, robot, seeds, video_dir=None, video_cam="third_view"
):
    args = _load_args(task_name, task_config, robot, video_dir=video_dir)
    TASK_ENV = _class_decorator(task_name)
    n_ok = 0
    for i in range(seeds):
        now_seed = 100000 * (1 + 0) + i
        vid = None
        try:
            TASK_ENV.setup_demo(now_ep_num=i, seed=now_seed, is_test=True, **args)
            if video_dir is not None:
                vid = _start_video(
                    TASK_ENV,
                    os.path.join(
                        video_dir,
                        f"expert_{robot}_{task_name}_{video_cam}_seed{now_seed}.mp4",
                    ),
                    cam=video_cam,
                )
            TASK_ENV.play_once()
            ok = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
            if vid is not None:
                TASK_ENV._del_eval_video_ffmpeg()
            TASK_ENV.close_env()
        except Exception as e:
            print(f"  seed {now_seed}: EXCEPTION {type(e).__name__}: {e}")
            try:
                TASK_ENV.close_env()
            except Exception:
                pass
            ok = False
        n_ok += int(ok)
        print(
            f"  seed {now_seed}: {'SUCCESS' if ok else 'fail'}"
            + (f"  video={vid}" if vid else "")
        )
    print(
        f"[expert] {task_name}/{task_config}/{robot}: {n_ok}/{seeds} scripted-expert success"
    )
    return n_ok


def run_ee_path(task_name, task_config, robot, seeds):
    """Feed the achieved endpose back as a 16-dim ee action to confirm the ee contract executes."""
    args = _load_args(task_name, task_config, robot)
    TASK_ENV = _class_decorator(task_name)
    for i in range(min(seeds, 1)):  # one scene is enough for a plumbing check
        now_seed = 100000 + i
        TASK_ENV.setup_demo(now_ep_num=i, seed=now_seed, is_test=True, **args)
        obs = TASK_ENV.get_obs()
        le = np.asarray(obs["endpose"]["left_endpose"], dtype=np.float64).reshape(
            7
        )  # xyz+wxyz
        re = np.asarray(obs["endpose"]["right_endpose"], dtype=np.float64).reshape(7)
        lg = float(np.asarray(obs["endpose"]["left_gripper"]).reshape(-1)[0])
        rg = float(np.asarray(obs["endpose"]["right_gripper"]).reshape(-1)[0])
        action = np.concatenate([le, [lg], re, [rg]])  # (16,) = [xyz+quat_wxyz+grip]x2
        assert action.shape == (16,), action.shape
        cnt0 = TASK_ENV.take_action_cnt
        TASK_ENV.take_action(
            action, action_type="ee"
        )  # identity move -> should plan+execute
        moved = TASK_ENV.take_action_cnt > cnt0
        TASK_ENV.close_env()
        print(
            f"[ee_path] seed {now_seed}: take_action('ee') executed={moved} "
            f"(action dim=16, endpose fed back as identity target)"
        )


def run_policy(
    task_name,
    task_config,
    robot,
    ckpt,
    seeds,
    max_replans=60,  # high cap; the real bound is take_action_cnt < step_lim (full episode)
    video_dir=None,
    video_cam="third_view",
):
    """Closed-loop plumbing smoke: EgoWAM deploy shim (model -> 16-dim ee -> curobo IK -> sim).

    A smoke ckpt is untrained, so we don't expect task success — we confirm the policy loads,
    produces finite 16-dim ee actions, and the sim executes receding-horizon replans without
    crashing. Mirrors eval_policy.eval_policy's inner loop but bounded to a couple episodes."""
    sys.path.append("./policy")
    import EgoWAM  # deploy_policy shim

    args = _load_args(task_name, task_config, robot, video_dir=video_dir)
    TASK_ENV = _class_decorator(task_name)
    model = EgoWAM.get_model({"ckpt_path": ckpt, "robot": robot})
    # Tag videos by the ckpt's run dir (e.g. hpt_single_pick) so multiple variants of the SAME
    # task don't overwrite each other's clip.
    pol_tag = Path(ckpt).parents[1].name if ckpt else "policy"
    for i in range(min(seeds, 2)):
        now_seed = 100000 + i
        vid = None
        try:
            TASK_ENV.setup_demo(now_ep_num=i, seed=now_seed, is_test=True, **args)
        except Exception as e:
            print(
                f"[policy] seed {now_seed}: setup EXCEPTION {type(e).__name__}: {e} (skip)"
            )
            continue
        EgoWAM.reset_model(model)
        if video_dir is not None:
            vid = _start_video(
                TASK_ENV,
                os.path.join(
                    video_dir,
                    f"policy_{pol_tag}_{task_name}_{video_cam}_seed{now_seed}.mp4",
                ),
                cam=video_cam,
            )
        replans = 0
        try:
            while (
                TASK_ENV.take_action_cnt < TASK_ENV.step_lim and replans < max_replans
            ):
                observation = TASK_ENV.get_obs()
                EgoWAM.eval(
                    TASK_ENV, model, observation
                )  # one replan = exec_horizon ee steps
                replans += 1
                if TASK_ENV.eval_success:
                    break
            if vid is not None:
                TASK_ENV._del_eval_video_ffmpeg()
            print(
                f"[policy] seed {now_seed}: replans={replans} steps={TASK_ENV.take_action_cnt} "
                f"success={TASK_ENV.eval_success} (closed-loop executed OK)"
                + (f"  video={vid}" if vid else "")
            )
        except Exception as e:
            import traceback

            print(
                f"[policy] seed {now_seed}: RUNTIME EXCEPTION {type(e).__name__}: {e}"
            )
            traceback.print_exc()
        finally:
            try:
                TASK_ENV.close_env()
            except Exception:
                pass


def run_gt_replay(
    task_name, task_config, robot, seeds, video_dir=None, video_cam="third_view"
):
    """Proper GT-replay sanity gate (plan §5/§9) — validates the deploy bridge's frame/quaternion/
    gripper conversion *closed-loop through SAPIEN*, which ``--selftest`` (numeric only) and
    ``ee_path`` (bypasses the conversion) both miss.

    Per seed, two phases in the SAME seeded scene:
      1. Run RoboTwin's scripted expert; wrap ``take_dense_action`` to record the achieved
         WORLD-frame endpose after each planned segment -> a known-solvable GT trajectory.
      2. Re-setup the scene and replay each recorded world endpose through the EXACT policy
         conversion (world -> cam14 via the training transform -> :func:`cam14_to_ee16` -> 16-dim
         ee) into ``take_action("ee")``, with the same ``EGOWAM_IK_FAIL`` counter the learned
         policy uses, then ``check_success``.

    HIGH replay success + NEAR-ZERO IK fails => the conversion is sound (so a low learned-policy
    success rate is a genuine policy result, not a bridge bug). Low replay success / many IK fails
    => the bridge's frame/quat/gripper conversion is broken."""
    from egowam.scripts.robotwin_process.egowam_deploy import (
        cam14_to_ee16,
        robotwin_extrinsics,
        world_endpose_to_cam14,
    )

    extr = robotwin_extrinsics()
    # NOTE: deliberately do NOT route video through _load_args' eval_video_save_dir — that makes
    # take_action auto-write STALE now_obs frames (now_obs is only refreshed on get_obs). We instead
    # refresh now_obs and write one frame per waypoint MANUALLY, uniformly for both phases. Videos
    # are recorded for seed 0 only (one expert + one replay clip per task).
    args = _load_args(task_name, task_config, robot)
    args.setdefault("data_type", {})
    args["data_type"]["endpose"] = True
    if video_dir is not None:
        args["data_type"]["third_view"] = True  # 3rd-person observer view for the clips
        os.makedirs(video_dir, exist_ok=True)
    TASK_ENV = _class_decorator(task_name)

    def _grab_frame(env):
        """Refresh now_obs and append one fresh frame to the active eval-video ffmpeg."""
        env.get_obs()
        env.eval_video_ffmpeg.stdin.write(env._eval_video_frame_bytes())

    n_expert_ok = n_replay_ok = 0
    video_taken = (
        False  # record clips for the first seed whose EXPERT SOLVES (robust to
    )
    # scenes where the expert fails / RoboTwin raises UnStableError at setup, e.g. pick).
    for i in range(seeds):
        now_seed = 100000 + i
        rec = video_dir is not None and not video_taken

        # ---- Phase 1: capture the scripted expert's achieved world-frame trajectory (GT). --------
        traj = []
        try:
            TASK_ENV.setup_demo(now_ep_num=i, seed=now_seed, is_test=True, **args)
        except Exception as e:
            print(
                f"[gt_replay] seed {now_seed}: expert setup EXCEPTION {type(e).__name__}: {e} (skip)"
            )
            continue

        # Capture the achieved world-frame endpose after EACH expert motion segment. TWO actuation
        # primitives must be wrapped: take_dense_action (single-arm / mixed moves) AND
        # together_move_to_pose (bimanual "together" moves — it runs its own set_arm_joints loop and
        # never calls take_dense_action, so wrapping only the latter silently drops the bimanual
        # lift/place segments, e.g. all of pick_diverse_bottles).
        def _snapshot(_rec=rec):
            try:
                le = np.asarray(TASK_ENV.get_arm_pose("left"), np.float64).reshape(7)
                re = np.asarray(TASK_ENV.get_arm_pose("right"), np.float64).reshape(7)
                lg = float(
                    np.asarray(TASK_ENV.robot.get_left_gripper_val()).reshape(-1)[0]
                )
                rg = float(
                    np.asarray(TASK_ENV.robot.get_right_gripper_val()).reshape(-1)[0]
                )
                traj.append((le, re, lg, rg))
                if _rec and getattr(TASK_ENV, "eval_video_ffmpeg", None) is not None:
                    _grab_frame(TASK_ENV)  # one expert frame per segment
            except Exception:
                pass

        orig_tda = TASK_ENV.take_dense_action
        orig_tmp = TASK_ENV.together_move_to_pose

        def _cap_tda(*a, _orig=orig_tda, **k):
            r = _orig(*a, **k)
            _snapshot()
            return r

        def _cap_tmp(*a, _orig=orig_tmp, **k):
            r = _orig(*a, **k)
            _snapshot()
            return r

        TASK_ENV.take_dense_action = _cap_tda
        TASK_ENV.together_move_to_pose = _cap_tmp
        exp_vid = None
        if rec:
            exp_vid = _start_video(
                TASK_ENV,
                os.path.join(
                    video_dir,
                    f"expert_{robot}_{task_name}_{video_cam}_seed{now_seed}.mp4",
                ),
                cam=video_cam,
            )
            _grab_frame(TASK_ENV)  # initial (home) frame
        expert_ok = False
        try:
            TASK_ENV.play_once()
            expert_ok = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
        except Exception as e:
            print(
                f"[gt_replay] seed {now_seed}: expert play EXCEPTION {type(e).__name__}: {e}"
            )
        TASK_ENV.take_dense_action = orig_tda
        TASK_ENV.together_move_to_pose = orig_tmp
        if exp_vid is not None:
            try:
                TASK_ENV._del_eval_video_ffmpeg()
            except Exception:
                pass
            if expert_ok:
                video_taken = True  # keep this expert clip; record its replay below
            else:
                try:
                    os.remove(
                        exp_vid
                    )  # discard failed clip; record the next solved seed
                except Exception:
                    pass
        try:
            TASK_ENV.close_env()
        except Exception:
            pass

        n_expert_ok += int(expert_ok)
        if not expert_ok or len(traj) == 0:
            print(
                f"[gt_replay] seed {now_seed}: expert {'ok' if expert_ok else 'FAILED'}, "
                f"waypoints={len(traj)} -> skip replay (need a solved expert demo as GT)"
            )
            continue

        # ---- Phase 2: replay GT world endposes THROUGH the conversion into take_action('ee'). ----
        try:
            TASK_ENV.setup_demo(now_ep_num=i, seed=now_seed, is_test=True, **args)
        except Exception as e:
            print(
                f"[gt_replay] seed {now_seed}: replay setup EXCEPTION {type(e).__name__}: {e} (skip)"
            )
            continue

        vid = None
        if rec:
            vid = _start_video(
                TASK_ENV,
                os.path.join(
                    video_dir,
                    f"gtreplay_{robot}_{task_name}_{video_cam}_seed{now_seed}.mp4",
                ),
                cam=video_cam,
            )
            _grab_frame(TASK_ENV)  # initial (home) frame

        ik = {
            "n": 0
        }  # EGOWAM_IK_FAIL markers on the ee planning path (mirror deploy_policy).
        for side in ("left", "right"):
            name = f"{side}_plan_path"
            if hasattr(TASK_ENV.robot, name):

                def _wrap(orig, s):
                    def w(*a, **k):
                        r = orig(*a, **k)
                        try:
                            if isinstance(r, dict) and r.get("status") != "Success":
                                ik["n"] += 1
                                print(f"EGOWAM_IK_FAIL:{s}", flush=True)
                        except Exception:
                            pass
                        return r

                    return w

                setattr(
                    TASK_ENV.robot, name, _wrap(getattr(TASK_ENV.robot, name), side)
                )

        replay_ok = False
        try:
            for le, re, lg, rg in traj:
                a14 = world_endpose_to_cam14(le, re, lg, rg, robot)
                a16 = cam14_to_ee16(a14, extr)[0]  # (16,)
                TASK_ENV.take_action(a16, action_type="ee")
                if rec:
                    _grab_frame(TASK_ENV)  # one replay frame per waypoint
                if TASK_ENV.eval_success:
                    break
            replay_ok = bool(TASK_ENV.eval_success or TASK_ENV.check_success())
        except Exception as e:
            import traceback

            print(
                f"[gt_replay] seed {now_seed}: replay EXCEPTION {type(e).__name__}: {e}"
            )
            traceback.print_exc()
        finally:
            if vid is not None:
                try:
                    TASK_ENV._del_eval_video_ffmpeg()
                except Exception:
                    pass
            try:
                TASK_ENV.close_env()
            except Exception:
                pass

        n_replay_ok += int(replay_ok)
        print(
            f"[gt_replay] seed {now_seed}: expert=ok replay_success={replay_ok} "
            f"waypoints={len(traj)} ik_fails={ik['n']} steps={TASK_ENV.take_action_cnt}"
            + (f"  video={vid}" if vid else "")
        )

    denom = n_expert_ok if n_expert_ok else seeds
    print(
        f"[gt_replay] {task_name}/{task_config}/{robot}: expert {n_expert_ok}/{seeds} solved; "
        f"GT-replay-through-conversion {n_replay_ok}/{denom} success"
    )
    return n_replay_ok


# --------------------------------------------------------------------------------------------
# Train-episode debug gates (sim-rollout debug plan): the dataset's seed.txt maps episode index
# -> generation seed (collect_data.py:205 `seed=seed_list[episode_idx]`), so we can re-create the
# EXACT scene of a training episode in the sim.
# --------------------------------------------------------------------------------------------
# Override with EGOWAM_ROBOTWIN_ROOT (mirrors paths.robotwin_root in the Hydra configs).
DATASET_ROOT = Path(
    os.environ.get("EGOWAM_ROBOTWIN_ROOT", "/coc/flash7/scratch/egowm/robotwin")
)


def _dataset_dir(task_name, robot, variant="clean_50"):
    return DATASET_ROOT / "dataset" / task_name / f"{robot}_{variant}"


def _episode_seed(task_name, robot, ep_idx, variant="clean_50"):
    """Training episode index -> RoboTwin generation seed (positional map in seed.txt)."""
    seeds = (_dataset_dir(task_name, robot, variant) / "seed.txt").read_text().split()
    return int(seeds[ep_idx])


def _episode_zarr(task_name, robot, ep_idx, variant="clean_50"):
    return (
        DATASET_ROOT
        / "zarrDataset"
        / task_name
        / f"{robot}_{variant}"
        / f"episode{ep_idx}.zarr"
    )


def _zarr_frame(z, key, t):
    """zarr JPEG bytes -> (H,W,3) uint8 RGB."""
    import cv2

    raw = z[key][t]
    while isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = raw.item()
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_obs_parity(task_name, task_config, robot, ep_idx, variant, out_dir):
    """Gate B: sim scene on the training episode's seed == the training episode's frame 0?"""
    import cv2
    import zarr

    seed = _episode_seed(task_name, robot, ep_idx, variant)
    zpath = _episode_zarr(task_name, robot, ep_idx, variant)
    z = zarr.open(str(zpath), mode="r")
    os.makedirs(out_dir, exist_ok=True)

    args = _load_args(task_name, task_config, robot)
    args.setdefault("data_type", {})
    args["data_type"]["endpose"] = True
    TASK_ENV = _class_decorator(task_name)
    # Mirror collect_data.py: no is_test (scene draw depends only on the numpy seed).
    TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=seed, **args)
    obs = TASK_ENV.get_obs()

    cam_pairs = [
        ("head_camera", "images.front_1"),
        ("left_camera", "images.left_wrist"),
        ("right_camera", "images.right_wrist"),
    ]
    print(f"[obs_parity] episode{ep_idx} (seed {seed}) vs {zpath}")
    for sim_cam, zkey in cam_pairs:
        sim_img = np.asarray(obs["observation"][sim_cam]["rgb"])
        ref_img = _zarr_frame(z, zkey, 0)
        if sim_img.shape != ref_img.shape:
            print(
                f"  {sim_cam}: SHAPE MISMATCH sim={sim_img.shape} zarr={ref_img.shape}"
            )
            continue
        diff = np.abs(sim_img.astype(np.int16) - ref_img.astype(np.int16))
        side = np.concatenate(
            [ref_img, sim_img, np.clip(diff * 4, 0, 255).astype(np.uint8)], axis=1
        )
        out_png = os.path.join(out_dir, f"parity_{sim_cam}_ep{ep_idx}.png")
        cv2.imwrite(out_png, cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
        print(
            f"  {sim_cam}: mean|d|={diff.mean():.2f} max|d|={diff.max()} "
            f"frac(|d|>10)={float((diff > 10).mean()):.4f}  [zarr | sim | 4x diff] -> {out_png}"
        )
    for side_name in ("left", "right"):
        sim_ee = np.asarray(obs["endpose"][f"{side_name}_endpose"], np.float64).reshape(
            7
        )
        ref_ee = np.asarray(z[f"{side_name}.obs_ee_pose"][0], np.float64).reshape(7)
        dxyz = float(np.abs(sim_ee[:3] - ref_ee[:3]).max())
        qdot = abs(float(np.dot(sim_ee[3:7], ref_ee[3:7])))
        sim_g = float(np.asarray(obs["endpose"][f"{side_name}_gripper"]).reshape(-1)[0])
        ref_g = float(np.asarray(z[f"{side_name}.obs_gripper"][0]).reshape(-1)[0])
        print(
            f"  {side_name} endpose: max|dxyz|={dxyz:.5f} |quat.dot|={qdot:.6f} "
            f"gripper sim={sim_g:.4f} zarr={ref_g:.4f} (expect ~0 / ~1 / equal)"
        )
    TASK_ENV.close_env()


def run_traj_replay(
    task_name,
    task_config,
    robot,
    ep_idx,
    variant,
    video_dir=None,
    video_cam="third_view",
):
    """Gate B': replay the dataset's collection joint paths (_traj_data/episodeN.pkl) via the
    exact collect_data.py mechanism (need_plan=False -> move primitives pop stored plans)."""
    import pickle

    seed = _episode_seed(task_name, robot, ep_idx, variant)
    pkl = (
        _dataset_dir(task_name, robot, variant) / "_traj_data" / f"episode{ep_idx}.pkl"
    )
    with open(pkl, "rb") as f:
        traj_data = pickle.load(f)

    args = _load_args(task_name, task_config, robot, video_dir=video_dir)
    args["need_plan"] = False
    TASK_ENV = _class_decorator(task_name)
    ok = False
    vid = None
    try:
        TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=seed, **args)
        TASK_ENV.set_path_lst(
            {
                "need_plan": False,
                "left_joint_path": traj_data["left_joint_path"],
                "right_joint_path": traj_data["right_joint_path"],
            }
        )
        if video_dir is not None:
            vid = _start_video(
                TASK_ENV,
                os.path.join(
                    video_dir,
                    f"trajreplay_{robot}_{task_name}_{video_cam}_ep{ep_idx}_seed{seed}.mp4",
                ),
                cam=video_cam,
            )

            # play_once never calls take_action, so eval-video frames must be grabbed manually
            # (same trick as run_gt_replay): one frame after each motion segment.
            def _snap():
                try:
                    TASK_ENV.get_obs()
                    TASK_ENV.eval_video_ffmpeg.stdin.write(
                        TASK_ENV._eval_video_frame_bytes()
                    )
                except Exception:
                    pass

            for _name in ("take_dense_action", "together_move_to_pose"):
                _orig = getattr(TASK_ENV, _name)

                def _wrapped(*a, _orig=_orig, **k):
                    r = _orig(*a, **k)
                    _snap()
                    return r

                setattr(TASK_ENV, _name, _wrapped)
            _snap()  # initial (home) frame
        TASK_ENV.play_once()
        ok = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
        if vid is not None:
            TASK_ENV._del_eval_video_ffmpeg()
    except Exception as e:
        import traceback

        print(
            f"[traj_replay] ep{ep_idx} seed {seed}: EXCEPTION {type(e).__name__}: {e}"
        )
        traceback.print_exc()
    finally:
        try:
            TASK_ENV.close_env()
        except Exception:
            pass
    print(
        f"[traj_replay] {task_name}/{robot} episode{ep_idx} (seed {seed}): "
        f"{'SUCCESS' if ok else 'FAIL'} (native joint replay of _traj_data)"
        + (f"  video={vid}" if vid else "")
    )
    return ok


def _read_seed_file(path):
    """Read a whitespace-separated seed list (RoboTwin's own seed.txt format). None if unset."""
    if not path:
        return None
    from pathlib import Path as _P

    return [int(t) for t in _P(path).expanduser().read_text().split()]


def run_eval_seeds(
    task_name,
    task_config,
    robot,
    ckpt,
    variant_tag,
    results_path,
    n_target=50,
    seed_base=100000,
    max_candidates=600,
    eval_seeds=None,
    exec_horizon=30,
    steps_per_wp=15,
    num_inference_steps=10,
    max_joint_vel=1.0,
    ik="mink",
    ensemble=False,
    ensemble_m=0.1,
    query_stride=10,
):
    """Held-out success-rate eval: run the policy closed-loop (mink IK + velocity cap) over
    ``n_target`` VALID (setup-stable), NON-training seeds and log per-seed success.

    Fixed across variants: candidate seeds are the deterministic ordered list ``seed_base, +1, ...``
    (>= 100000, disjoint from the training seed.txt which is <200). Setup-unstable seeds (RoboTwin
    ``UnStableError`` etc.) are deterministic per seed, so every variant skips the same ones and
    evaluates the identical first-``n_target`` stable seeds. Budget per episode = the task's
    production ``step_lim``.

    Resumable: results are written to ``results_path`` (JSON list) after EVERY seed, so a kill /
    overcap preemption loses at most one in-flight episode; on restart, seeds already in the file
    are skipped and evaluation continues until ``n_target`` valid evals."""
    import json

    from egowam.scripts.robotwin_process.egowam_deploy import EgoWAMPolicy

    sys.path.append("./policy")
    import EgoWAM  # deploy shim: encode_obs

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results = []
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                results = json.load(f)
        except Exception:
            results = []
    done = {r["seed"] for r in results}

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
        f"({n_succ()} success) already logged"
    )
    if n_valid() >= n_target:
        _eval_summary(variant_tag, task_name, results, n_target)
        return

    model = EgoWAMPolicy(
        ckpt,
        robot=robot,
        num_inference_steps=num_inference_steps,
        exec_horizon=exec_horizon,
        native_rate=True,
        ensemble=ensemble,
        ensemble_m=ensemble_m,
        query_stride=query_stride,
    )
    args = _load_args(task_name, task_config, robot)
    args.setdefault("data_type", {})
    args["data_type"]["endpose"] = True
    TASK_ENV = _class_decorator(task_name)

    # An explicit list pins the evaluation set so a published table can be reproduced exactly;
    # without one, walk candidates from seed_base and keep the first n_target stable ones.
    if eval_seeds:
        candidates = list(eval_seeds)
        n_target = len(candidates)
    else:
        candidates = range(seed_base, seed_base + max_candidates)
    tried = 0
    for s in candidates:
        if not eval_seeds and (n_valid() >= n_target or tried >= max_candidates):
            break
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
        # ---- closed-loop policy rollout (mink IK + vcap) to the production step_lim ----
        restore_ik = (
            patch_mink_ik(TASK_ENV, verbose=False) if ik == "mink" else (lambda: None)
        )
        EgoWAM.reset_model(model)
        rec = {"seed": s}
        try:
            replans, ikf = 0, 0
            while (
                TASK_ENV.take_action_cnt < TASK_ENV.step_lim
                and not TASK_ENV.eval_success
            ):
                obs = EgoWAM.encode_obs(TASK_ENV.get_obs())
                wps = model.get_action(obs)
                st = execute_ee_dense(
                    TASK_ENV,
                    wps,
                    steps_per_wp=steps_per_wp,
                    max_joint_vel=max_joint_vel,
                )
                ikf += st["ik_fails"]
                replans += 1
            ok = bool(TASK_ENV.eval_success or TASK_ENV.check_success())
            rec.update(
                status="success" if ok else "fail",
                steps=int(TASK_ENV.take_action_cnt),
                replans=replans,
                ik_fails=ikf,
            )
        except Exception as e:
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


def _eval_summary(variant_tag, task_name, results, n_target):
    succ = sum(1 for r in results if r["status"] == "success")
    valid = sum(1 for r in results if r["status"] in ("success", "fail", "error"))
    skips = sum(1 for r in results if str(r["status"]).startswith("skip"))
    rate = 100.0 * succ / valid if valid else 0.0
    done = "DONE" if valid >= n_target else f"INCOMPLETE({valid}/{n_target})"
    print(
        f"[eval_summary] {variant_tag} {task_name}: {succ}/{valid} = {rate:.1f}% success "
        f"({skips} unstable seeds skipped)  [{done}]",
        flush=True,
    )


def _arm_base_pose(robot, side):
    """World pose (7,) xyz+wxyz of the arm's BASE LINK (fl_base_link / fr_base_link) — the true
    parent frame of the 6-DoF chain my MJCF starts at. NOT the articulation root
    (`{side}_entity.get_pose()`), which is the whole-robot footprint at `robot_pose`: the fixed
    root->base_link offset sits on the LEFT of FK(q) and a right-side TCP calibration cannot absorb
    it, so using the root gives config-dependent IK error. Read the link pose live from the sim."""
    base_link = "fl_base_link" if side == "left" else "fr_base_link"
    ent = getattr(robot, f"{side}_entity")
    for lk in ent.get_links():
        if lk.get_name() == base_link:
            p = lk.get_pose()
            return np.concatenate(
                [np.array(p.p, np.float64), np.array(p.q, np.float64)]
            )
    raise RuntimeError(f"{base_link} not found on {side}_entity")


def run_mink_selftest(task_name, task_config, robot, ep_idx, variant, n_probe=5):
    """Validate that the real-robot mink solver (EvaMinkKinematicsSolver + model_x5.xml) reproduces
    the aloha arm's kinematics in sim: calibrate the base/TCP frames at the home config, then at
    several DIFFERENT joint configs check (a) FK+calib predicts the achieved endpose and (b) IK
    round-trips the joints. Small errors => mink IK is a faithful drop-in for the executor."""
    from egowam.scripts.robotwin_process.egowam_mink import MinkArmIK

    seed = _episode_seed(task_name, robot, ep_idx, variant)
    args = _load_args(task_name, task_config, robot)
    args.setdefault("data_type", {})
    args["data_type"]["endpose"] = True
    TASK_ENV = _class_decorator(task_name)
    TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=seed, **args)
    robot_ = TASK_ENV.robot
    mink = MinkArmIK()

    getj = {
        "left": robot_.get_left_arm_real_jointState,
        "right": robot_.get_right_arm_real_jointState,
    }
    base = {s: _arm_base_pose(robot_, s) for s in ("left", "right")}

    def _read(side):
        j = np.asarray(getj[side]()[:6], np.float64)
        ep = np.asarray(TASK_ENV.get_arm_pose(side), np.float64).reshape(7)
        return j, ep

    def _drive(targets, steps=25):
        for _ in range(steps):
            for side in ("left", "right"):
                robot_.set_arm_joints(targets[side], np.zeros(6), side)
            TASK_ENV.scene.step()

    # ---- Phase 1: collect sign-calibration samples (home + several distinct configs). ----------
    rng = np.random.RandomState(0)
    samples = {"left": [], "right": []}
    for side in ("left", "right"):
        samples[side].append(_read(side))  # home (anchor)
    for _ in range(6):
        tg = {
            s: samples[s][0][0] + rng.uniform(-0.5, 0.5, 6) for s in ("left", "right")
        }
        _drive(tg)
        for side in ("left", "right"):
            samples[side].append(_read(side))
    for side in ("left", "right"):
        res = mink.calibrate_signs(side, base[side], samples[side])
        print(
            f"[mink_selftest] {side}: signs={mink.sign[side].astype(int).tolist()} "
            f"calib_residual={res*1000:.3f}mm  base={np.round(base[side][:3],3)}"
        )

    # ---- Phase 2: verify on fresh configs. ----------------------------------------------------
    fk_err, ik_err = {"left": [], "right": []}, {"left": [], "right": []}
    for k in range(n_probe):
        # Perturb both arms to a fresh config (distinct from the calibration samples) and settle.
        tg = {
            s: samples[s][0][0] + rng.uniform(-0.6, 0.6, 6) for s in ("left", "right")
        }
        _drive(tg)
        for side in ("left", "right"):
            j, ep = _read(side)
            dpos, drot = mink.verify_fk(side, j, ep)
            fk_err[side].append((dpos, drot))
            sol = mink.ik_world(side, ep, j)  # seed from the true joints
            ije = float(np.linalg.norm(sol - j)) if sol is not None else float("nan")
            ik_err[side].append(ije)
            print(
                f"[mink_selftest] probe {k} {side}: FK dpos={dpos*1000:6.2f}mm drot={np.degrees(drot):5.2f}deg "
                f"| IK |dq|={ije:.4f}rad"
            )
    for side in ("left", "right"):
        fp = np.array([e[0] for e in fk_err[side]])
        fr = np.array([e[1] for e in fk_err[side]])
        iq = np.array(ik_err[side])
        print(
            f"[mink_selftest] {side} SUMMARY: FK dpos mean={fp.mean()*1000:.2f}mm max={fp.max()*1000:.2f}mm "
            f"drot mean={np.degrees(fr.mean()):.2f}deg | IK |dq| mean={np.nanmean(iq):.4f} max={np.nanmax(iq):.4f}rad"
        )
    verdict = all(
        np.array([e[0] for e in fk_err[s]]).max() < 0.008
        and np.degrees(np.array([e[1] for e in fk_err[s]]).max()) < 2.0
        and np.nanmax(ik_err[s]) < 0.05
        for s in ("left", "right")
    )
    print(
        f"[mink_selftest] VERDICT: {'FAITHFUL (FK<8mm/<2deg, IK|dq|<0.05) — mink reproduces the aloha arm; safe as the real-world IK' if verdict else 'STILL OFF — inspect signs/base frame'}"
    )
    TASK_ENV.close_env()
    return verdict


def _calibrate_mink(TASK_ENV, perturb=0.2, n=6, verbose=True):
    """Build + auto-calibrate a MinkArmIK against the live sim (base-link poses, per-joint sign,
    TCP offset) by sampling a few small joint perturbations, then restore the arms to their start
    config. Returns the calibrated MinkArmIK. Perturbations are small and the arms start folded
    near the (elevated, behind-table) base, so scene objects are not disturbed."""
    from egowam.scripts.robotwin_process.egowam_mink import MinkArmIK

    robot_ = TASK_ENV.robot
    getj = {
        "left": robot_.get_left_arm_real_jointState,
        "right": robot_.get_right_arm_real_jointState,
    }
    base = {s: _arm_base_pose(robot_, s) for s in ("left", "right")}

    def _read(side):
        return (
            np.asarray(getj[side]()[:6], np.float64),
            np.asarray(TASK_ENV.get_arm_pose(side), np.float64).reshape(7),
        )

    def _drive(targets, steps=25):
        for _ in range(steps):
            for side in ("left", "right"):
                robot_.set_arm_joints(targets[side], np.zeros(6), side)
            TASK_ENV.scene.step()

    home = {s: _read(s)[0] for s in ("left", "right")}
    rng = np.random.RandomState(0)
    samples = {s: [_read(s)] for s in ("left", "right")}
    for _ in range(n):
        _drive(
            {s: home[s] + rng.uniform(-perturb, perturb, 6) for s in ("left", "right")}
        )
        for s in ("left", "right"):
            samples[s].append(_read(s))
    _drive(home)  # restore the arms to the start config before the rollout
    mink = MinkArmIK()
    for s in ("left", "right"):
        res = mink.calibrate_signs(s, base[s], samples[s])
        if verbose:
            print(
                f"[mink_ik] {s}: calibrated signs={mink.sign[s].astype(int).tolist()} "
                f"residual={res*1000:.3f}mm base={np.round(base[s][:3],3)}"
            )
    return mink


def patch_mink_ik(TASK_ENV, verbose=True):
    """Reproduce the REAL-WORLD IK in sim: make ``robot.{left,right}_plan_path`` resolve each ee
    target with the hardware solver (``EvaMinkKinematicsSolver`` via MinkArmIK, seeded from the
    current joints) instead of curobo's global ``plan_single``. Local + seeded => nearest-branch
    joints => no replan-boundary snaps, matching ``rollout.py``. Returns restore()."""
    robot = TASK_ENV.robot
    mink = _calibrate_mink(TASK_ENV, verbose=verbose)
    getj = {
        "left": robot.get_left_arm_real_jointState,
        "right": robot.get_right_arm_real_jointState,
    }
    saved = {}

    def _make(side):
        orig = getattr(robot, f"{side}_plan_path")

        def patched(target_pose, *a, _s=side, _orig=orig, **k):
            tp = np.asarray(target_pose, np.float64).reshape(-1)[:7]  # world xyz+wxyz
            try:
                sol = mink.ik_world(_s, tp, np.asarray(getj[_s]()[:6], np.float64))
            except Exception as e:
                # mink raises (e.g. NoSolutionFound: QP daqp failed) on unreachable targets —
                # e.g. an OOD policy pose. Degrade to Fail -> the executor holds the arm
                # (a real robot simply wouldn't track an infeasible target either).
                print(
                    f"[mink_ik] {_s} IK exception {type(e).__name__} -> hold",
                    flush=True,
                )
                return {"status": "Fail"}
            if sol is None:
                return {"status": "Fail"}
            return {
                "status": "Success",
                "position": np.asarray(sol, np.float64)[
                    None, :
                ],  # (1,6): executor takes [-1]
                "velocity": np.zeros((1, 6)),
            }

        setattr(robot, f"{side}_plan_path", patched)
        saved[side] = orig

    for side in ("left", "right"):
        _make(side)
    print("[mink_ik] real-robot mink IK installed on left/right plan_path")

    def restore():
        for side, orig in saved.items():
            setattr(robot, f"{side}_plan_path", orig)

    return restore


def patch_local_ik(TASK_ENV):
    """Make `robot.{left,right}_plan_path` resolve each ee target with a **local IK seeded from the
    current joints** instead of curobo's global `plan_single` motion plan — the sim analog of the
    real robot's `kinematics_solver.ik(..., cur_jnts=current)` (Mink, robot_interface.py:258). A
    global planner freely picks any IK branch / collision route, so consecutive near-identical ee
    targets (and the first target of each replanned chunk) can land on a different branch → the
    joint-space snaps seen in the video. Seeding curobo's IKSolver with the current qpos returns the
    solution nearest the seed, guaranteeing joint continuity.

    Implemented by monkeypatching each in-process planner's `motion_gen.plan_single` (aloha-agilex
    planners are in-process: `communication_flag=False`), so the EXACT `plan_path` transform chain
    (gripper→endlink, world→base, aloha bias) is reused — only the solver at the end changes.
    Returns a restore() callable. No-op (returns a callable) if the IK solver is unavailable."""
    from types import SimpleNamespace

    import torch

    robot = TASK_ENV.robot
    planners = [
        getattr(robot, "left_planner", None),
        getattr(robot, "right_planner", None),
    ]
    if any(p is None for p in planners) or getattr(robot, "communication_flag", False):
        print(
            "[local_ik] planners not in-process (or missing) -> keeping global plan_single"
        )
        return lambda: None

    saved = []

    def _make(mg):
        orig = mg.plan_single
        ik = getattr(mg, "ik_solver", None)

        def patched(
            start_joint_states, goal_pose, plan_config=None, _ik=ik, _orig=orig
        ):
            if _ik is None:
                return _orig(start_joint_states, goal_pose, plan_config)
            try:
                seed = start_joint_states.position.view(1, 1, -1)
                res = _ik.solve_single(
                    goal_pose,
                    retract_config=start_joint_states.position.view(1, -1),
                    seed_config=seed,
                    num_seeds=1,
                    use_nn_seed=False,
                )
                ok = bool(res.success.view(-1)[0].item())
                if not ok:  # fall back to the global planner if local IK misses
                    return _orig(start_joint_states, goal_pose, plan_config)
                sol = res.solution.view(1, -1)
                return SimpleNamespace(
                    success=torch.tensor(True),
                    interpolated_plan=SimpleNamespace(
                        position=sol, velocity=torch.zeros_like(sol)
                    ),
                )
            except (
                Exception
            ) as e:  # any API mismatch -> global planner (never crash the rollout)
                print(
                    f"[local_ik] solve_single failed ({type(e).__name__}: {e}); using global plan"
                )
                return _orig(start_joint_states, goal_pose, plan_config)

        mg.plan_single = patched
        saved.append((mg, orig))

    for p in planners:
        mg = getattr(p, "motion_gen", None)
        if mg is not None and hasattr(mg, "plan_single"):
            _make(mg)
    print(f"[local_ik] seeded local IK installed on {len(saved)} planner(s)")

    def restore():
        for mg, orig in saved:
            mg.plan_single = orig

    return restore


def execute_ee_dense(
    TASK_ENV, waypoints, steps_per_wp=15, max_joint_vel=1.0, record=False
):
    """Stream 16-dim ee waypoints at the data's native rate: ``steps_per_wp`` sim steps per
    waypoint (15 @ 250 Hz = the recorded 60 ms frame spacing), IK per waypoint, linear joint
    interpolation between COMMANDED targets — the sim analog of the real robot's streamed-target +
    low-level-interpolator loop (take_dense_action semantics driven by ee targets), replacing
    take_action('ee')'s plan-to-stop-per-waypoint regime.

    ``max_joint_vel`` (rad/s, matching EvaMinkKinematicsSolver's 1.0 limit) is the sim analog of the
    hardware's 0.4 s ``ts_offset`` interpolation lookahead: a waypoint whose joint move would exceed
    this velocity over ``steps_per_wp`` is spread across MORE sim steps so peak joint velocity stays
    bounded. Normal 60 ms waypoints are untouched; only replan-boundary command discontinuities and
    the initial reach-from-home get eased out (they'd otherwise snap in one 60 ms step). Set to 0 to
    disable (fixed steps_per_wp).

    Contract (drop-in for take_action-based eval loops): increments ``TASK_ENV.take_action_cnt``
    once per waypoint, respects ``step_lim``, sets ``eval_success`` (+ refreshes obs) on success,
    and (record=True) writes one eval-video frame per waypoint via the manually attached ffmpeg.
    Returns ``{"executed": n, "ik_fails": m}``."""
    robot = TASK_ENV.robot
    dt = 1.0 / 250.0
    debug = os.environ.get("EGOWAM_DEBUG_JUMPS")  # per-waypoint delta logging when set
    # Previous COMMANDED targets (drive targets, not measured qpos): interpolating from these
    # avoids PD-lag discontinuities at waypoint boundaries.
    prev_q = {
        "left": np.asarray(robot.get_left_arm_jointState()[:-1], np.float64),
        "right": np.asarray(robot.get_right_arm_jointState()[:-1], np.float64),
    }
    prev_g = {
        "left": float(robot.get_left_gripper_val()),
        "right": float(robot.get_right_gripper_val()),
    }
    prev_wp = {
        "left": None,
        "right": None,
    }  # previous waypoint's Cartesian target (for debug)
    plan = {"left": robot.left_plan_path, "right": robot.right_plan_path}
    stats = {"executed": 0, "ik_fails": 0}
    sim_ctr = {
        "n": 0
    }  # global sim-step counter -> uniform video cadence (every steps_per_wp)

    def _frame():
        # Emit a video frame at a fixed SIM-STEP cadence (not per waypoint): a velocity-capped
        # waypoint takes >steps_per_wp steps, so per-waypoint framing would still show it as a
        # 1-frame jump. Uniform sim-time sampling makes the eased motion visible.
        if record and getattr(TASK_ENV, "eval_video_ffmpeg", None) is not None:
            try:
                TASK_ENV.get_obs()
                TASK_ENV.eval_video_ffmpeg.stdin.write(
                    TASK_ENV._eval_video_frame_bytes()
                )
            except Exception:
                pass

    for wp in np.asarray(waypoints, np.float64):
        if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
            break
        TASK_ENV.take_action_cnt += 1
        tgt_q, tgt_g = {}, {}
        for side, sl_pose, i_grip in (
            ("left", slice(0, 7), 7),
            ("right", slice(8, 15), 15),
        ):
            res = plan[side](wp[sl_pose])  # IK: curobo plan; final row = joint solution
            pos = res.get("position") if isinstance(res, dict) else None
            if (
                isinstance(res, dict)
                and res.get("status") == "Success"
                and pos is not None
                and len(pos)
            ):
                tgt_q[side] = np.asarray(pos[-1], np.float64)
            else:  # plan/IK failure returns {"status": "Fail"} with NO "position" -> hold
                stats["ik_fails"] += 1
                print(f"EGOWAM_IK_FAIL:{side}", flush=True)
                tgt_q[side] = prev_q[side]
            tgt_g[side] = float(np.clip(wp[i_grip], 0.0, 1.0))
        if debug:
            # Per-waypoint jerk diagnosis: a small Cartesian delta with a large joint delta = an IK
            # branch switch (curobo picked a different solution) -> a visible jump in the video.
            msg = [f"[wp {stats['executed']:3d}]"]
            for side, sl_pose in (("left", slice(0, 3)), ("right", slice(8, 11))):
                dcart = (
                    float(np.linalg.norm(wp[sl_pose] - prev_wp[side]))
                    if prev_wp[side] is not None
                    else 0.0
                )
                djoint = float(np.linalg.norm(tgt_q[side] - prev_q[side]))
                flag = " <<JUMP" if (djoint > 0.35 and dcart < 0.03) else ""
                msg.append(
                    f"{side}:dxyz={dcart*1000:5.1f}mm djoint={djoint:5.3f}rad{flag}"
                )
                prev_wp[side] = np.asarray(wp[sl_pose])
            print("  ".join(msg), flush=True)
        # Velocity cap (ts_offset analog): spread large joint moves over more sim steps so peak
        # joint velocity stays <= max_joint_vel; normal native-rate waypoints keep steps_per_wp.
        n_steps = steps_per_wp
        if max_joint_vel and max_joint_vel > 0:
            max_dq = max(
                float(np.abs(tgt_q[s] - prev_q[s]).max()) for s in ("left", "right")
            )
            n_steps = max(steps_per_wp, int(np.ceil(max_dq / (max_joint_vel * dt))))
        # Finite-difference velocity targets (constant over the segment; zero at arrival/hold).
        vel = {s: (tgt_q[s] - prev_q[s]) / (n_steps * dt) for s in ("left", "right")}
        for s_i in range(1, n_steps + 1):
            a = s_i / n_steps
            for side in ("left", "right"):
                q = prev_q[side] + a * (tgt_q[side] - prev_q[side])
                v = vel[side] if s_i < n_steps else np.zeros_like(vel[side])
                robot.set_arm_joints(q, v, side)
                robot.set_gripper(prev_g[side] + a * (tgt_g[side] - prev_g[side]), side)
            TASK_ENV.scene.step()
            TASK_ENV._update_render()
            sim_ctr["n"] += 1
            if sim_ctr["n"] % steps_per_wp == 0:  # uniform ~60 ms video cadence
                _frame()
            if TASK_ENV.check_success():
                TASK_ENV.eval_success = True
                TASK_ENV.get_obs()
                _frame()
                stats["executed"] += 1
                return stats
        prev_q, prev_g = tgt_q, tgt_g
        stats["executed"] += 1
    return stats


def run_gt_dense_replay(
    task_name,
    task_config,
    robot,
    ep_idx,
    variant,
    steps_per_wp=15,
    max_joint_vel=1.0,
    ik="mink",
    video_dir=None,
    video_cam="third_view",
):
    """Gate C: replay ALL recorded endposes of a training episode (world -> cam14 -> ee16, the
    exact deploy conversion) through the native-rate dense executor. Success with ~0 IK fails =>
    the executor handles dense streams of known-good targets."""
    import zarr

    from egowam.scripts.robotwin_process.egowam_deploy import (
        cam14_to_ee16,
        robotwin_extrinsics,
        world_endpose_to_cam14,
    )

    seed = _episode_seed(task_name, robot, ep_idx, variant)
    z = zarr.open(str(_episode_zarr(task_name, robot, ep_idx, variant)), mode="r")
    T = int(z.attrs["total_frames"])
    extr = robotwin_extrinsics()
    print(
        f"[gt_dense_replay] building {T} native-rate ee16 waypoints from the zarr ..."
    )
    wps = []
    for t in range(T):
        le = np.asarray(z["left.obs_ee_pose"][t], np.float64).reshape(7)
        re_ = np.asarray(z["right.obs_ee_pose"][t], np.float64).reshape(7)
        lg = float(np.asarray(z["left.cmd_gripper"][t]).reshape(-1)[0])
        rg = float(np.asarray(z["right.cmd_gripper"][t]).reshape(-1)[0])
        a14 = world_endpose_to_cam14(le, re_, lg, rg, robot)
        wps.append(cam14_to_ee16(a14, extr)[0])

    args = _load_args(task_name, task_config, robot)
    args.setdefault("data_type", {})
    args["data_type"]["endpose"] = True
    if video_dir is not None:
        args["data_type"]["third_view"] = True
        os.makedirs(video_dir, exist_ok=True)
    TASK_ENV = _class_decorator(task_name)
    ok = False
    vid = None

    def restore_ik():  # no-op unless patch_mink_ik installs a real restorer
        pass

    stats = {"executed": 0, "ik_fails": 0}
    try:
        TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=seed, **args)
        if ik == "mink":
            restore_ik = patch_mink_ik(TASK_ENV)
        if video_dir is not None:
            vid = _start_video(
                TASK_ENV,
                os.path.join(
                    video_dir,
                    f"gtdense_{ik}_{robot}_{task_name}_{video_cam}_ep{ep_idx}_seed{seed}.mp4",
                ),
                cam=video_cam,
            )
        stats = execute_ee_dense(
            TASK_ENV,
            wps,
            steps_per_wp=steps_per_wp,
            max_joint_vel=max_joint_vel,
            record=vid is not None,
        )
        ok = bool(TASK_ENV.eval_success or TASK_ENV.check_success())
        if vid is not None:
            TASK_ENV._del_eval_video_ffmpeg()
    except Exception as e:
        import traceback

        print(
            f"[gt_dense_replay] ep{ep_idx} seed {seed}: EXCEPTION {type(e).__name__}: {e}"
        )
        traceback.print_exc()
    finally:
        restore_ik()
        try:
            TASK_ENV.close_env()
        except Exception:
            pass
    print(
        f"[gt_dense_replay] {task_name}/{robot} episode{ep_idx} (seed {seed}): "
        f"{'SUCCESS' if ok else 'FAIL'} executed={stats['executed']}/{T} "
        f"ik_fails={stats['ik_fails']} steps_per_wp={steps_per_wp} ik={ik}"
        + (f"  video={vid}" if vid else "")
    )
    return ok


def run_policy_train_ep(
    task_name,
    task_config,
    robot,
    ckpt,
    ep_idx,
    variant,
    exec_horizon=30,
    steps_per_wp=15,
    num_inference_steps=50,
    old_path=False,
    ik="mink",
    max_joint_vel=1.0,
    ensemble=False,
    ensemble_m=0.1,
    query_stride=10,
    budget_mult=3.0,
    video_dir=None,
    video_cam="third_view",
):
    """Gate D: closed-loop policy rollout on the training episode's exact scene. native (default):
    downsample-to-native chunk + dense streaming executor (mirrors the real-robot rollout), with
    ik="mink" (the real-robot EvaMinkKinematicsSolver, seeded from current joints) so
    replans/waypoints stay on-branch. old_path=True: the original take_action('ee')-per-upsampled-
    waypoint regime, for the A/B. ik="curobo" uses the global plan_single (jumpy) for comparison."""
    import zarr

    from egowam.scripts.robotwin_process.egowam_deploy import EgoWAMPolicy

    sys.path.append("./policy")
    import EgoWAM  # deploy shim: encode_obs

    seed = _episode_seed(task_name, robot, ep_idx, variant)
    z = zarr.open(str(_episode_zarr(task_name, robot, ep_idx, variant)), mode="r")
    T = int(z.attrs["total_frames"])

    model = EgoWAMPolicy(
        ckpt,
        robot=robot,
        num_inference_steps=num_inference_steps,
        exec_horizon=exec_horizon,
        native_rate=not old_path,
        ensemble=ensemble and not old_path,
        ensemble_m=ensemble_m,
        query_stride=query_stride,
    )
    ens_tag = f"_ens{query_stride}m{ensemble_m}" if (ensemble and not old_path) else ""
    tag = "old" if old_path else f"native{steps_per_wp}_{ik}{ens_tag}"

    args = _load_args(task_name, task_config, robot)
    args.setdefault("data_type", {})
    args["data_type"]["endpose"] = True
    if video_dir is not None:
        args["data_type"]["third_view"] = True
        os.makedirs(video_dir, exist_ok=True)
    TASK_ENV = _class_decorator(task_name)
    ok = False
    vid = None
    replans = 0

    def restore_ik():  # no-op unless patch_mink_ik installs a real restorer
        pass

    ik_n = {"n": 0}
    try:
        TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=seed, **args)
        if (
            ik == "mink" and not old_path
        ):  # real-robot mink IK for the native streaming executor
            restore_ik = patch_mink_ik(TASK_ENV)
        if old_path:  # dense path counts IK fails itself; old path needs the wrapper
            for side in ("left", "right"):
                name = f"{side}_plan_path"
                orig = getattr(TASK_ENV.robot, name)

                def _wrap(*a, _orig=orig, _s=side, **k):
                    r = _orig(*a, **k)
                    if isinstance(r, dict) and r.get("status") != "Success":
                        ik_n["n"] += 1
                        print(f"EGOWAM_IK_FAIL:{_s}", flush=True)
                    return r

                setattr(TASK_ENV.robot, name, _wrap)
        if video_dir is not None:
            vid = _start_video(
                TASK_ENV,
                os.path.join(
                    video_dir,
                    f"policytrain_{tag}_{robot}_{task_name}_{video_cam}_ep{ep_idx}_seed{seed}.mp4",
                ),
                cam=video_cam,
            )
        # native waypoints are ~60 ms of motion each -> budget_mult x episode length; the old
        # path's upsampled waypoints get production semantics (step_lim).
        limit = (
            TASK_ENV.step_lim
            if old_path
            else min(TASK_ENV.step_lim, int(budget_mult * T))
        )
        while TASK_ENV.take_action_cnt < limit and not TASK_ENV.eval_success:
            obs = EgoWAM.encode_obs(TASK_ENV.get_obs())
            wps = model.get_action(obs)
            replans += 1
            if old_path:
                for a in wps:
                    TASK_ENV.take_action(a, action_type="ee")
                    if vid is not None:
                        try:
                            TASK_ENV.get_obs()
                            TASK_ENV.eval_video_ffmpeg.stdin.write(
                                TASK_ENV._eval_video_frame_bytes()
                            )
                        except Exception:
                            pass
                    if TASK_ENV.eval_success:
                        break
            else:
                if os.environ.get("EGOWAM_DEBUG_JUMPS"):
                    print(
                        f"[REPLAN {replans}] new chunk of {len(wps)} native waypoints",
                        flush=True,
                    )
                st = execute_ee_dense(
                    TASK_ENV,
                    wps,
                    steps_per_wp=steps_per_wp,
                    max_joint_vel=max_joint_vel,
                    record=vid is not None,
                )
                ik_n["n"] += st["ik_fails"]
        ok = bool(TASK_ENV.eval_success or TASK_ENV.check_success())
        if vid is not None:
            TASK_ENV._del_eval_video_ffmpeg()
    except Exception as e:
        import traceback

        print(
            f"[policy_train_ep:{tag}] ep{ep_idx} seed {seed}: EXCEPTION {type(e).__name__}: {e}"
        )
        traceback.print_exc()
    finally:
        restore_ik()
        try:
            TASK_ENV.close_env()
        except Exception:
            pass
    print(
        f"[policy_train_ep:{tag}] {task_name}/{robot} episode{ep_idx} (seed {seed}): "
        f"{'SUCCESS' if ok else 'FAIL'} replans={replans} steps={TASK_ENV.take_action_cnt} "
        f"ik_fails={ik_n['n']} exec_horizon={exec_horizon} nis={num_inference_steps} "
        f"ik={'n/a' if old_path else ik}" + (f"  video={vid}" if vid else "")
    )
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="pick_diverse_bottles")
    p.add_argument("--task-config", default="demo_clean")
    p.add_argument("--robot", default="aloha-agilex")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument(
        "--eval-seed-file",
        type=str,
        default=None,
        help="Whitespace-separated seed list to evaluate verbatim (eval_seeds mode). Pins the "
        "evaluation set for reproducing a published table; overrides --seed-base/--n-target.",
    )
    p.add_argument(
        "--mode",
        choices=[
            "expert",
            "ee_path",
            "gt_replay",
            "policy",
            "obs_parity",
            "traj_replay",
            "gt_dense_replay",
            "policy_train_ep",
            "mink_selftest",
            "eval_seeds",
        ],
        default="expert",
    )
    p.add_argument(
        "--ckpt", default=None, help="required for --mode policy / policy_train_ep"
    )
    # Train-episode debug gates: episode index -> generation seed via the dataset's seed.txt.
    p.add_argument("--episode-idx", type=int, default=0, help="training episode index")
    p.add_argument(
        "--data-variant", default="clean_50", help="dataset variant folder suffix"
    )
    p.add_argument(
        "--exec-horizon",
        type=int,
        default=30,
        help="waypoints per replan (policy_train_ep)",
    )
    p.add_argument(
        "--steps-per-wp", type=int, default=15, help="sim steps per native waypoint"
    )
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument(
        "--max-joint-vel",
        type=float,
        default=1.0,
        help="dense executor joint-velocity cap (rad/s); spreads big moves over more steps "
        "(ts_offset analog). 0 disables.",
    )
    p.add_argument(
        "--old-path",
        action="store_true",
        help="policy_train_ep: original take_action('ee') A/B",
    )
    p.add_argument(
        "--ensemble",
        action="store_true",
        help="policy_train_ep: ACT temporal ensembling (re-query every --query-stride, average overlapping chunks)",
    )
    p.add_argument(
        "--ensemble-m",
        type=float,
        default=0.1,
        help="ensemble recency weight exp(-m*age)",
    )
    p.add_argument(
        "--query-stride",
        type=int,
        default=10,
        help="native steps between re-queries when ensembling",
    )
    # eval_seeds mode
    p.add_argument(
        "--results-path", default=None, help="eval_seeds: per-seed JSON log (resumable)"
    )
    p.add_argument(
        "--variant-tag", default="variant", help="eval_seeds: label for logs"
    )
    p.add_argument(
        "--n-target",
        type=int,
        default=50,
        help="eval_seeds: number of valid held-out seeds",
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=100000,
        help="eval_seeds: first candidate held-out seed",
    )
    p.add_argument(
        "--ik",
        choices=["mink", "curobo"],
        default="mink",
        help="IK for the dense executor: mink (real-robot EvaMinkKinematicsSolver, seeded/on-branch) "
        "or curobo (global plan_single per waypoint, jumpy)",
    )
    p.add_argument("--out-dir", default=None, help="obs_parity: dir for parity PNGs")
    p.add_argument("--video-dir", default=None, help="if set, save rollout mp4s here")
    # NOTE: this only sets the video-presentation camera. The policy's rollout observations always
    # come from head_camera (deploy_policy.encode_obs -> front_img_1); front_camera is a nicer view.
    p.add_argument(
        "--video-cam",
        default="third_view",
        help="presentation-video camera: third_view (3rd-person observer, default) | "
        "head_camera | front_camera | left_camera | right_camera",
    )
    args = p.parse_args()

    _setup_robotwin_path()
    from test_render import Sapien_TEST

    Sapien_TEST()  # render/vulkan check (raises if the GPU can't render)
    print(f"[sim_smoke] Sapien render check passed; mode={args.mode}")

    if args.mode == "expert":
        run_expert(
            args.task,
            args.task_config,
            args.robot,
            args.seeds,
            video_dir=args.video_dir,
            video_cam=args.video_cam,
        )
    elif args.mode == "ee_path":
        run_ee_path(args.task, args.task_config, args.robot, args.seeds)
    elif args.mode == "obs_parity":
        out = args.out_dir or "$EGOWAM_ROBOTWIN_ROOT/_sim_rollout/_debug/gateB"
        run_obs_parity(
            args.task,
            args.task_config,
            args.robot,
            args.episode_idx,
            args.data_variant,
            out,
        )
    elif args.mode == "mink_selftest":
        run_mink_selftest(
            args.task, args.task_config, args.robot, args.episode_idx, args.data_variant
        )
    elif args.mode == "traj_replay":
        run_traj_replay(
            args.task,
            args.task_config,
            args.robot,
            args.episode_idx,
            args.data_variant,
            video_dir=args.video_dir,
            video_cam=args.video_cam,
        )
    elif args.mode == "gt_dense_replay":
        run_gt_dense_replay(
            args.task,
            args.task_config,
            args.robot,
            args.episode_idx,
            args.data_variant,
            steps_per_wp=args.steps_per_wp,
            max_joint_vel=args.max_joint_vel,
            ik=args.ik,
            video_dir=args.video_dir,
            video_cam=args.video_cam,
        )
    elif args.mode == "policy_train_ep":
        if not args.ckpt:
            raise SystemExit("--mode policy_train_ep requires --ckpt")
        run_policy_train_ep(
            args.task,
            args.task_config,
            args.robot,
            args.ckpt,
            args.episode_idx,
            args.data_variant,
            exec_horizon=args.exec_horizon,
            steps_per_wp=args.steps_per_wp,
            max_joint_vel=args.max_joint_vel,
            num_inference_steps=args.num_inference_steps,
            old_path=args.old_path,
            ik=args.ik,
            ensemble=args.ensemble,
            ensemble_m=args.ensemble_m,
            query_stride=args.query_stride,
            video_dir=args.video_dir,
            video_cam=args.video_cam,
        )
    elif args.mode == "eval_seeds":
        if not args.ckpt or not args.results_path:
            raise SystemExit("--mode eval_seeds requires --ckpt and --results-path")
        run_eval_seeds(
            args.task,
            args.task_config,
            args.robot,
            args.ckpt,
            args.variant_tag,
            args.results_path,
            n_target=args.n_target,
            seed_base=args.seed_base,
            eval_seeds=_read_seed_file(args.eval_seed_file),
            exec_horizon=args.exec_horizon,
            steps_per_wp=args.steps_per_wp,
            num_inference_steps=args.num_inference_steps,
            max_joint_vel=args.max_joint_vel,
            ik=args.ik,
            ensemble=args.ensemble,
            ensemble_m=args.ensemble_m,
            query_stride=args.query_stride,
        )
    elif args.mode == "gt_replay":
        run_gt_replay(
            args.task,
            args.task_config,
            args.robot,
            args.seeds,
            video_dir=args.video_dir,
            video_cam=args.video_cam,
        )
    else:
        if not args.ckpt:
            raise SystemExit("--mode policy requires --ckpt")
        run_policy(
            args.task,
            args.task_config,
            args.robot,
            args.ckpt,
            args.seeds,
            video_dir=args.video_dir,
            video_cam=args.video_cam,
        )


if __name__ == "__main__":
    main()
