#!/usr/bin/env python3
"""EgoWAM -> RoboTwin sim deploy bridge (plan §5).

Loads an EgoVerse ``hpt_wm`` checkpoint (single arx or cross shared-head) and turns RoboTwin
observations into 14-dim camera-frame ``actions_cartesian`` predictions, then into RoboTwin's
**16-dim** ``ee`` action = ``[xyz(3) + quat_wxyz(4) + grip(1)] x 2``.

Model-side pipeline mirrors ``EvaOpenLoopPolicy.predict_cartesian_chunk``
used by the real-robot open-loop rollout, with RoboTwin specifics:
  * obs endpose is already xyz+wxyz **world** frame -> fed directly as ``{l,r}.obs_ee_pose``
    (no ``ee_pose_to_rot_ee_frame``; that is EVA-only).
  * transform list = ``Robotwin<Robot>.get_transform_list`` (extrinsics_key="robotwin", is_quat=True).
  * **NO ``rot_ee_frame_to_ee_pose_batch`` on output** — robotwin_to_zarr used identity tool
    rotation (no R_t_e baked at conversion), so cam->world alone yields the correct ee ypr.
  * sim RGB is already RGB and native 240x320 -> no channel swap, no resize.

Standalone validation (no SAPIEN):
    python -m egowam.scripts.robotwin_process.egowam_deploy --selftest \
        --episode $EGOWAM_ROBOTWIN_ROOT/zarrDataset/pick_diverse_bottles/arx-x5_clean_50/episode0.zarr \
        [--ckpt /path/to/last.ckpt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cosmos unpickling stub (VAE/RAE checkpoints): the classes must import before torch.load.
_repo_root = Path(__file__).resolve().parents[3]
_cosmos_root = _repo_root / "external" / "cosmos-policy"
if str(_cosmos_root) not in sys.path:
    sys.path.append(str(_cosmos_root))
# RAE checkpoints pickle-reference `stage1.RAE` (the RAE stage-1 decoder) — put external/RAE/src on
# the path so the RAE variant's ckpt unpickles under the sim env (else ModuleNotFoundError: stage1).
_rae_src = _repo_root / "external" / "RAE" / "src"
if _rae_src.is_dir() and str(_rae_src) not in sys.path:
    sys.path.append(str(_rae_src))
try:
    from egowam.models.cosmos_vae import _maybe_stub_megatron

    _maybe_stub_megatron()
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from egowam.compat import install_legacy_module_alias  # noqa: E402
from egowam.models.denoising_policy import DenoisingPolicy  # noqa: E402
from egowam.pl_utils.pl_model import ModelWrapper  # noqa: E402

# Checkpoints trained before the egomimic -> egowam rename pickle-reference `egomimic.*`, so
# torch.load fails with ModuleNotFoundError without this alias. trainHydra installs it for the
# training/validation path; the sim eval path reaches load_from_checkpoint without going through
# trainHydra, so it has to install it too.
install_legacy_module_alias()
from egowam.rldb.embodiment.embodiment import get_embodiment_id  # noqa: E402
from egowam.rldb.embodiment.robotwin import (  # noqa: E402
    RobotwinAloha,
    RobotwinArx,
    RobotwinFranka,
    RobotwinPiper,
    RobotwinUr5,
)
from egowam.utils.egowamUtils import (  # noqa: E402
    CameraTransforms,
    cam_frame_to_base_frame,
)
from egowam.utils.pose_utils import xyzw_to_wxyz  # noqa: E402

ROBOT_CLASS = {
    "aloha-agilex": RobotwinAloha,
    "arx-x5": RobotwinArx,
    "piper": RobotwinPiper,
    "ur5": RobotwinUr5,
    "franka": RobotwinFranka,
}
ROBOT_EMB = {
    "aloha-agilex": "robotwin_aloha_bimanual",
    "arx-x5": "robotwin_arx_bimanual",
    "piper": "robotwin_piper_bimanual",
    "ur5": "robotwin_ur5_bimanual",
    "franka": "robotwin_franka_bimanual",
}


def _to_chw_rgb(img_hwc: np.ndarray, device) -> torch.Tensor:
    """(H,W,3) uint8 RGB -> (3,H,W) float [0,1]. Sim RGB is already RGB (no BGR swap)."""
    t = (
        torch.from_numpy(np.ascontiguousarray(img_hwc)).to(device, dtype=torch.float32)
        / 255.0
    )
    return t.permute(2, 0, 1)


# ---- conversion primitives (checkpoint-free — reused by the GT-replay sanity gate) ----------
def robotwin_extrinsics(extrinsics_key: str = "robotwin") -> dict:
    """Per-arm head-camera extrinsics (RoboTwin head cam is static & shared across arms)."""
    return CameraTransforms(
        intrinsics_key="robotwin", extrinsics_key=extrinsics_key
    ).extrinsics


def cam14_to_ee16(a14: np.ndarray, extrinsics: dict) -> np.ndarray:
    """14-dim CAM-frame ``[L xyz+ypr+grip, R xyz+ypr+grip]`` -> 16-dim world ``ee`` action
    ``[xyz + quat_wxyz + grip] x 2``. This is the exact conversion the learned policy applies at
    inference (cam->base + euler(ZYX)->quat_wxyz + gripper clamp); factored out so the GT-replay
    gate exercises the identical path without loading a checkpoint."""
    a14 = np.asarray(a14, dtype=np.float64)
    if a14.ndim == 1:
        a14 = a14[None]
    left, right = a14[:, :7], a14[:, 7:]
    out = []
    for arm, side in ((left, "left"), (right, "right")):
        world = cam_frame_to_base_frame(
            arm[:, :6].copy(), extrinsics[side]
        )  # (H,6) xyz+ypr
        # RoboTwin used identity tool rotation at conversion -> NO rot_ee_frame_to_ee_pose_batch.
        quat_wxyz = xyzw_to_wxyz(R.from_euler("ZYX", world[:, 3:6]).as_quat())  # (H,4)
        # Gripper is physically bounded [0,1] (1=open); clamp defends against out-of-range values.
        grip = np.clip(arm[:, 6:7], 0.0, 1.0)
        out.append(np.hstack([world[:, :3], quat_wxyz, grip]))  # (H,8)
    return np.hstack(out).astype(np.float32)  # (H,16)


def world14_to_ee16(a14: np.ndarray) -> np.ndarray:
    """14-dim WORLD-frame ``[L xyz+ypr+grip, R xyz+ypr+grip]`` -> 16-dim RoboTwin ``ee`` action
    ``[xyz + quat_wxyz + grip] x 2``. Same as :func:`cam14_to_ee16` but the input xyz+ypr is ALREADY
    in the world/base frame (no `cam_frame_to_base_frame`) — used by the EE-Cartesian ACT/DP baselines
    whose action labels are world-frame endpose (xyz + ZYX-euler), not camera-frame."""
    a14 = np.asarray(a14, dtype=np.float64)
    if a14.ndim == 1:
        a14 = a14[None]
    out = []
    for arm in (a14[:, :7], a14[:, 7:]):
        quat_wxyz = xyzw_to_wxyz(R.from_euler("ZYX", arm[:, 3:6]).as_quat())  # (H,4)
        grip = np.clip(arm[:, 6:7], 0.0, 1.0)
        out.append(np.hstack([arm[:, :3], quat_wxyz, grip]))  # (H,8)
    return np.hstack(out).astype(np.float32)  # (H,16)


def endpose_to_world14(le, re, lg, rg) -> np.ndarray:
    """RoboTwin obs endpose (world xyz+wxyz per arm) + grippers -> 14-dim world EE state
    ``[L xyz+ypr+grip, R xyz+ypr+grip]`` (the EE-Cartesian baselines' state/qpos input)."""
    out = []
    for ee, g in ((le, lg), (re, rg)):
        ee = np.asarray(ee, np.float64).reshape(7)
        w, x, y, z = ee[3], ee[4], ee[5], ee[6]  # endpose quat is wxyz
        ypr = R.from_quat([x, y, z, w]).as_euler("ZYX")
        out.extend([ee[0], ee[1], ee[2], ypr[0], ypr[1], ypr[2], float(g)])
    return np.asarray(out, np.float64)  # (14,)


def downsample_cam14_chunk(chunk: np.ndarray, n: int = 45) -> np.ndarray:
    """(T,14) cam-frame ``[L xyz+ypr+grip, R xyz+ypr+grip]`` -> (n,14) at the data's native rate.

    Port of the real-robot rollout's ``PolicyRollout._downsample_chunk`` (egowam/robot/
    rollout.py:278): per arm, xyz linear + angle-unwrapped euler interpolation + linear gripper via
    :func:`interpolate_arr_euler` ((B,T,7) form). The model predicts the InterpolatePose-upsampled
    100-chunk; executing it raw gives ~27 ms-apart waypoints — this restores the recorded ~60 ms
    (16.67 Hz) spacing the sim's data was collected at, mirroring the real robot's 30 Hz streaming
    of native-rate targets."""
    from egowam.utils.egowamUtils import interpolate_arr_euler

    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.shape[0] == n:
        return chunk
    left = interpolate_arr_euler(chunk[None, :, :7], n)[0]  # (n,7) xyz+ypr+grip
    right = interpolate_arr_euler(chunk[None, :, 7:14], n)[0]
    return np.hstack([left, right])  # (n,14)


def world_endpose_to_cam14(
    le, re, lg, rg, robot: str, chunk_length: int = 100, action_horizon: int = 45
) -> np.ndarray:
    """Recorded WORLD-frame bimanual endpose -> 14-dim CAM-frame ``actions_cartesian`` (one step).

    Runs the SAME embodiment transform list the training/deploy encode uses, so feeding the result
    through :func:`cam14_to_ee16` reproduces the full world->cam->world conversion round-trip on GT
    data. ``le``/``re``: (7,) xyz+wxyz world; ``lg``/``rg``: float gripper. Returns (14,)."""
    emb_cls = ROBOT_CLASS[robot]
    emb_name = ROBOT_EMB[robot]
    data = {
        emb_cls.VIZ_IMAGE_KEY: torch.zeros(3, 240, 320),
        "observations.images.right_wrist_img": torch.zeros(3, 240, 320),
        "observations.images.left_wrist_img": torch.zeros(3, 240, 320),
        "pad_mask": torch.ones((1, chunk_length, 1), dtype=torch.bool),
    }
    for side, ee, g in (("left", le, lg), ("right", re, rg)):
        ee = np.asarray(ee, dtype=np.float64).reshape(7)  # xyz+wxyz world
        data[f"{side}.obs_ee_pose"] = torch.from_numpy(ee)
        data[f"{side}.obs_gripper"] = torch.tensor([float(g)], dtype=torch.float64)
        data[f"{side}.cmd_ee_pose"] = (
            torch.from_numpy(ee).view(1, 7).repeat(action_horizon, 1)
        )
        data[f"{side}.gripper"] = (
            torch.tensor(float(g)).view(1, 1).repeat(action_horizon, 1)
        )
    data["embodiment"] = [emb_name]
    data["metadata.robot_name"] = [emb_name]
    for tf in emb_cls.get_transform_list(chunk_length=chunk_length, use_future=False):
        data = tf.transform(data)
    ac = np.asarray(data["actions_cartesian"])  # (chunk_length,14) cam-frame
    return ac[0].astype(np.float64)  # (14,)


class EgoWAMPolicy:
    """hpt_wm checkpoint -> RoboTwin 16-dim ``ee`` action chunks."""

    def __init__(
        self,
        ckpt_path: str,
        robot: str = "aloha-agilex",
        *,
        extrinsics_key: str = "robotwin",
        num_inference_steps: int = 10,
        action_horizon: int = 45,
        chunk_length: int = 100,
        exec_horizon: int = 30,
        native_rate: bool = False,
        ensemble: bool = False,
        ensemble_m: float = 0.1,
        query_stride: int | None = None,
    ) -> None:
        self.robot = robot
        self.emb_cls = ROBOT_CLASS[robot]
        self.embodiment_name = ROBOT_EMB[robot]
        self.embodiment_id = get_embodiment_id(self.embodiment_name.upper())
        self.action_horizon = action_horizon
        self.chunk_length = chunk_length
        self.exec_horizon = exec_horizon
        # native_rate: downsample the InterpolatePose-upsampled chunk (100) back to the recorded
        # action_horizon (45, ~60 ms spacing) before execution — mirrors the real-robot rollout
        # (_downsample_chunk in rollout.py). exec_horizon then counts NATIVE-rate waypoints.
        self.native_rate = native_rate
        # ACT-style temporal ensembling: re-query every `query_stride` native steps, keep a buffer
        # of the overlapping predicted chunks, and execute the recency-weighted average per timestep
        # (weight = exp(-ensemble_m * age), age 0 = newest). Averages out diffusion sample variance
        # and blends across replans -> fixes both boundary discontinuities and compounding drift.
        self.ensemble = ensemble
        self.ensemble_m = ensemble_m
        self.query_stride = query_stride if query_stride else exec_horizon

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = ModelWrapper.load_from_checkpoint(
            ckpt_path, weights_only=False, map_location=torch.device("cpu")
        )
        # Speed up diffusion heads for closed-loop control.
        if getattr(self.policy.model, "diffusion", False):
            for _name, head in self.policy.model.nets.policy.heads.items():
                if isinstance(head, DenoisingPolicy):
                    head.num_inference_steps = num_inference_steps
        self.policy.to(self.device)
        cv = getattr(self.policy.model, "cosmos_vae", None)  # not in nn.Module tree
        if cv is not None:
            cv.to(self.device)
        self.policy.eval()

        # RoboTwin head cam is static & shared across arms; both arms use the same extrinsic.
        self.extrinsics = robotwin_extrinsics(extrinsics_key)
        self._transforms = self.emb_cls.get_transform_list(
            chunk_length=chunk_length, use_future=False
        )
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self._buf = []  # list of [start_t, chunk_native (H,14) cam-frame] for temporal ensembling
        self.global_t = 0

    # ---- observation -> model batch --------------------------------------------------------
    def _build_batch(self, obs: dict) -> dict:
        """obs keys: front_img_1,left_wrist_img,right_wrist_img (H,W,3 uint8 RGB);
        left_endpose,right_endpose (7: xyz+wxyz world); left_gripper,right_gripper (float)."""
        dev = self.device
        H = self.action_horizon
        data = {
            self.emb_cls.VIZ_IMAGE_KEY: _to_chw_rgb(obs["front_img_1"], dev),
            "observations.images.right_wrist_img": _to_chw_rgb(
                obs["right_wrist_img"], dev
            ),
            "observations.images.left_wrist_img": _to_chw_rgb(
                obs["left_wrist_img"], dev
            ),
            "pad_mask": torch.ones(
                (1, self.chunk_length, 1), device=dev, dtype=torch.bool
            ),
        }
        for side in ("left", "right"):
            ee = np.asarray(obs[f"{side}_endpose"], dtype=np.float64).reshape(
                7
            )  # xyz+wxyz world
            grip = float(obs[f"{side}_gripper"])
            data[f"{side}.obs_ee_pose"] = torch.from_numpy(ee)
            data[f"{side}.obs_gripper"] = torch.tensor([grip], dtype=torch.float64)
            # Action-key placeholders (values irrelevant at inference; shapes drive the pipeline).
            data[f"{side}.cmd_ee_pose"] = torch.from_numpy(ee).view(1, 7).repeat(H, 1)
            data[f"{side}.gripper"] = torch.tensor(grip).view(1, 1).repeat(H, 1)
        data["embodiment"] = [self.embodiment_name]
        data["metadata.robot_name"] = [self.embodiment_name]

        for tf in self._transforms:
            data = tf.transform(data)
        for k, v in list(data.items()):
            if torch.is_tensor(v):
                data[k] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                data[k] = v[None, ...]
        return {self.embodiment_name: data}

    # ---- forward + 14-dim cam ypr -> 16-dim world ee ---------------------------------------
    def predict_cam14(self, obs: dict) -> np.ndarray:
        """One diffusion sample of the (chunk_length,14) CAM-frame actions_cartesian chunk.

        Action-only inference (rollout_2.py): ``attach_future_targets=False`` +
        ``include_aux_predictions=False`` disables the world-model head — the VAE/RAE/3D-flow
        future decoders aren't needed for the action chunk and would otherwise KeyError on the
        absent future camera/proprio/flow keys. No-op for the HPT (no-WM) baseline."""
        batch = self._build_batch(obs)
        try:
            processed = self.policy.model.process_batch_for_training(
                batch, attach_future_targets=False
            )
        except TypeError:  # older signature without the kwarg
            processed = self.policy.model.process_batch_for_training(batch)
        with torch.no_grad():
            preds_all = self.policy.model.forward_eval(
                processed, include_aux_predictions=False
            )
        preds = preds_all[f"{self.embodiment_name}_actions_cartesian"]
        return (
            preds.detach().cpu().numpy().squeeze()
        )  # (H,14): [L xyz+ypr+grip, R xyz+ypr+grip], cam frame

    def predict_chunk(self, obs: dict) -> np.ndarray:
        return self._cam14_to_ee16(self.predict_cam14(obs))

    def _cam14_to_ee16(self, a14: np.ndarray) -> np.ndarray:
        return cam14_to_ee16(a14, self.extrinsics)

    def get_action(self, obs: dict) -> np.ndarray:
        """Return the next batch of 16-dim ee waypoints to execute (receding horizon).

        native_rate=True: the upsampled chunk (100) is first downsampled back to the recorded
        rate (action_horizon=45, ~60 ms spacing) — the executed waypoints are then native-rate,
        mirroring the real-robot rollout's resample-then-stream loop.

        ensemble=True: predicts a fresh chunk, appends it to the overlap buffer, and returns the
        next ``query_stride`` timesteps as the recency-weighted average of all buffered chunks that
        cover each timestep (ACT temporal ensembling). Otherwise returns the first ``exec_horizon``
        waypoints of this chunk directly."""
        cam14 = self.predict_cam14(obs)
        if self.native_rate:
            cam14 = downsample_cam14_chunk(cam14, self.action_horizon)
        self.t += 1
        if not self.ensemble:
            return self._cam14_to_ee16(cam14)[: self.exec_horizon]

        H = cam14.shape[0]
        self._buf.append([self.global_t, cam14])
        self._buf = [
            b for b in self._buf if b[0] + H > self.global_t
        ]  # drop expired chunks
        out14 = []
        for tau in range(self.query_stride):
            t = self.global_t + tau
            preds, weights = [], []
            for age, (start_t, chunk) in enumerate(
                reversed(self._buf)
            ):  # age 0 = newest
                off = t - start_t
                if 0 <= off < H:
                    preds.append(chunk[off])
                    weights.append(np.exp(-self.ensemble_m * age))
            if not preds:
                break
            w = np.asarray(weights) / np.sum(weights)
            # xyz+ypr+grip are close across overlapping chunks (that's the premise) -> linear mean;
            # cam-frame ypr stays well inside (-pi,pi) for these tasks, so no wrap handling needed.
            out14.append(np.average(np.stack(preds), axis=0, weights=w))
        self.global_t += self.query_stride
        return self._cam14_to_ee16(np.stack(out14))


# ---- standalone validation (no SAPIEN) -----------------------------------------------------
def _selftest(episode: str, ckpt: str | None, robot: str) -> None:
    import cv2
    import zarr

    emb_cls = ROBOT_CLASS[robot]
    z = zarr.open(episode, mode="r")

    def _frame(key, t):
        raw = z[key][t]  # nested 0-d object arrays -> ... -> JPEG bytes
        while isinstance(raw, np.ndarray) and raw.ndim == 0:
            raw = raw.item()
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ---- Round-trip: world endpose -> transform (cam frame actions_cartesian) -> inverse -> world.
    extr = CameraTransforms(
        intrinsics_key="robotwin", extrinsics_key="robotwin"
    ).extrinsics
    t0 = 0
    le = np.asarray(z["left.obs_ee_pose"][t0], np.float64)  # xyz+wxyz world
    re = np.asarray(z["right.obs_ee_pose"][t0], np.float64)
    data = {
        emb_cls.VIZ_IMAGE_KEY: torch.zeros(3, 240, 320),
        "observations.images.right_wrist_img": torch.zeros(3, 240, 320),
        "observations.images.left_wrist_img": torch.zeros(3, 240, 320),
        "pad_mask": torch.ones((1, 100, 1), dtype=torch.bool),
    }
    for side, ee in (("left", le), ("right", re)):
        data[f"{side}.obs_ee_pose"] = torch.from_numpy(ee)
        _g = float(np.asarray(z[f"{side}.obs_gripper"][t0]).reshape(-1)[0])
        data[f"{side}.obs_gripper"] = torch.tensor([_g], dtype=torch.float64)
        data[f"{side}.cmd_ee_pose"] = torch.from_numpy(ee).view(1, 7).repeat(45, 1)
        data[f"{side}.gripper"] = torch.tensor(_g).view(1, 1).repeat(45, 1)
    data["embodiment"] = [ROBOT_EMB[robot]]
    data["metadata.robot_name"] = [ROBOT_EMB[robot]]
    for tf in emb_cls.get_transform_list(chunk_length=100, use_future=False):
        data = tf.transform(data)
    ac = np.asarray(
        data["actions_cartesian"]
    )  # (100,14) cam-frame [L xyz+ypr+grip, R ...]

    def inv(arm, side):
        w = cam_frame_to_base_frame(arm[None, :6].copy(), extr[side])[0]
        q = xyzw_to_wxyz(R.from_euler("ZYX", w[3:6]).as_quat())
        return np.concatenate([w[:3], q])

    for name, src, arm in (("left", le, ac[0, :7]), ("right", re, ac[0, 7:])):
        got = inv(arm, name)
        dxyz = np.abs(src[:3] - got[:3]).max()
        qdot = abs(float(np.dot(src[3:7], got[3:7])))
        print(
            f"[roundtrip {name}] max|dxyz|={dxyz:.4e}  |quat.dot|={qdot:.6f}  (expect ~0 and ~1)"
        )

    if ckpt:
        pol = EgoWAMPolicy(ckpt, robot=robot)
        obs = {
            "front_img_1": _frame("images.front_1", t0),
            "left_wrist_img": _frame("images.left_wrist", t0),
            "right_wrist_img": _frame("images.right_wrist", t0),
            "left_endpose": le,
            "right_endpose": re,
            "left_gripper": float(np.asarray(z["left.obs_gripper"][t0]).reshape(-1)[0]),
            "right_gripper": float(
                np.asarray(z["right.obs_gripper"][t0]).reshape(-1)[0]
            ),
        }
        chunk = pol.get_action(obs)
        assert chunk.ndim == 2 and chunk.shape[1] == 16, chunk.shape
        qn_l = np.linalg.norm(chunk[:, 3:7], axis=1)
        qn_r = np.linalg.norm(chunk[:, 11:15], axis=1)
        print(
            f"[ckpt] action chunk shape={chunk.shape} finite={bool(np.isfinite(chunk).all())} "
            f"quat_norm L~{qn_l.mean():.4f} R~{qn_r.mean():.4f} "
            f"grip L[{chunk[:,7].min():.2f},{chunk[:,7].max():.2f}] R[{chunk[:,15].min():.2f},{chunk[:,15].max():.2f}]"
        )
        print(
            "[ckpt] OK — model-side deploy pipeline produces valid 16-dim ee actions."
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--episode", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument(
        "--robot", type=str, default="aloha-agilex", choices=list(ROBOT_CLASS)
    )
    args = p.parse_args()
    if args.selftest:
        _selftest(args.episode, args.ckpt, args.robot)


if __name__ == "__main__":
    main()
