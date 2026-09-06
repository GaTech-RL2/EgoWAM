#!/usr/bin/env python3
"""Mink IK for the RoboTwin sim executor — reproduce the REAL-WORLD IK path.

The physical rollout (`egowam/robot/rollout.py`) turns each predicted EE pose into joint commands
with ``EvaMinkKinematicsSolver`` (mink, MuJoCo ``model_x5.xml``, seeded from the current joints).
Using the SAME solver + model in sim (a) makes IK feasibility/branch behavior match hardware, and
(b) sidesteps curobo's ``plan_single``/``ik_solver`` CUDA-graph incompatibility on this cluster
(``changing goal type, cuda graph reset not available``).

RoboTwin's aloha-agilex arms are ARX5, the same arm ``model_x5.xml`` describes, so one solver serves
both arms; only the per-arm base transform (and a fixed TCP offset vs RoboTwin's endlink) differ.
Both are auto-calibrated from the sim at setup via an FK round-trip (`calibrate`), so we never
hand-guess frames — if calibration residual is large, the caller aborts rather than execute garbage.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from egowam.robot.eva.eva_kinematics import EvaMinkKinematicsSolver

# model_x5.xml (real robot's Eva ARX5) lives at repo_root/egowam/resources/. It has the right
# joint axes but different LINK LENGTHS than RoboTwin's aloha arm, so for the SIM we use an MJCF
# extracted from RoboTwin's own aloha URDF (build_aloha_arm_mjcf.py) — same solver, correct lengths.
_REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_X5 = str(_REPO_ROOT / "egowam" / "resources" / "model_x5.xml")
# Built by build_aloha_arm_mjcf.py. Expanded here: a "$VAR" inside a Python string is a
# literal, so the unexpanded form opened a directory named "$EGOWAM_ROBOTWIN_ROOT".
MODEL_ALOHA_ARM = os.path.join(
    os.environ.get("EGOWAM_ROBOTWIN_ROOT", "/coc/flash7/scratch/egowm/robotwin"),
    "_sim_rollout",
    "model_aloha_arm.xml",
)


def _pose_to_Rt(p, q_wxyz):
    """(xyz, wxyz quat) -> (3x3 R, 3 t)."""
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    return R.from_quat(q_xyzw).as_matrix(), np.asarray(p, dtype=np.float64)


class MinkArmIK:
    """Per-arm mink IK wrapper reproducing the real-robot solver.

    Targets are given in the SAPIEN WORLD frame (xyz + wxyz, same as RoboTwin ``ee`` actions); this
    class transforms them into the model's base frame using each arm's base pose and a calibrated
    world->tcp offset, then runs the same ``EvaMinkKinematicsSolver.ik`` the hardware uses."""

    def __init__(self, model_path: str = MODEL_ALOHA_ARM):
        # One solver instance per arm (mink keeps internal MjData state; do not share across arms).
        self.solver = {
            "left": EvaMinkKinematicsSolver(model_path=model_path),
            "right": EvaMinkKinematicsSolver(model_path=model_path),
        }
        self.T_wb = {}  # arm base -> world (4x4), per arm
        self.T_tcp = {}  # fixed correction: model tcp frame -> RoboTwin endpose frame (4x4), per arm
        self.sign = {
            "left": np.ones(6),
            "right": np.ones(6),
        }  # sim qpos -> model joint sign
        self.calib_residual = {}

    @staticmethod
    def _Rt_to_T(Rm, t):
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = t
        return T

    def _fk_tcp(self, side, joints6):
        """mink FK: 6 joints -> tcp pose in the model BASE frame (3x3 R, 3 t). Applies the
        per-joint sign that maps SAPIEN qpos -> the MJCF joint convention."""
        q = self.sign[side] * np.asarray(joints6, np.float64)[:6]
        pos, rot = self.solver[side].fk(q)
        return rot.as_matrix(), np.asarray(pos, np.float64)

    def calibrate_signs(self, side, base_pose_world, samples):
        """Resolve the SAPIEN-qpos -> MJCF-axis sign per joint (the FK is exact vs the URDF, but the
        sim's joint sign convention may differ: error is 0 at q=0 and grows with |q|). ``samples`` =
        list of (joints6, endpose_world); the first is the calibration anchor. Brute-forces the 64
        sign patterns, picks the one minimizing FK-vs-endpose residual over the rest. Sets
        ``self.sign[side]`` and returns the winning mean residual (m)."""
        import itertools

        j0, ep0 = samples[0]
        best_sign, best_res = np.ones(6), np.inf
        for combo in itertools.product((1.0, -1.0), repeat=6):
            self.sign[side] = np.array(combo)
            self.calibrate(
                side, base_pose_world, j0, ep0
            )  # T_tcp under this sign hypothesis
            # Score BOTH position and orientation: wrist joints (near the TCP axis) barely move the
            # position, so a position-only objective can't fix their sign — include rotation.
            errs = [self.verify_fk(side, j, ep) for j, ep in samples[1:]]
            m = float(np.mean([dp + 0.1 * dr for dp, dr in errs])) if errs else 0.0
            if m < best_res:
                best_res, best_sign = m, np.array(combo)
        self.sign[side] = best_sign
        self.calibrate(
            side, base_pose_world, j0, ep0
        )  # final T_tcp under the winning sign
        return best_res

    def calibrate(self, side, base_pose_world, joints6, endpose_world):
        """Pin the two unknown frames from one (joints, achieved endpose) sample:
          * ``T_wb``  = arm base pose in world  (from RoboTwin's per-arm origin pose)
          * ``T_tcp`` = model-tcp -> RoboTwin-endpose fixed offset, so that
                        world_endpose == T_wb @ FK_tcp(joints) @ T_tcp.
        Returns the residual position error (m) of the fit (0 by construction; a sanity echo)."""
        Rb, tb = _pose_to_Rt(base_pose_world[:3], base_pose_world[3:7])
        self.T_wb[side] = self._Rt_to_T(Rb, tb)
        R_tcp_b, t_tcp_b = self._fk_tcp(side, joints6)
        T_b_tcp = self._Rt_to_T(R_tcp_b, t_tcp_b)  # tcp in base
        Re, te = _pose_to_Rt(endpose_world[:3], endpose_world[3:7])
        T_w_ep = self._Rt_to_T(Re, te)  # endpose in world
        # world_endpose = T_wb @ T_b_tcp @ T_tcp  ->  T_tcp = inv(T_wb @ T_b_tcp) @ T_w_ep
        self.T_tcp[side] = np.linalg.inv(self.T_wb[side] @ T_b_tcp) @ T_w_ep
        self.calib_residual[side] = 0.0
        return 0.0

    def verify_fk(self, side, joints6, endpose_world):
        """Independent check: predicted world endpose from FK+calibration vs the actual one."""
        R_tcp_b, t_tcp_b = self._fk_tcp(side, joints6)
        T_pred = self.T_wb[side] @ self._Rt_to_T(R_tcp_b, t_tcp_b) @ self.T_tcp[side]
        Re, te = _pose_to_Rt(endpose_world[:3], endpose_world[3:7])
        dpos = float(np.linalg.norm(T_pred[:3, 3] - te))
        drot = float(np.linalg.norm(R.from_matrix(T_pred[:3, :3] @ Re.T).as_rotvec()))
        return dpos, drot

    def ik_world(self, side, ee_world_pose, cur_jnts6):
        """World-frame ee target (xyz+wxyz) -> 6 joint angles, seeded from cur_jnts6 (local IK).

        Inverts the calibration: model-tcp target in base = inv(T_wb) @ world_endpose @ inv(T_tcp)."""
        Re, te = _pose_to_Rt(ee_world_pose[:3], ee_world_pose[3:7])
        T_w_ep = self._Rt_to_T(Re, te)
        T_b_tcp = (
            np.linalg.inv(self.T_wb[side]) @ T_w_ep @ np.linalg.inv(self.T_tcp[side])
        )
        # Seed with the CURRENT joints in MODEL convention; un-map the solution back to SAPIEN qpos.
        seed = self.sign[side] * np.asarray(cur_jnts6, np.float64)[:6]
        sol = self.solver[side].ik(T_b_tcp[:3, 3], T_b_tcp[:3, :3], seed)
        return (
            None if sol is None else self.sign[side] * np.asarray(sol, np.float64)[:6]
        )
