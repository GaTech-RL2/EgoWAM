#!/usr/bin/env python3
"""Extract the aloha-agilex (ARX5) 6-DoF arm sub-chain from RoboTwin's URDF into a kinematics-only
MuJoCo MJCF for mink IK.

Why: `model_x5.xml` (the real robot's Eva ARX5) has the right joint AXES but different LINK LENGTHS
than RoboTwin's aloha arm (mink_selftest: rotation exact, position off 10-100mm, config-dependent —
a constant TCP calibration can't absorb it). mink must run on the SIM arm's true kinematics. The
full URDF won't load in MuJoCo (mobile base, xacro `$(find ...)`, wrong meshdir), so we walk just
the `fl_joint1..6` chain and emit bodies/hinges with the URDF origins+axes (no meshes needed for IK).

Output names match `EvaMinkKinematicsSolver`'s expectations (base_link, joint1..6, site
tcp_match_trac) so it loads with zero solver code changes. Both arms share this chain (identical arm;
only the base mount differs, handled per-arm in MinkArmIK)."""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# RoboTwin's embodiment assets are downloaded into the vendored tree rather than committed (see
# external/README.md), so this resolves against the repo rather than an env var.
_REPO_ROOT = Path(__file__).resolve().parents[3]
URDF = str(
    _REPO_ROOT
    / "external/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
)
ARM_JOINTS = [f"fl_joint{i}" for i in range(1, 7)]  # fl_base_link -> ... -> fl_link6


def _rpy_to_quat_wxyz(rpy):
    r, p, y = [float(v) for v in rpy.split()]
    q = R.from_euler("xyz", [r, p, y]).as_quat()  # URDF rpy = extrinsic xyz; scipy xyzw
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz for MJCF


def build(urdf_path=URDF, out_path=None):
    if not os.path.exists(urdf_path):
        raise SystemExit(
            f"RoboTwin embodiment URDF not found: {urdf_path}\n"
            "These assets are downloaded, not vendored -- see external/README.md, or pass --urdf."
        )
    joints = {j.get("name"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
    # tcp offset: fl_joint7 (link6 -> gripper base) origin, so the site sits at the gripper attach.
    j7 = joints["fl_joint7"].find("origin").get("xyz")

    body_xml = []
    indent = "      "
    close = []
    for i, jn in enumerate(ARM_JOINTS, start=1):
        j = joints[jn]
        o = j.find("origin")
        xyz = o.get("xyz", "0 0 0")
        rpy = o.get("rpy", "0 0 0")
        ax = j.find("axis").get("xyz")
        q = _rpy_to_quat_wxyz(rpy)
        pad = indent * i
        body_xml.append(
            f'{pad}<body name="link{i}" pos="{xyz}" quat="{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}">\n'
            f'{pad}  <joint name="joint{i}" type="hinge" axis="{ax}" pos="0 0 0" range="-3.1416 3.1416"/>\n'
            f'{pad}  <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.01" density="100"/>\n'
        )
        close.append(f"{pad}</body>\n")
        if i == 6:
            body_xml.append(
                f'{pad}  <site name="tcp_match_trac" type="sphere" size="0.01" pos="{j7}"/>\n'
            )

    xml = (
        '<mujoco model="aloha_arm">\n'
        '  <compiler angle="radian" autolimits="true"/>\n'
        '  <option gravity="0 0 0"/>\n'
        "  <worldbody>\n"
        '    <body name="base_link" pos="0 0 0">\n'
        '      <geom type="box" size="0.03 0.03 0.03" density="100"/>\n'
        + "".join(body_xml)
        + "".join(reversed(close))
        + "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    if out_path:
        with open(out_path, "w") as f:
            f.write(xml)
        print(f"wrote {out_path}")
    return xml


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", default=URDF)
    # Expanded here, not left as "$EGOWAM_ROBOTWIN_ROOT/...": a "$VAR" inside a Python string is a
    # literal, so the unexpanded form wrote into a directory named "$EGOWAM_ROBOTWIN_ROOT".
    # Must stay in step with MODEL_ALOHA_ARM in egowam_mink.py, which reads this file.
    p.add_argument(
        "--out",
        default=os.path.join(
            os.environ.get(
                "EGOWAM_ROBOTWIN_ROOT", "/coc/flash7/scratch/egowm/robotwin"
            ),
            "_sim_rollout",
            "model_aloha_arm.xml",
        ),
    )
    a = p.parse_args()
    build(urdf_path=a.urdf, out_path=a.out)
