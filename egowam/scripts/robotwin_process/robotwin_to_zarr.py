"""Convert RoboTwin 2.0 HDF5 episodes to EgoVerse Zarr.

RoboTwin 2.0 clean_50/randomized_500 archives unpack to
``<robot>_<variant>/data/episodeN.hdf5`` (observations), plus
``instructions/episodeN.json`` (language) and ``scene_info.json``.

HDF5 schema (verified on pick_diverse_bottles):
    endpose/{left,right}_endpose      (T,7)  float64  xyz + wxyz  (SAPIEN world frame)
    endpose/{left,right}_gripper      (T,)   float64  1.0=open
    joint_action/{left,right}_arm     (T,6)  float64  (diagnostic)
    observation/{head,front,left,right}_camera/rgb  (T,)  JPEG bytes (240x320)
    observation/*/{intrinsic_cv,extrinsic_cv,cam2world_gl}   per-frame calibration

We store poses in world frame (xyz+wxyz); the RoboTwin embodiment transform expresses
them relative to the head camera at load time. Camera mapping: head->front_1,
left->left_wrist, right->right_wrist (front_camera unused; EgoVerse uses 3 views).

Usage (one robot folder -> one zarr dataset dir):
    python -m egowam.scripts.robotwin_process.robotwin_to_zarr \
        --robot-dir $EGOWAM_ROBOTWIN_ROOT/dataset/pick_diverse_bottles/arx-x5_clean_50 \
        --robot arx-x5 --task pick_diverse_bottles \
        --output-root $EGOWAM_ROBOTWIN_ROOT/zarrDataset [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np

from egowam.rldb.zarr.zarr_writer import ZarrWriter

logger = logging.getLogger(__name__)

# RoboTwin robot id -> EgoVerse embodiment string (must match EMBODIMENT enum).
ROBOT_TO_EMBODIMENT = {
    "aloha-agilex": "robotwin_aloha_bimanual",
    "arx-x5": "robotwin_arx_bimanual",
    "piper": "robotwin_piper_bimanual",
    "ur5": "robotwin_ur5_bimanual",
    "franka": "robotwin_franka_bimanual",
}

# RoboTwin camera name -> EgoVerse zarr image key.
CAMERA_MAP = {
    "head_camera": "images.front_1",
    "left_camera": "images.left_wrist",
    "right_camera": "images.right_wrist",
}


def _decode_rgb(dset) -> np.ndarray:
    """Decode a (T,) array of per-frame JPEG bytes to (T,240,320,3) uint8 RGB.

    Kept at RoboTwin's native 240x320: it is identical across all robots (self-consistent,
    no standardization needed) and the VAE/RAE encoders resize internally, so upsampling
    would only add interpolation cost with no information gain. INTRINSICS["robotwin"] is
    defined for this native 240x320 resolution.
    """
    frames = []
    for t in range(dset.shape[0]):
        buf = np.frombuffer(dset[t], dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # cv2 decodes to BGR
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0).astype(np.uint8)


def _read_instruction(instr_path: Path) -> str:
    if not instr_path.exists():
        return ""
    try:
        obj = json.loads(instr_path.read_text())
        for key in ("seen", "unseen"):
            vals = obj.get(key)
            if isinstance(vals, list) and vals:
                return str(vals[0])
    except Exception:
        pass
    return ""


def convert_episode(
    hdf5_path: Path,
    output_dir: Path,
    dataset_name: str,
    robot: str,
    fps: int = 25,
    task_name: str = "",
    task_description: str = "",
    chunk_timesteps: int = 100,
) -> Path:
    embodiment = ROBOT_TO_EMBODIMENT[robot]
    with h5py.File(hdf5_path, "r") as h:
        numeric_data: dict[str, np.ndarray] = {}
        for arm in ("left", "right"):
            # Achieved EE pose (xyz + wxyz quat, SAPIEN world frame).
            ee = np.asarray(h[f"endpose/{arm}_endpose"], dtype=np.float64)  # (T,7)
            # obs/cmd split mirrors eva_to_zarr: obs from achieved state, cmd from the
            # commanded action stream. RoboTwin: endpose/* = achieved, joint_action/* = commanded.
            obs_grip = np.asarray(
                h[f"endpose/{arm}_gripper"], dtype=np.float32
            ).reshape(-1, 1)
            cmd_grip = np.asarray(
                h[f"joint_action/{arm}_gripper"], dtype=np.float32
            ).reshape(-1, 1)
            joints = np.asarray(
                h[f"joint_action/{arm}_arm"], dtype=np.float32
            )  # (T,6) commanded

            numeric_data[f"{arm}.obs_ee_pose"] = ee
            # RoboTwin records no commanded *endpose*; the EE-Cartesian action target is the
            # achieved future-EE chunk (same formulation as Aria's observed-future actions).
            numeric_data[f"{arm}.cmd_ee_pose"] = ee.copy()
            numeric_data[f"{arm}.obs_gripper"] = obs_grip
            numeric_data[f"{arm}.cmd_gripper"] = cmd_grip
            numeric_data[f"{arm}.obs_joints"] = joints
            numeric_data[f"{arm}.cmd_joints"] = (
                joints.copy()
            )  # RoboTwin records commanded joints only

        image_data: dict[str, np.ndarray] = {}
        for cam, zarr_key in CAMERA_MAP.items():
            image_data[zarr_key] = _decode_rgb(h[f"observation/{cam}/rgb"])

    zarr_path = ZarrWriter.create_and_write(
        episode_path=output_dir / f"{dataset_name}.zarr",
        numeric_data=numeric_data,
        image_data=image_data,
        embodiment=embodiment,
        fps=fps,
        task_name=task_name,
        task_description=task_description,
        chunk_timesteps=chunk_timesteps,
    )
    logger.info("Wrote %s", zarr_path)
    return zarr_path


def convert_robot_folder(
    robot_dir: Path,
    robot: str,
    task: str,
    output_root: Path,
    fps: int = 25,
    limit: int | None = None,
) -> list[Path]:
    data_dir = robot_dir / "data"
    instr_dir = robot_dir / "instructions"
    eps = sorted(
        data_dir.glob("episode*.hdf5"),
        key=lambda p: int(p.stem.replace("episode", "")),
    )
    if limit is not None:
        eps = eps[:limit]

    # Mirror the original RoboTwin dataset layout: <task>/<robot>_<variant>/episodeN.zarr
    out_dir = output_root / task / robot_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for hdf5_path in eps:
        idx = int(hdf5_path.stem.replace("episode", ""))
        desc = _read_instruction(instr_dir / f"episode{idx}.json")
        try:
            zp = convert_episode(
                hdf5_path=hdf5_path,
                output_dir=out_dir,
                dataset_name=f"episode{idx}",
                robot=robot,
                fps=fps,
                task_name=task,
                task_description=desc,
            )
            written.append(zp)
        except Exception:
            logger.error("Failed %s:\n%s", hdf5_path, traceback.format_exc())
    logger.info("Converted %d/%d episodes -> %s", len(written), len(eps), out_dir)
    return written


def argument_parse():
    p = argparse.ArgumentParser(
        description="Convert RoboTwin 2.0 HDF5 -> EgoVerse zarr."
    )
    p.add_argument(
        "--robot-dir",
        type=Path,
        required=True,
        help="Path to <robot>_<variant> dir (containing data/, instructions/).",
    )
    p.add_argument(
        "--robot", type=str, required=True, choices=list(ROBOT_TO_EMBODIMENT)
    )
    p.add_argument("--task", type=str, required=True, help="RoboTwin task name.")
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root for zarr datasets (e.g. .../robotwin/zarrDataset).",
    )
    p.add_argument("--fps", type=int, default=25)
    p.add_argument(
        "--limit", type=int, default=None, help="Convert only first N episodes."
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = argument_parse()
    convert_robot_folder(
        robot_dir=args.robot_dir,
        robot=args.robot,
        task=args.task,
        output_root=args.output_root,
        fps=args.fps,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
