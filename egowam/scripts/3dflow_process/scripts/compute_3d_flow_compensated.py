#!/usr/bin/env python3
"""Compute 3D flow with head-pose camera compensation from Aria zarr data.

Uses Track4World (PI3X) to compute dense 3D scene flow between frame pairs.
The three visualized flows satisfy: total ≈ cam + compensated, and the
compensated 2D flow ≈ projected 3D flow (they differ only by perspective
non-linearity between the source and target cameras).

  - Compensated 2D flow: u_flow - u_static, where u_static is the target-frame
    projection of source points warped by T_rel = w2c[t+k] @ c2w[t]. Object
    motion only, target pixel grid.
  - Cam flow: u_static - uu. Pixel displacement of static world points induced
    purely by head motion. Source pixel grid.
  - Projected 3D flow: flow_3d (target frame under force_projection=True) is
    pulled back to the source camera via T_rel^-1, then the endpoint is
    projected through source intrinsics and differenced against (uu, vv).

Left panel:   Total flow (ego + object)
Middle panel: Compensated (target-frame, T_rel-based)
Right panel:  Compensated (source-frame, projection-based)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import track4world.utils.basic
from track4world.data.aria_pose import build_head_only_c2w
from track4world.data.aria_zarr import export_frames, open_episode
from track4world.utils.flow_vis import (
    draw_flow_arrows,
    get_2d_colors,
    make_fixed_grid,
    prepare_pair_tensor,
)

LOGGER = logging.getLogger("compute_3d_flow_compensated")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 3D flow with head-pose camera compensation."
    )
    parser.add_argument("--zarr-path", required=True, help="Path to Aria episode zarr")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument(
        "--frame-skip", type=int, default=30,
        help="Temporal gap between paired frames",
    )
    parser.add_argument(
        "--config-path", default="track4world/config/eval/v1.json",
        help="Track4World model config",
    )
    parser.add_argument(
        "--ckpt-init", default="checkpoints/track4world_da3.pth",
        help="Path to Track4World checkpoint",
    )
    parser.add_argument(
        "--coordinate", default="world_depthanythingv3",
        help="Coordinate system / backbone selector",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rate", type=int, default=16,
        help="Grid spacing for visualization arrows",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30, help="Output video framerate")
    parser.add_argument(
        "--flow-scale", type=float, default=1.0,
        help="Arrow length multiplier",
    )
    parser.add_argument(
        "--bkg-opacity", type=float, default=0.5,
        help="Background dimming factor",
    )
    parser.add_argument(
        "--min-flow-mag", type=float, default=0.5,
        help="Skip arrows below this magnitude (pixels)",
    )
    parser.add_argument(
        "--comp-threshold", type=float, default=5.0,
        help="Magnitude threshold to suppress background noise in compensated flow",
    )
    parser.add_argument(
        "--image-size", type=int, default=640,
        help="Max image dimension for Track4World input",
    )
    parser.add_argument(
        "--inference-iters", type=int, default=4,
        help="Number of refinement iterations in Track4World",
    )
    parser.add_argument(
        "--segment-index", type=int, default=None,
        help="Aria segment index within the zarr (timestamp-gap split)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading (reuses patterns from efep_export / demo)
# ---------------------------------------------------------------------------

def load_track4world_model(
    config_path: str,
    ckpt_init: str,
    coordinate: str,
    device: str = "cuda",
) -> torch.nn.Module:
    from track4world.pipeline.efep_export import load_model_from_config

    return load_model_from_config(
        config_path=config_path,
        ckpt_init=ckpt_init,
        coordinate=coordinate,
        device=device,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Load episode metadata
    # ------------------------------------------------------------------
    episode = open_episode(args.zarr_path, segment_index=args.segment_index)
    total_valid = episode.valid_indices.shape[0]

    start_frame = max(0, args.start_frame)
    if args.end_frame is not None:
        end_frame = min(total_valid, args.end_frame)
    elif args.max_frames is not None:
        end_frame = min(total_valid, start_frame + args.max_frames)
    else:
        end_frame = total_valid

    num_frames = end_frame - start_frame
    if num_frames <= 0:
        raise ValueError(f"No frames with start={start_frame}, end={end_frame}")

    valid_indices = episode.valid_indices[start_frame:end_frame]
    head_pose = episode.head_pose[start_frame:end_frame]

    # ------------------------------------------------------------------
    # 2. Export frames
    # ------------------------------------------------------------------
    frames_tmp_dir = output_dir / f"temp_frames_{os.getpid()}"
    frames_tmp_dir.mkdir(exist_ok=True)
    LOGGER.info("Exporting %d frames...", num_frames)
    frame_paths = export_frames(
        args.zarr_path, frames_tmp_dir, valid_indices, max_frames=num_frames
    )

    frames_rgb = []
    for p in frame_paths:
        frames_rgb.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
    frames_rgb = np.stack(frames_rgb)

    # ------------------------------------------------------------------
    # 3. Build head-pose c2w matrices
    # ------------------------------------------------------------------
    c2w = build_head_only_c2w(head_pose)  # (T, 4, 4)
    w2c = np.linalg.inv(c2w)             # (T, 4, 4)

    # ------------------------------------------------------------------
    # 4. Load Track4World model
    # ------------------------------------------------------------------
    LOGGER.info("Loading Track4World model...")
    model = load_track4world_model(
        config_path=args.config_path,
        ckpt_init=args.ckpt_init,
        coordinate=args.coordinate,
        device=args.device,
    )
    LOGGER.info("Model loaded.")

    # ------------------------------------------------------------------
    # 5. Run pairwise inference and compute compensated flow
    # ------------------------------------------------------------------
    T = num_frames
    rate = args.rate

    # We'll compute H/W from the first pair
    H_proc = W_proc = None
    uu = vv = None
    grid_uv = None
    all_proj_flow_2d = []
    all_sampled_pts_now = []
    all_sampled_pts_fut = []

    for t in range(T - 1):
        tskip = min(t + args.frame_skip, T - 1)

        # First pair sets the processing resolution
        if H_proc is None:
            pair_tensor, H_proc, W_proc = prepare_pair_tensor(
                frames_rgb[t], frames_rgb[tskip], args.image_size, device
            )
            uu, vv, grid_uv = make_fixed_grid(H_proc, W_proc, rate)
        else:
            pair_tensor, _, _ = prepare_pair_tensor(
                frames_rgb[t], frames_rgb[tskip], args.image_size, device
            )

        # ----- Track4World inference for this pair -----
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                results = model.infer_pair(
                    pair_tensor,
                    iters=args.inference_iters,
                    sw=None,
                    is_training=False,
                    tracking3d=True,
                    force_projection=True,
                )

        geo_dict = results[0]   # per-frame geometry
        motion_dict = results[1]  # pairwise motion

        flow_2d_abs = motion_dict["flow_2d"][0, 0].cpu().numpy().transpose(1, 2, 0)
        flow_3d = motion_dict["flow_3d"][0, 0].cpu().numpy()  # (H, W, 3)
        points_t = geo_dict["points"][0, 0].cpu().numpy()     # (H, W, 3)
        intrinsics_t = geo_dict["intrinsics"][0, 0].cpu().numpy()  # (3, 3)

        fx = intrinsics_t[0, 0] * W_proc
        fy = intrinsics_t[1, 1] * H_proc
        cx = intrinsics_t[0, 2] * W_proc
        cy = intrinsics_t[1, 2] * H_proc

        # Only projected 3D flow + sampled points are persisted; the total/
        # compensated/cam intermediates were removed to shrink outputs.
        T_rel = w2c[tskip] @ c2w[t]  # (4, 4) relative transform
        T_inv = np.linalg.inv(T_rel).astype(np.float32)
        R_inv = T_inv[:3, :3]
        t_inv = T_inv[:3, 3]
        P_src_end = (flow_3d @ R_inv.T) + t_inv

        # Sample 3D positions at the visualization grid. Both expressed in
        # the source camera frame at time t.
        gu_arr = grid_uv[:, 0]
        gv_arr = grid_uv[:, 1]
        sampled_pts_now = points_t[gv_arr, gu_arr].astype(np.float32)
        sampled_pts_fut = P_src_end[gv_arr, gu_arr].astype(np.float32)

        u_end_src = (P_src_end[..., 0] * fx / np.clip(P_src_end[..., 2], 1e-6, None)) + cx
        v_end_src = (P_src_end[..., 1] * fy / np.clip(P_src_end[..., 2], 1e-6, None)) + cy
        proj_flow_2d = np.stack([u_end_src - uu, v_end_src - vv], axis=-1).astype(np.float32)

        LOGGER.info(
            "Frame %d->%d  proj3d=%.1f",
            t, tskip,
            np.nanmax(np.linalg.norm(proj_flow_2d, axis=-1)),
        )

        # Store raw data for npy saving. total/comp flow + viz frames were
        # removed from output (viz-only); skip their accumulation to save CPU
        # and RAM during processing.
        all_proj_flow_2d.append(proj_flow_2d)
        all_sampled_pts_now.append(sampled_pts_now)
        all_sampled_pts_fut.append(sampled_pts_fut)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    # Training only reads 2D flow at the sampled anchor positions. Store a
    # compact (T, N, 2) slice (sampled_proj_flow_2d) instead of the full
    # (T, H, W, 2) field — ~2 MB/ep vs ~520 MB/ep.
    LOGGER.info("Saving results...")
    grid_uv_i32 = grid_uv.astype(np.int32)
    # grid_uv rows are (u, v) = (x, y); numpy index order is [y, x].
    vy = grid_uv_i32[:, 1]
    vx = grid_uv_i32[:, 0]
    sampled_proj_flow_2d = np.stack(
        [f[vy, vx] for f in all_proj_flow_2d]
    ).astype(np.float32)
    np.save(output_dir / "sampled_proj_flow_2d.npy", sampled_proj_flow_2d)

    # Sampled 3D positions at grid_uv, both in source camera frame at time t.
    np.save(output_dir / "sampled_uv.npy", grid_uv_i32)
    np.save(output_dir / "sampled_points_t.npy", np.stack(all_sampled_pts_now))
    np.save(output_dir / "sampled_points_tk.npy", np.stack(all_sampled_pts_fut))

    raw_flow_artifacts = {
        "sampled_proj_flow_2d_path": str(output_dir / "sampled_proj_flow_2d.npy"),
        "sampled_uv_path": str(output_dir / "sampled_uv.npy"),
        "sampled_points_t_path": str(output_dir / "sampled_points_t.npy"),
        "sampled_points_tk_path": str(output_dir / "sampled_points_tk.npy"),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "zarr_path": args.zarr_path,
                "frame_skip": args.frame_skip,
                "num_frames": num_frames,
                "num_processed": T - 1,
                "image_size": [H_proc, W_proc],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "segment_index": episode.segment_index,
                "raw_flow_artifacts": raw_flow_artifacts,
            },
            f,
            indent=2,
        )

    processed_indices = episode.valid_indices[start_frame : start_frame + T - 1]
    segment_metadata = {
        "source_zarr_path": str(args.zarr_path),
        "segment_index": episode.segment_index,
        "segment_start": episode.segment_start,
        "segment_end": episode.segment_end,
        "valid_frame_count": int(total_valid),
        "valid_frame_indices": episode.valid_indices.tolist(),
        "processed_frame_indices": processed_indices.tolist(),
        "timestamps_ns": episode.timestamps_ns.tolist(),
        "fps": int(episode.fps),
        "frame_skip": int(args.frame_skip),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "image_size": [int(H_proc), int(W_proc)],
        "raw_flow_artifacts": raw_flow_artifacts,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    with open(output_dir / "segment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(segment_metadata, f, indent=2, sort_keys=True)

    shutil.rmtree(frames_tmp_dir)
    (output_dir / "_SUCCESS").touch()
    LOGGER.info("Done! Output saved to %s", output_dir)


if __name__ == "__main__":
    main()
