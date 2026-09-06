"""Shared helpers for 3D-flow visualization scripts."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def prepare_pair_tensor(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Resize and stack two RGB frames into a (1, 2, C, H, W) tensor."""
    H_orig, W_orig = frame_a.shape[:2]
    scale = min(float(image_size) / H_orig, float(image_size) / W_orig)
    H = max(64, (int(H_orig * scale) // 64) * 64)
    W = max(64, (int(W_orig * scale) // 64) * 64)

    a_resized = cv2.resize(frame_a, (W, H))
    b_resized = cv2.resize(frame_b, (W, H))

    tensor = torch.stack(
        [
            torch.from_numpy(a_resized).permute(2, 0, 1),
            torch.from_numpy(b_resized).permute(2, 0, 1),
        ],
        dim=0,
    )
    return tensor.unsqueeze(0).float().to(device), H, W


def get_2d_colors(grid_uv: np.ndarray, height: int, width: int) -> np.ndarray:
    u = grid_uv[:, 0] / width
    v = grid_uv[:, 1] / height
    hue = (0.6 * u + 0.4 * v) % 1.0
    hsv = np.stack(
        [hue * 179, np.ones_like(u) * 230, np.ones_like(u) * 255], axis=-1
    ).astype(np.uint8)
    rgb = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)
    return rgb


def make_fixed_grid(
    height: int, width: int, rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create dense and sparse pixel grids anchored to the processed image."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    uu, vv = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    offset_u = min(rate // 2, width - 1)
    offset_v = min(rate // 2, height - 1)
    grid_u, grid_v = np.meshgrid(
        np.arange(offset_u, width, rate, dtype=np.int32),
        np.arange(offset_v, height, rate, dtype=np.int32),
        indexing="xy",
    )
    grid_uv = np.stack([grid_u, grid_v], axis=-1).reshape(-1, 2)
    return uu, vv, grid_uv


def draw_flow_arrows(
    canvas: np.ndarray,
    flow_2d: np.ndarray,
    grid_uv: np.ndarray,
    colors: np.ndarray,
    flow_scale: float,
    min_flow_mag: float,
) -> np.ndarray:
    """Draw arrowed flow on canvas at grid locations, clipping endpoints to edges."""
    H, W = canvas.shape[:2]
    for idx, (u_grid, v_grid) in enumerate(grid_uv):
        du, dv = flow_2d[v_grid, u_grid]
        if np.isnan(du) or np.isnan(dv):
            continue
        mag = np.sqrt(du * du + dv * dv)
        if mag < min_flow_mag:
            continue
        color = tuple(int(c) for c in colors[idx])
        pt1 = (int(u_grid), int(v_grid))
        end_u = int(np.clip(u_grid + du * flow_scale, 0, W - 1))
        end_v = int(np.clip(v_grid + dv * flow_scale, 0, H - 1))
        pt2 = (end_u, end_v)
        cv2.arrowedLine(
            canvas, pt1, pt2, color, 1, tipLength=0.2, line_type=cv2.LINE_AA
        )
    return canvas
