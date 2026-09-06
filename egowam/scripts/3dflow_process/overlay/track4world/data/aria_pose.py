from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < eps:
        raise ValueError("Cannot normalize degenerate vector")
    return v / norm


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}")
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    return R.from_quat(quat_xyzw).as_matrix()


def build_head_only_c2w(head_pose: np.ndarray) -> np.ndarray:
    head_pose = np.asarray(head_pose, dtype=np.float64)
    if head_pose.ndim != 2 or head_pose.shape[1] != 7:
        raise ValueError(f"Expected head_pose shape (T, 7), got {head_pose.shape}")

    c2w = np.tile(np.eye(4, dtype=np.float64), (head_pose.shape[0], 1, 1))
    for i, pose in enumerate(head_pose):
        c2w[i, :3, :3] = quat_wxyz_to_matrix(pose[3:7])
        c2w[i, :3, 3] = pose[:3]
    return c2w


def sanitize_eye_gaze(eye_gaze: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eye_gaze = np.asarray(eye_gaze, dtype=np.float64)
    if eye_gaze.ndim != 2 or eye_gaze.shape[1] != 3:
        raise ValueError(f"Expected eye_gaze shape (T, 3), got {eye_gaze.shape}")

    sanitized = eye_gaze.copy()
    fallback_mask = np.zeros(eye_gaze.shape[0], dtype=bool)
    default = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    for i, gaze in enumerate(sanitized):
        if not np.all(np.isfinite(gaze)) or np.linalg.norm(gaze) < 1e-6:
            sanitized[i] = default
            fallback_mask[i] = True
        else:
            sanitized[i] = gaze / np.linalg.norm(gaze)
    return sanitized, fallback_mask


def _fallback_right(forward_world: np.ndarray, right_hint_world: np.ndarray) -> np.ndarray:
    right_world = right_hint_world - np.dot(right_hint_world, forward_world) * forward_world
    if np.linalg.norm(right_world) >= 1e-8:
        return right_world

    basis = np.eye(3, dtype=np.float64)
    candidates = sorted(basis, key=lambda axis: abs(np.dot(axis, forward_world)))
    for axis in candidates:
        candidate = axis - np.dot(axis, forward_world) * forward_world
        if np.linalg.norm(candidate) >= 1e-8:
            return candidate
    raise ValueError("Failed to construct a valid right axis")


def build_head_gaze_c2w(head_pose: np.ndarray, eye_gaze: np.ndarray) -> np.ndarray:
    head_pose = np.asarray(head_pose, dtype=np.float64)
    if head_pose.ndim != 2 or head_pose.shape[1] != 7:
        raise ValueError(f"Expected head_pose shape (T, 7), got {head_pose.shape}")
    gaze_head, _ = sanitize_eye_gaze(eye_gaze)
    if gaze_head.shape[0] != head_pose.shape[0]:
        raise ValueError("head_pose and eye_gaze must have the same number of frames")

    c2w = np.tile(np.eye(4, dtype=np.float64), (head_pose.shape[0], 1, 1))
    for i, pose in enumerate(head_pose):
        R_head = quat_wxyz_to_matrix(pose[3:7])
        forward_world = normalize(R_head @ gaze_head[i])
        down_hint_world = R_head[:, 1]
        right_world = np.cross(down_hint_world, forward_world)
        if np.linalg.norm(right_world) < 1e-8:
            right_world = _fallback_right(forward_world, R_head[:, 0])
        right_world = normalize(right_world)
        down_world = normalize(np.cross(forward_world, right_world))
        c2w[i, :3, :3] = np.stack([right_world, down_world, forward_world], axis=1)
        c2w[i, :3, 3] = pose[:3]
    return c2w


def rebase_c2w(c2w: np.ndarray, reference_index: int = 0) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w shape (T, 4, 4), got {c2w.shape}")
    if not (0 <= reference_index < c2w.shape[0]):
        raise ValueError(f"reference_index {reference_index} out of range for {c2w.shape[0]} frames")

    w2ref = np.linalg.inv(c2w[reference_index])
    rebased = np.einsum("ij,tjk->tik", w2ref, c2w)
    return rebased
