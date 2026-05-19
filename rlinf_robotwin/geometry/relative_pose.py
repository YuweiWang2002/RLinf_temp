"""Relative pose utilities for dual-arm RobotWin geometry."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def compute_relative_pose_error(
    left_ee_pose: np.ndarray,
    right_ee_pose: np.ndarray,
) -> np.ndarray:
    """Compute right end-effector pose relative to left end-effector pose.

    Args:
        left_ee_pose: Left end-effector pose with shape ``(7,)`` in
            ``[x, y, z, qx, qy, qz, qw]`` order.
        right_ee_pose: Right end-effector pose with shape ``(7,)`` in
            ``[x, y, z, qx, qy, qz, qw]`` order.

    Returns:
        Relative error with shape ``(4,)`` as ``[ex, ey, ez, eyaw]`` in
        meters and radians, where ``T_rel = inv(T_left_ee) @ T_right_ee``.
    """
    left_transform = _pose_to_transform(left_ee_pose)
    right_transform = _pose_to_transform(right_ee_pose)
    relative_transform = np.linalg.inv(left_transform) @ right_transform

    relative_translation = relative_transform[:3, 3]
    relative_yaw = math.atan2(relative_transform[1, 0], relative_transform[0, 0])

    return np.array(
        [
            relative_translation[0],
            relative_translation[1],
            relative_translation[2],
            _normalize_angle(relative_yaw),
        ],
        dtype=np.float64,
    )


def _pose_to_transform(pose: np.ndarray) -> np.ndarray:
    """Convert a ``[x, y, z, qx, qy, qz, qw]`` pose to a 4x4 transform."""
    pose_array = np.asarray(pose, dtype=np.float64)
    if pose_array.shape != (7,):
        raise ValueError(
            "Expected pose shape (7,) with order [x, y, z, qx, qy, qz, qw], "
            f"got {pose_array.shape}."
        )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(pose_array[3:7]).as_matrix()
    transform[:3, 3] = pose_array[:3]
    return transform


def _normalize_angle(angle: float) -> float:
    """Normalize an angle to ``[-pi, pi]``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
