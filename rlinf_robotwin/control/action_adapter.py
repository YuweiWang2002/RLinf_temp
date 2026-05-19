"""Named RoboTwin/OpenPI action format adapters.

The project currently has several 14D action/state vectors with different
semantics. Keep the names explicit at conversion boundaries:

RoboTwinQpos14:
    [left qpos6, left gripper1, right qpos6, right gripper1]

RoboTwinEndpose16:
    [left xyz_quat7, left gripper1, right xyz_quat7, right gripper1]
    where xyz_quat7 is [x, y, z, qx, qy, qz, qw].

RoboTwinEEPose14:
    [left xyz_quat7, right xyz_quat7]
    This is the internal format used by the current residual wrapper and
    relative EE fusion.

OpenPIEEAction14:
    [left xyz_rotvec6, left gripper1, right xyz_rotvec6, right gripper1]
    where xyz_rotvec6 is [x, y, z, rx, ry, rz]. The rotation vector is
    axis-angle, not Euler.
"""

from __future__ import annotations

import numpy as np

ROBOTWIN_QPOS14_DIM = 14
ROBOTWIN_ENDPOSE16_DIM = 16
ROBOTWIN_EE_POSE14_DIM = 14
OPENPI_EE_ACTION14_DIM = 14


def endpose16_to_pose14(endpose16: np.ndarray) -> np.ndarray:
    """Convert RoboTwinEndpose16 to RoboTwinEEPose14.

    This drops the left/right gripper values and keeps only
    ``[left xyz_quat7, right xyz_quat7]``.
    """
    endpose = _as_float_array(endpose16, ROBOTWIN_ENDPOSE16_DIM, "endpose16")
    left_pose = endpose[..., 0:7]
    right_pose = endpose[..., 8:15]
    return np.concatenate((left_pose, right_pose), axis=-1)


def pose14_to_endpose16(
    pose14: np.ndarray,
    left_gripper: np.ndarray | float,
    right_gripper: np.ndarray | float,
) -> np.ndarray:
    """Convert RoboTwinEEPose14 plus grippers to RoboTwinEndpose16."""
    pose = _as_float_array(pose14, ROBOTWIN_EE_POSE14_DIM, "pose14")
    left = pose[..., 0:7]
    right = pose[..., 7:14]
    left_grip = _as_gripper_column(left_gripper, pose.shape[:-1], "left_gripper")
    right_grip = _as_gripper_column(right_gripper, pose.shape[:-1], "right_gripper")
    return np.concatenate((left, left_grip, right, right_grip), axis=-1)


def quat_xyzw_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert normalized xyzw quaternions to rotation vectors."""
    q = normalize_quat_xyzw(_as_float_array(quat, 4, "quat"))
    xyz = q[..., 0:3]
    w = np.clip(q[..., 3:4], -1.0, 1.0)
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(xyz_norm, w)
    angle = np.where(angle > np.pi, angle - 2.0 * np.pi, angle)
    scale = np.divide(
        angle,
        xyz_norm,
        out=np.zeros_like(angle),
        where=xyz_norm > _eps(q),
    )
    return xyz * scale


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """Convert rotation vectors to normalized xyzw quaternions."""
    rv = _as_float_array(rotvec, 3, "rotvec")
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    half_angle = 0.5 * angle
    scale = np.divide(
        np.sin(half_angle),
        angle,
        out=np.full_like(angle, 0.5),
        where=angle > _eps(rv),
    )
    quat = np.concatenate((rv * scale, np.cos(half_angle)), axis=-1)
    return normalize_quat_xyzw(quat)


def endpose16_to_openpi_ee14(endpose16: np.ndarray) -> np.ndarray:
    """Convert RoboTwinEndpose16 to OpenPIEEAction14."""
    endpose = _as_float_array(endpose16, ROBOTWIN_ENDPOSE16_DIM, "endpose16")
    left_xyz = endpose[..., 0:3]
    left_rotvec = quat_xyzw_to_rotvec(endpose[..., 3:7])
    left_gripper = endpose[..., 7:8]
    right_xyz = endpose[..., 8:11]
    right_rotvec = quat_xyzw_to_rotvec(endpose[..., 11:15])
    right_gripper = endpose[..., 15:16]
    return np.concatenate(
        (
            left_xyz,
            left_rotvec,
            left_gripper,
            right_xyz,
            right_rotvec,
            right_gripper,
        ),
        axis=-1,
    )


def openpi_ee14_to_pose14(openpi_action14: np.ndarray) -> np.ndarray:
    """Convert OpenPIEEAction14 to RoboTwinEEPose14, dropping grippers."""
    action = _as_float_array(openpi_action14, OPENPI_EE_ACTION14_DIM, "openpi_action14")
    left_pose = np.concatenate(
        (action[..., 0:3], rotvec_to_quat_xyzw(action[..., 3:6])),
        axis=-1,
    )
    right_pose = np.concatenate(
        (action[..., 7:10], rotvec_to_quat_xyzw(action[..., 10:13])),
        axis=-1,
    )
    return np.concatenate((left_pose, right_pose), axis=-1)


def openpi_ee14_to_endpose16(openpi_action14: np.ndarray) -> np.ndarray:
    """Convert OpenPIEEAction14 to RoboTwinEndpose16, preserving grippers."""
    action = _as_float_array(openpi_action14, OPENPI_EE_ACTION14_DIM, "openpi_action14")
    pose14 = openpi_ee14_to_pose14(action)
    return pose14_to_endpose16(
        pose14,
        left_gripper=action[..., 6],
        right_gripper=action[..., 13],
    )


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """Normalize xyzw quaternions and reject zero-length quaternions."""
    q = _as_float_array(quat, 4, "quat")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= _eps(q)):
        msg = "quat must not contain zero-length quaternions."
        raise ValueError(msg)
    return q / norm


def _as_float_array(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        msg = f"{name} must be a numpy.ndarray."
        raise TypeError(msg)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape[-1:] != (dim,):
        msg = f"{name} must have shape (..., {dim}), got {arr.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return arr


def _as_gripper_column(
    value: np.ndarray | float,
    batch_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        return np.full((*batch_shape, 1), float(arr), dtype=np.float64)
    if arr.shape == batch_shape:
        return arr[..., None]
    if arr.shape == (*batch_shape, 1):
        return arr
    msg = (
        f"{name} must be scalar, shape {batch_shape}, or shape "
        f"{(*batch_shape, 1)}, got {arr.shape}."
    )
    raise ValueError(msg)


def _eps(value: np.ndarray) -> np.ndarray:
    return np.asarray(np.finfo(value.dtype).eps, dtype=value.dtype)
