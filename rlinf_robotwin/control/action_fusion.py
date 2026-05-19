"""Utilities for fusing residual actions into configured action layouts."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for lightweight tests.
    torch = None


@dataclass(frozen=True)
class ActionLayout:
    """Action indices used by residual action fusion.

    This layout intentionally has no RoboTwin default indices. Callers must pass
    the action mapping confirmed for the concrete environment/action space.
    """

    right_slice: slice
    right_pos_indices: tuple[int, int, int]
    right_yaw_index: int
    left_slice: slice | None = None
    gripper_indices: tuple[int, ...] = field(default_factory=tuple)
    action_type: str | None = None
    action_dim: int | None = None
    pose_format: str | None = None
    quat_order: str | None = None

    def __post_init__(self) -> None:
        """Normalize tuple-like fields."""
        if len(self.right_pos_indices) != 3:
            msg = "right_pos_indices must contain exactly three indices."
            raise ValueError(msg)
        object.__setattr__(
            self,
            "right_pos_indices",
            tuple(operator.index(index) for index in self.right_pos_indices),
        )
        object.__setattr__(
            self,
            "right_yaw_index",
            operator.index(self.right_yaw_index),
        )
        object.__setattr__(
            self,
            "gripper_indices",
            tuple(operator.index(index) for index in self.gripper_indices),
        )
        if self.action_dim is not None:
            object.__setattr__(self, "action_dim", operator.index(self.action_dim))


def fuse_residual_action(
    base_action: np.ndarray,
    residual_action: np.ndarray,
    action_layout: ActionLayout,
) -> np.ndarray:
    """Fuse a 4D residual into the right-arm action dimensions.

    Args:
        base_action: One-dimensional base action vector.
        residual_action: One-dimensional residual with shape ``(4,)`` and
            semantic order ``[dx, dy, dz, dyaw]``. The meter/rad meaning is valid
            only when ``action_layout`` maps these entries to same-unit right-arm
            EE or delta-EE action dimensions. Do not use this function to add
            Cartesian residuals directly into qpos joint dimensions.
        action_layout: Explicit layout for the action vector. No RoboTwin action
            indices are assumed by this function.

    Returns:
        A copy of ``base_action`` with residual position and yaw added to the
        configured right-arm dimensions.

    Raises:
        TypeError: If either action input is not a ``numpy.ndarray``.
        ValueError: If shapes or layout indices are invalid.
    """
    _validate_action_inputs(base_action, residual_action)
    normalized_layout = _validate_action_layout(base_action, action_layout)

    fused_action = np.array(base_action, copy=True)
    fused_action[list(normalized_layout.right_pos_indices)] += residual_action[:3]
    fused_action[normalized_layout.right_yaw_index] += residual_action[3]
    return fused_action


def fuse_relative_ee_action_with_residual(
    base_action: Any,
    residual: Any,
    layout: ActionLayout,
    *,
    residual_frame: str = "left",
    yaw_axis: str = "z",
    quat_order: str = "xyzw",
) -> Any:
    """Fuse a relative 4D EE residual into a dual-arm 14D EE action.

    The residual is interpreted in the left/reference arm frame. The left arm is
    copied unchanged, while the right arm is recomputed from
    ``T_left @ (Delta_T_rel @ inv(T_left) @ T_right)``.
    """
    _validate_relative_ee_inputs(base_action, residual, layout, quat_order)
    _validate_relative_options(residual_frame, yaw_axis)

    fused_action = _copy_array(base_action)
    left_pose = base_action[..., 0:7]
    right_pose = base_action[..., 7:14]

    left_transform = pose7d_to_matrix_xyzw(left_pose)
    right_transform = pose7d_to_matrix_xyzw(right_pose)
    relative_transform = compose_transform(
        invert_transform(left_transform),
        right_transform,
    )
    residual_transform = residual4d_to_transform(residual)
    new_relative_transform = compose_transform(residual_transform, relative_transform)
    new_right_transform = compose_transform(left_transform, new_relative_transform)

    fused_action[..., 0:7] = base_action[..., 0:7]
    fused_action[..., 7:14] = matrix_to_pose7d_xyzw(new_right_transform)
    return fused_action


def normalize_quat_xyzw(q: Any) -> Any:
    """Normalize quaternions in ``[qx, qy, qz, qw]`` order."""
    norm = _linalg_norm(q, axis=-1, keepdims=True)
    return q / _clip_min(norm, _eps(q))


def quat_to_rotmat_xyzw(q: Any) -> Any:
    """Convert xyzw quaternions to rotation matrices."""
    quat = normalize_quat_xyzw(q)
    x, y, z, w = _unbind_last(quat)

    two = _scalar_like(quat, 2.0)
    one = _scalar_like(quat, 1.0)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    row0 = _stack(
        (one - two * (yy + zz), two * (xy - wz), two * (xz + wy)),
        axis=-1,
        like=quat,
    )
    row1 = _stack(
        (two * (xy + wz), one - two * (xx + zz), two * (yz - wx)),
        axis=-1,
        like=quat,
    )
    row2 = _stack(
        (two * (xz - wy), two * (yz + wx), one - two * (xx + yy)),
        axis=-1,
        like=quat,
    )
    return _stack((row0, row1, row2), axis=-2, like=quat)


def rotmat_to_quat_xyzw(rotmat: Any) -> Any:
    """Convert rotation matrices to normalized xyzw quaternions."""
    if _is_torch(rotmat):
        return _rotmat_to_quat_xyzw_torch(rotmat)
    return _rotmat_to_quat_xyzw_numpy(rotmat)


def yaw_to_rotmat(yaw: Any) -> Any:
    """Convert z-axis yaw angles to rotation matrices."""
    cos_yaw = _cos(yaw)
    sin_yaw = _sin(yaw)
    zero = _zeros_like(yaw)
    one = _ones_like(yaw)
    row0 = _stack((cos_yaw, -sin_yaw, zero), axis=-1, like=yaw)
    row1 = _stack((sin_yaw, cos_yaw, zero), axis=-1, like=yaw)
    row2 = _stack((zero, zero, one), axis=-1, like=yaw)
    return _stack((row0, row1, row2), axis=-2, like=yaw)


def pose7d_to_matrix_xyzw(pose: Any) -> Any:
    """Convert ``[..., x, y, z, qx, qy, qz, qw]`` pose arrays to matrices."""
    _require_last_dim(pose, 7, "pose")
    transform = _eye4_like(pose, pose.shape[:-1])
    transform[..., :3, :3] = quat_to_rotmat_xyzw(pose[..., 3:7])
    transform[..., :3, 3] = pose[..., 0:3]
    return transform


def matrix_to_pose7d_xyzw(transform: Any) -> Any:
    """Convert transform matrices to normalized xyzw pose arrays."""
    if transform.shape[-2:] != (4, 4):
        msg = f"transform must have shape (..., 4, 4), got {transform.shape}."
        raise ValueError(msg)
    position = transform[..., :3, 3]
    quat = rotmat_to_quat_xyzw(transform[..., :3, :3])
    return _concat((position, normalize_quat_xyzw(quat)), axis=-1, like=transform)


def invert_transform(transform: Any) -> Any:
    """Invert rigid transforms with shape ``(..., 4, 4)``."""
    if transform.shape[-2:] != (4, 4):
        msg = f"transform must have shape (..., 4, 4), got {transform.shape}."
        raise ValueError(msg)
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    rot_inv = _swap_last_two(rot)
    trans_inv = -_matvec(rot_inv, trans)
    out = _eye4_like(transform, transform.shape[:-2])
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = trans_inv
    return out


def compose_transform(a_transform: Any, b_transform: Any) -> Any:
    """Compose rigid transforms."""
    return _matmul(a_transform, b_transform)


def residual4d_to_transform(residual: Any) -> Any:
    """Convert ``[..., dx, dy, dz, dyaw]`` residuals to z-yaw transforms."""
    _require_last_dim(residual, 4, "residual")
    transform = _eye4_like(residual, residual.shape[:-1])
    transform[..., :3, :3] = yaw_to_rotmat(residual[..., 3])
    transform[..., :3, 3] = residual[..., 0:3]
    return transform


def _validate_action_inputs(
    base_action: np.ndarray,
    residual_action: np.ndarray,
) -> None:
    if not isinstance(base_action, np.ndarray):
        msg = "base_action must be a numpy.ndarray."
        raise TypeError(msg)
    if not isinstance(residual_action, np.ndarray):
        msg = "residual_action must be a numpy.ndarray."
        raise TypeError(msg)
    if base_action.ndim != 1:
        msg = "base_action must be one-dimensional."
        raise ValueError(msg)
    if residual_action.shape != (4,):
        msg = "residual_action must have shape (4,)."
        raise ValueError(msg)


def _validate_action_layout(
    base_action: np.ndarray,
    action_layout: ActionLayout,
) -> ActionLayout:
    if not isinstance(action_layout, ActionLayout):
        msg = "action_layout must be an ActionLayout."
        raise TypeError(msg)

    action_dim = base_action.shape[0]
    right_pos_indices = tuple(
        _normalize_index(index, action_dim, "right_pos_indices")
        for index in action_layout.right_pos_indices
    )
    right_yaw_index = _normalize_index(
        action_layout.right_yaw_index,
        action_dim,
        "right_yaw_index",
    )
    gripper_indices = tuple(
        _normalize_index(index, action_dim, "gripper_indices")
        for index in action_layout.gripper_indices
    )

    right_indices = (*right_pos_indices, right_yaw_index)
    if len(set(right_indices)) != len(right_indices):
        msg = "right_pos_indices and right_yaw_index must be unique."
        raise ValueError(msg)
    if set(right_indices).intersection(gripper_indices):
        msg = "right-arm residual indices must not overlap gripper_indices."
        raise ValueError(msg)

    right_slice_indices = _slice_indices(
        action_layout.right_slice,
        action_dim,
        "right_slice",
    )
    if not set(right_indices).issubset(right_slice_indices):
        msg = "right-arm residual indices must fall within right_slice."
        raise ValueError(msg)

    if action_layout.left_slice is not None:
        left_slice_indices = _slice_indices(
            action_layout.left_slice,
            action_dim,
            "left_slice",
        )
        if set(right_indices).intersection(left_slice_indices):
            msg = "right-arm residual indices must not overlap left_slice."
            raise ValueError(msg)

    return ActionLayout(
        right_slice=action_layout.right_slice,
        right_pos_indices=right_pos_indices,
        right_yaw_index=right_yaw_index,
        left_slice=action_layout.left_slice,
        gripper_indices=gripper_indices,
    )


def _validate_relative_ee_inputs(
    base_action: Any,
    residual: Any,
    layout: ActionLayout,
    quat_order: str,
) -> None:
    if not _is_supported_array(base_action):
        msg = "base_action must be a numpy.ndarray or torch.Tensor."
        raise TypeError(msg)
    if not _is_supported_array(residual):
        msg = "residual must be a numpy.ndarray or torch.Tensor."
        raise TypeError(msg)
    if _is_torch(base_action) != _is_torch(residual):
        msg = "base_action and residual must use the same array backend."
        raise TypeError(msg)
    if not isinstance(layout, ActionLayout):
        msg = "layout must be an ActionLayout."
        raise TypeError(msg)
    if layout.action_type == "qpos":
        msg = "Cartesian residual fusion is not supported for qpos actions."
        raise NotImplementedError(msg)
    if layout.action_type != "ee":
        msg = "action_type must be 'ee' for relative EE residual fusion."
        raise ValueError(msg)
    if layout.action_dim != 14:
        msg = "action_dim must be 14 for dual-arm EE xyz_quat actions."
        raise ValueError(msg)
    if layout.pose_format != "xyz_quat":
        msg = "pose_format must be 'xyz_quat' for relative EE residual fusion."
        raise ValueError(msg)
    if quat_order != "xyzw" or (
        layout.quat_order is not None and layout.quat_order != "xyzw"
    ):
        msg = "quat_order must be 'xyzw' for relative EE residual fusion."
        raise ValueError(msg)
    if base_action.shape[-1:] != (14,):
        msg = f"base_action must have shape (..., 14), got {base_action.shape}."
        raise ValueError(msg)
    if residual.shape[-1:] != (4,):
        msg = f"residual must have shape (..., 4), got {residual.shape}."
        raise ValueError(msg)
    if base_action.shape[:-1] != residual.shape[:-1]:
        msg = (
            "base_action and residual batch shapes must match: "
            f"base={base_action.shape}, residual={residual.shape}."
        )
        raise ValueError(msg)
    _validate_dual_ee_layout(layout)


def _validate_dual_ee_layout(layout: ActionLayout) -> None:
    action_dim = 14
    if _slice_indices(layout.left_slice, action_dim, "left_slice") != set(range(0, 7)):
        msg = "left_slice must select action[0:7] for the confirmed EE layout."
        raise ValueError(msg)
    if _slice_indices(layout.right_slice, action_dim, "right_slice") != set(
        range(7, 14)
    ):
        msg = "right_slice must select action[7:14] for the confirmed EE layout."
        raise ValueError(msg)
    right_pos_indices = tuple(
        _normalize_index(index, action_dim, "right_pos_indices")
        for index in layout.right_pos_indices
    )
    if right_pos_indices != (7, 8, 9):
        msg = "right_pos_indices must be (7, 8, 9) for the confirmed EE layout."
        raise ValueError(msg)


def _validate_relative_options(residual_frame: str, yaw_axis: str) -> None:
    if residual_frame != "left":
        msg = "Only residual_frame='left' is supported."
        raise ValueError(msg)
    if yaw_axis != "z":
        msg = "Only yaw_axis='z' is supported."
        raise ValueError(msg)


def _is_supported_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) or _is_torch(value)


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _copy_array(value: Any) -> Any:
    if _is_torch(value):
        return value.clone()
    return np.array(value, copy=True)


def _require_last_dim(value: Any, dim: int, name: str) -> None:
    if value.shape[-1:] != (dim,):
        msg = f"{name} must have shape (..., {dim}), got {value.shape}."
        raise ValueError(msg)


def _linalg_norm(value: Any, axis: int, keepdims: bool) -> Any:
    if _is_torch(value):
        return torch.linalg.norm(value, dim=axis, keepdim=keepdims)
    return np.linalg.norm(value, axis=axis, keepdims=keepdims)


def _clip_min(value: Any, minimum: Any) -> Any:
    if _is_torch(value):
        return torch.clamp(value, min=minimum)
    return np.maximum(value, minimum)


def _eps(value: Any) -> Any:
    if _is_torch(value):
        return torch.finfo(value.dtype).eps
    return np.finfo(value.dtype).eps


def _unbind_last(value: Any) -> tuple[Any, ...]:
    if _is_torch(value):
        return tuple(value.unbind(dim=-1))
    return tuple(np.moveaxis(value, -1, 0))


def _scalar_like(value: Any, scalar: float) -> Any:
    if _is_torch(value):
        return torch.as_tensor(scalar, dtype=value.dtype, device=value.device)
    return np.asarray(scalar, dtype=value.dtype)


def _stack(values: tuple[Any, ...], axis: int, like: Any) -> Any:
    if _is_torch(like):
        return torch.stack(values, dim=axis)
    return np.stack(values, axis=axis)


def _concat(values: tuple[Any, ...], axis: int, like: Any) -> Any:
    if _is_torch(like):
        return torch.cat(values, dim=axis)
    return np.concatenate(values, axis=axis)


def _cos(value: Any) -> Any:
    if _is_torch(value):
        return torch.cos(value)
    return np.cos(value)


def _sin(value: Any) -> Any:
    if _is_torch(value):
        return torch.sin(value)
    return np.sin(value)


def _zeros_like(value: Any) -> Any:
    if _is_torch(value):
        return torch.zeros_like(value)
    return np.zeros_like(value)


def _ones_like(value: Any) -> Any:
    if _is_torch(value):
        return torch.ones_like(value)
    return np.ones_like(value)


def _eye4_like(value: Any, batch_shape: tuple[int, ...]) -> Any:
    if _is_torch(value):
        eye = torch.eye(4, dtype=value.dtype, device=value.device)
        return eye.expand(*batch_shape, 4, 4).clone()
    eye = np.eye(4, dtype=value.dtype)
    return np.broadcast_to(eye, (*batch_shape, 4, 4)).copy()


def _swap_last_two(value: Any) -> Any:
    if _is_torch(value):
        return value.transpose(-1, -2)
    return np.swapaxes(value, -1, -2)


def _matvec(matrix: Any, vector: Any) -> Any:
    if _is_torch(matrix):
        return torch.matmul(matrix, vector.unsqueeze(-1)).squeeze(-1)
    return np.matmul(matrix, np.expand_dims(vector, axis=-1)).squeeze(axis=-1)


def _matmul(left: Any, right: Any) -> Any:
    if _is_torch(left):
        return torch.matmul(left, right)
    return np.matmul(left, right)


def _rotmat_to_quat_xyzw_numpy(rotmat: np.ndarray) -> np.ndarray:
    m00 = rotmat[..., 0, 0]
    m01 = rotmat[..., 0, 1]
    m02 = rotmat[..., 0, 2]
    m10 = rotmat[..., 1, 0]
    m11 = rotmat[..., 1, 1]
    m12 = rotmat[..., 1, 2]
    m20 = rotmat[..., 2, 0]
    m21 = rotmat[..., 2, 1]
    m22 = rotmat[..., 2, 2]

    qx = np.copysign(0.5 * np.sqrt(np.maximum(0.0, 1.0 + m00 - m11 - m22)), m21 - m12)
    qy = np.copysign(0.5 * np.sqrt(np.maximum(0.0, 1.0 - m00 + m11 - m22)), m02 - m20)
    qz = np.copysign(0.5 * np.sqrt(np.maximum(0.0, 1.0 - m00 - m11 + m22)), m10 - m01)
    qw = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + m00 + m11 + m22))
    quat = np.stack((qx, qy, qz, qw), axis=-1)
    return normalize_quat_xyzw(quat)


def _rotmat_to_quat_xyzw_torch(rotmat: Any) -> Any:
    m00 = rotmat[..., 0, 0]
    m01 = rotmat[..., 0, 1]
    m02 = rotmat[..., 0, 2]
    m10 = rotmat[..., 1, 0]
    m11 = rotmat[..., 1, 1]
    m12 = rotmat[..., 1, 2]
    m20 = rotmat[..., 2, 0]
    m21 = rotmat[..., 2, 1]
    m22 = rotmat[..., 2, 2]

    zero = torch.zeros((), dtype=rotmat.dtype, device=rotmat.device)
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = qx * torch.where(m21 - m12 < 0.0, -torch.ones_like(qx), torch.ones_like(qx))
    qy = qy * torch.where(m02 - m20 < 0.0, -torch.ones_like(qy), torch.ones_like(qy))
    qz = qz * torch.where(m10 - m01 < 0.0, -torch.ones_like(qz), torch.ones_like(qz))
    quat = torch.stack((qx, qy, qz, qw + zero), dim=-1)
    return normalize_quat_xyzw(quat)


def _normalize_index(index: Any, action_dim: int, field_name: str) -> int:
    try:
        normalized_index = operator.index(index)
    except TypeError as exc:
        msg = f"{field_name} contains a non-integer index: {index!r}."
        raise ValueError(msg) from exc

    if normalized_index < 0:
        normalized_index += action_dim
    if not 0 <= normalized_index < action_dim:
        msg = (
            f"{field_name} index {index!r} is out of bounds "
            f"for action_dim={action_dim}."
        )
        raise ValueError(msg)
    return normalized_index


def _slice_indices(slice_value: slice, action_dim: int, field_name: str) -> set[int]:
    if not isinstance(slice_value, slice):
        msg = f"{field_name} must be a slice."
        raise TypeError(msg)
    try:
        return set(range(action_dim)[slice_value])
    except ValueError as exc:
        msg = f"{field_name} is invalid for action_dim={action_dim}."
        raise ValueError(msg) from exc
