# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Canonical RoboTwin endpose16 bridge interfaces for residual TD3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch


class EndposeBridgeMode(str, Enum):
    """Supported ways to produce canonical RoboTwin endpose16 action chunks."""

    DIRECT_ENDPOSE16 = "direct_endpose16"
    CURRENT_STATE_EE_POSE = "current_state_ee_pose"
    LEARNED_QPOS_TO_ENDPOSE16 = "learned_qpos_to_endpose16"
    FK_QPOS_TO_ENDPOSE16 = "fk_qpos_to_endpose16"


@dataclass
class EndposeBridgeSpec:
    """Configuration for building canonical RoboTwin endpose16 chunks.

    Canonical ``robotwin_endpose16`` uses xyzw quaternions:
    ``[left xyz3, left quat_xyzw4, left gripper1,
    right xyz3, right quat_xyzw4, right gripper1]``.
    """

    mode: EndposeBridgeMode
    input_action_space: str
    output_action_space: str = "robotwin_endpose16"
    base_action_dim: int = 14
    output_action_dim: int = 16
    residual_chunk_len: int = 1
    env_action_chunk_len: Optional[int] = None
    right_xyz_indices: tuple[int, int, int] = (8, 9, 10)
    quat_order: str = "xyzw"
    require_current_ee_pose: bool = False
    allow_unimplemented_bridge: bool = False

    def __post_init__(self) -> None:
        self.mode = EndposeBridgeMode(self.mode)


class EndposeBridge:
    """Build base endpose16 chunks before residual extraction and composition.

    ``CURRENT_STATE_EE_POSE`` only repeats current EE pose and does not convert
    future qpos actions into future endpose actions. It is a smoke-test fallback,
    not a semantic replacement for a learned or FK qpos-to-endpose action bridge.
    """

    def __init__(self, spec: EndposeBridgeSpec) -> None:
        self.spec = spec
        self.validate()

    def build_base_endpose16_chunk(
        self,
        *,
        base_action_chunk: torch.Tensor,
        obs: Optional[dict[str, Any]] = None,
        bridge_model: Optional[Any] = None,
    ) -> torch.Tensor:
        """Return a canonical ``robotwin_endpose16`` chunk shaped ``[B, C, 16]``."""
        if bridge_model is not None:
            del bridge_model
        self._validate_base_action_chunk(base_action_chunk)
        mode = self.spec.mode
        if mode == EndposeBridgeMode.DIRECT_ENDPOSE16:
            return self._build_direct_endpose16(base_action_chunk)
        if mode == EndposeBridgeMode.CURRENT_STATE_EE_POSE:
            return self._build_current_state_ee_pose(base_action_chunk, obs)
        if mode == EndposeBridgeMode.LEARNED_QPOS_TO_ENDPOSE16:
            msg = (
                "future learned action bridge should map qpos14 action chunk + "
                "obs to endpose16 chunk."
            )
            raise NotImplementedError(msg)
        if mode == EndposeBridgeMode.FK_QPOS_TO_ENDPOSE16:
            msg = (
                "future FK bridge should map qpos target chunk through robot "
                "kinematics to endpose16 chunk."
            )
            raise NotImplementedError(msg)
        raise ValueError(f"Unsupported EndposeBridgeMode: {mode}.")

    def validate(self) -> None:
        """Validate static bridge configuration."""
        spec = self.spec
        if spec.output_action_space != "robotwin_endpose16":
            raise ValueError("output_action_space must be 'robotwin_endpose16'.")
        if spec.output_action_dim != 16:
            raise ValueError("output_action_dim must be 16.")
        if spec.quat_order != "xyzw":
            raise ValueError("quat_order must be 'xyzw'.")
        if spec.right_xyz_indices != (8, 9, 10):
            raise ValueError("right_xyz_indices must be (8, 9, 10).")
        if spec.residual_chunk_len <= 0:
            raise ValueError("residual_chunk_len must be positive.")
        if (
            spec.env_action_chunk_len is not None
            and spec.residual_chunk_len > spec.env_action_chunk_len
        ):
            raise ValueError("residual_chunk_len must be <= env_action_chunk_len.")
        if (
            spec.mode == EndposeBridgeMode.DIRECT_ENDPOSE16
            and spec.base_action_dim != 16
        ):
            raise ValueError("DIRECT_ENDPOSE16 requires base_action_dim=16.")

    def validate_endpose16_chunk(self, chunk: torch.Tensor) -> None:
        """Validate dynamic canonical endpose16 chunk shape."""
        if not isinstance(chunk, torch.Tensor):
            raise TypeError("chunk must be a torch.Tensor.")
        if chunk.ndim != 3:
            raise ValueError("endpose16 chunk must have shape [B, C, 16].")
        if chunk.shape[-1] != self.spec.output_action_dim:
            raise ValueError(
                f"endpose16 chunk must have last dim 16, got {chunk.shape[-1]}."
            )
        if chunk.shape[1] < self.spec.residual_chunk_len:
            raise ValueError(
                "residual_chunk_len must be <= available endpose chunk length."
            )

    def _validate_base_action_chunk(self, base_action_chunk: torch.Tensor) -> None:
        if not isinstance(base_action_chunk, torch.Tensor):
            raise TypeError("base_action_chunk must be a torch.Tensor.")
        if base_action_chunk.ndim != 3:
            raise ValueError("base_action_chunk must have shape [B, C, D].")
        if base_action_chunk.shape[-1] != self.spec.base_action_dim:
            raise ValueError(
                f"expected base_action_dim={self.spec.base_action_dim}, "
                f"got {base_action_chunk.shape[-1]}."
            )
        if base_action_chunk.shape[1] < self.spec.residual_chunk_len:
            raise ValueError(
                "residual_chunk_len must be <= available base action chunk length."
            )

    def _build_direct_endpose16(self, base_action_chunk: torch.Tensor) -> torch.Tensor:
        self.validate_endpose16_chunk(base_action_chunk)
        return base_action_chunk[:, : self.spec.residual_chunk_len, :].contiguous()

    def _build_current_state_ee_pose(
        self,
        base_action_chunk: torch.Tensor,
        obs: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        if obs is None:
            raise ValueError("CURRENT_STATE_EE_POSE requires obs.")
        batch_size = base_action_chunk.shape[0]
        left_pos = _require_obs_tensor(obs, "left_ee_pos", batch_size, 3)
        left_quat = _require_obs_tensor(obs, "left_ee_quat_xyzw", batch_size, 4)
        left_gripper = normalize_gripper_column(
            require_obs_tensor(obs, "left_gripper")
        )
        right_pos = _require_obs_tensor(obs, "right_ee_pos", batch_size, 3)
        right_quat = _require_obs_tensor(obs, "right_ee_quat_xyzw", batch_size, 4)
        right_gripper = normalize_gripper_column(
            require_obs_tensor(obs, "right_gripper")
        )
        _require_batch(left_gripper, batch_size, "left_gripper")
        _require_batch(right_gripper, batch_size, "right_gripper")
        pose = torch.cat(
            (
                left_pos,
                left_quat,
                left_gripper,
                right_pos,
                right_quat,
                right_gripper,
            ),
            dim=-1,
        )
        chunk = pose[:, None, :].repeat(1, self.spec.residual_chunk_len, 1)
        self.validate_endpose16_chunk(chunk)
        return chunk.contiguous()


def normalize_gripper_column(x: torch.Tensor) -> torch.Tensor:
    """Normalize gripper tensors from ``[B]`` or ``[B, 1]`` to ``[B, 1]``."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("gripper value must be a torch.Tensor.")
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError(f"gripper tensor must have shape [B] or [B, 1], got {x.shape}.")


def require_obs_tensor(obs: dict[str, Any], key: str) -> torch.Tensor:
    """Read an observation tensor or raise ``ValueError`` for missing keys."""
    if obs is None or key not in obs:
        raise ValueError(f"obs must contain '{key}'.")
    value = obs[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"obs['{key}'] must be a torch.Tensor.")
    return value


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion tensors from ``[..., x, y, z, w]`` to ``[..., w, x, y, z]``."""
    _require_quat(q, "q")
    return torch.cat((q[..., 3:4], q[..., 0:3]), dim=-1)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion tensors from ``[..., w, x, y, z]`` to ``[..., x, y, z, w]``."""
    _require_quat(q, "q")
    return torch.cat((q[..., 1:4], q[..., 0:1]), dim=-1)


def _require_obs_tensor(
    obs: dict[str, Any],
    key: str,
    batch_size: int,
    dim: int,
) -> torch.Tensor:
    tensor = require_obs_tensor(obs, key)
    if tensor.shape != (batch_size, dim):
        raise ValueError(
            f"obs['{key}'] must have shape [{batch_size}, {dim}], got {tensor.shape}."
        )
    return tensor


def _require_batch(tensor: torch.Tensor, batch_size: int, key: str) -> None:
    if tensor.shape != (batch_size, 1):
        raise ValueError(f"obs['{key}'] must have shape [{batch_size}] or [{batch_size}, 1].")


def _require_quat(q: torch.Tensor, name: str) -> None:
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if q.shape[-1:] != (4,):
        raise ValueError(f"{name} must have shape [..., 4], got {q.shape}.")
