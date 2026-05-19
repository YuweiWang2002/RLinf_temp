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
"""Action-space dependent residual composition utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from .schema import ResidualBaseActionSpace, ResidualFrame, ResidualMode


@dataclass
class ResidualActionSpec:
    """Configuration for residual action extraction and composition."""

    base_action_space: ResidualBaseActionSpace | str
    residual_mode: ResidualMode | str
    residual_frame: ResidualFrame | str = ResidualFrame.ACTION
    base_action_dim: int = 0
    residual_action_indices: Optional[list[int]] = None
    right_xyz_indices: list[int] = field(default_factory=lambda: [8, 9, 10])
    residual_chunk_len: int = 1
    env_action_chunk_len: Optional[int] = None
    base_action_source: str = "env_action"
    fail_on_incompatible_action_space: bool = True
    delta_max: Optional[float | Sequence[float]] = None
    clamp_residual: bool = True
    left_ee_rotation_key: str = "left_ee_rot"

    def __post_init__(self) -> None:
        self.base_action_space = ResidualBaseActionSpace(self.base_action_space)
        self.residual_mode = ResidualMode(self.residual_mode)
        self.residual_frame = ResidualFrame(self.residual_frame)

    @property
    def residual_action_dim(self) -> int:
        """Return residual dimensionality implied by the action space."""

        if self.base_action_space == ResidualBaseActionSpace.ALOHA_QPOS14:
            return len(self.residual_action_indices or [])
        if self.base_action_space == ResidualBaseActionSpace.ROBOTWIN_ENDPOSE16:
            return 3
        raise ValueError(f"Unsupported base_action_space: {self.base_action_space}")


class ResidualActionAdapter:
    """Extract residual references, compose actions, and compute BC targets."""

    def __init__(self, spec: ResidualActionSpec) -> None:
        self.spec = spec
        self.validate()

    def get_base_action_dim(self) -> int:
        return self.spec.base_action_dim

    def get_residual_action_dim(self) -> int:
        return self.spec.residual_action_dim

    def extract_residual_ref(
        self,
        base_action_chunk: torch.Tensor,
        obs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        del obs
        self._validate_base_action_chunk(base_action_chunk)
        if self.spec.base_action_space == ResidualBaseActionSpace.ALOHA_QPOS14:
            return base_action_chunk[..., self._aloha_indices()]
        return base_action_chunk[..., self.spec.right_xyz_indices]

    def compose_action(
        self,
        base_action_chunk: torch.Tensor,
        residual_chunk: torch.Tensor,
        gate: torch.Tensor,
        obs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        self._validate_base_action_chunk(base_action_chunk)
        residual_chunk = self._clamp_residual(residual_chunk)
        self._validate_residual_chunk(base_action_chunk, residual_chunk)
        gated_residual = residual_chunk * self._broadcast_gate(gate, residual_chunk)
        executed = base_action_chunk.clone()

        if self.spec.base_action_space == ResidualBaseActionSpace.ALOHA_QPOS14:
            executed[..., self._aloha_indices()] = (
                executed[..., self._aloha_indices()] + gated_residual
            )
            return executed

        transformed_residual = self._to_world_residual(gated_residual, obs)
        executed[..., self.spec.right_xyz_indices] = (
            executed[..., self.spec.right_xyz_indices] + transformed_residual
        )
        return executed

    def compute_bc_target(
        self,
        base_action_chunk: torch.Tensor,
        expert_action_chunk: torch.Tensor,
        obs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        self._validate_base_action_chunk(base_action_chunk)
        self._validate_base_action_chunk(expert_action_chunk)
        if base_action_chunk.shape != expert_action_chunk.shape:
            raise ValueError("base_action_chunk and expert_action_chunk must match.")

        if self.spec.base_action_space == ResidualBaseActionSpace.ALOHA_QPOS14:
            indices = self._aloha_indices()
            return expert_action_chunk[..., indices] - base_action_chunk[..., indices]

        delta = (
            expert_action_chunk[..., self.spec.right_xyz_indices]
            - base_action_chunk[..., self.spec.right_xyz_indices]
        )
        if self.spec.residual_frame == ResidualFrame.LEFT_EE:
            rot = self._get_left_ee_rot(obs, delta)
            return torch.einsum("...ji,...j->...i", rot, delta)
        return delta

    def validate(self) -> None:
        spec = self.spec
        if spec.residual_chunk_len <= 0:
            raise ValueError("residual_chunk_len must be positive.")
        if (
            spec.env_action_chunk_len is not None
            and spec.residual_chunk_len > spec.env_action_chunk_len
        ):
            raise ValueError("residual_chunk_len must be <= env_action_chunk_len.")

        if spec.base_action_space == ResidualBaseActionSpace.ALOHA_QPOS14:
            if spec.base_action_dim != 14:
                raise ValueError("aloha_qpos14 requires base_action_dim=14.")
            if spec.residual_mode != ResidualMode.JOINT_DELTA:
                raise ValueError("aloha_qpos14 only supports joint_delta residuals.")
            if spec.residual_action_indices is None:
                raise ValueError("aloha_qpos14 requires explicit residual_action_indices.")
            self._validate_indices(spec.residual_action_indices, spec.base_action_dim)
            if spec.residual_action_dim == 0:
                raise ValueError("residual_action_indices must not be empty.")
            return

        if spec.base_action_space == ResidualBaseActionSpace.ROBOTWIN_ENDPOSE16:
            if spec.base_action_dim != 16:
                raise ValueError("robotwin_endpose16 requires base_action_dim=16.")
            allowed_modes = {
                ResidualMode.RIGHT_XYZ_DELTA,
                ResidualMode.RIGHT_XYZ_LEFT_EE_FRAME,
                ResidualMode.RIGHT_XYZ_WORLD_FRAME,
            }
            if spec.residual_mode not in allowed_modes:
                raise ValueError("robotwin_endpose16 only supports right_xyz residuals.")
            if len(spec.right_xyz_indices) != 3:
                raise ValueError("right_xyz_indices must contain exactly 3 indices.")
            self._validate_indices(spec.right_xyz_indices, spec.base_action_dim)
            if spec.residual_mode == ResidualMode.RIGHT_XYZ_LEFT_EE_FRAME:
                if spec.residual_frame != ResidualFrame.LEFT_EE:
                    raise ValueError("right_xyz_left_ee_frame requires residual_frame=left_ee.")
            return

        raise ValueError(f"Unsupported base_action_space: {spec.base_action_space}")

    def _aloha_indices(self) -> list[int]:
        indices = self.spec.residual_action_indices
        if indices is None:
            raise ValueError("aloha_qpos14 requires explicit residual_action_indices.")
        return indices

    def _validate_base_action_chunk(self, action_chunk: torch.Tensor) -> None:
        if action_chunk.ndim != 3:
            raise ValueError("action chunk must have shape [B, C, D].")
        if action_chunk.shape[-1] != self.spec.base_action_dim:
            raise ValueError(
                f"expected base action dim {self.spec.base_action_dim}, "
                f"got {action_chunk.shape[-1]}."
            )

    def _validate_residual_chunk(
        self,
        base_action_chunk: torch.Tensor,
        residual_chunk: torch.Tensor,
    ) -> None:
        if residual_chunk.ndim != 3:
            raise ValueError("residual_chunk must have shape [B, C, K].")
        if residual_chunk.shape[:2] != base_action_chunk.shape[:2]:
            raise ValueError("residual_chunk must share [B, C] with base_action_chunk.")
        if residual_chunk.shape[-1] != self.spec.residual_action_dim:
            raise ValueError(
                f"expected residual dim {self.spec.residual_action_dim}, "
                f"got {residual_chunk.shape[-1]}."
            )

    def _broadcast_gate(self, gate: torch.Tensor, residual_chunk: torch.Tensor) -> torch.Tensor:
        if gate.ndim == 2 and gate.shape == (residual_chunk.shape[0], 1):
            gate = gate[:, None, :]
        elif gate.ndim == 3:
            valid_shape = (
                gate.shape[0] == residual_chunk.shape[0]
                and gate.shape[1] in (1, residual_chunk.shape[1])
                and gate.shape[2] == 1
            )
            if not valid_shape:
                raise ValueError("gate must have shape [B, 1], [B, 1, 1], or [B, C, 1].")
        else:
            raise ValueError("gate must have shape [B, 1], [B, 1, 1], or [B, C, 1].")
        try:
            return gate.expand_as(residual_chunk)
        except RuntimeError as exc:
            raise ValueError("gate shape is not broadcastable to residual_chunk.") from exc

    def _clamp_residual(self, residual_chunk: torch.Tensor) -> torch.Tensor:
        if not self.spec.clamp_residual or self.spec.delta_max is None:
            return residual_chunk
        delta_max = self.spec.delta_max
        if isinstance(delta_max, (float, int)):
            return residual_chunk.clamp(min=-float(delta_max), max=float(delta_max))
        limit = torch.as_tensor(delta_max, dtype=residual_chunk.dtype, device=residual_chunk.device)
        if limit.numel() != residual_chunk.shape[-1]:
            raise ValueError("delta_max sequence length must match residual_action_dim.")
        limit = limit.reshape(*([1] * (residual_chunk.ndim - 1)), -1)
        return torch.maximum(torch.minimum(residual_chunk, limit), -limit)

    def _to_world_residual(
        self,
        residual_chunk: torch.Tensor,
        obs: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        if self.spec.residual_frame != ResidualFrame.LEFT_EE:
            return residual_chunk
        rot = self._get_left_ee_rot(obs, residual_chunk)
        return torch.einsum("...ij,...j->...i", rot, residual_chunk)

    def _get_left_ee_rot(
        self,
        obs: Optional[dict[str, Any]],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if obs is None or self.spec.left_ee_rotation_key not in obs:
            raise ValueError(
                f"left_ee frame requires obs['{self.spec.left_ee_rotation_key}']."
            )
        rot = obs[self.spec.left_ee_rotation_key]
        if rot.ndim == 3 and rot.shape == (reference.shape[0], 3, 3):
            rot = rot[:, None, :, :]
        elif rot.ndim != 4:
            raise ValueError("left_ee_rot must have shape [B, 3, 3] or [B, C, 3, 3].")
        valid_shape = (
            rot.shape[0] == reference.shape[0]
            and rot.shape[1] in (1, reference.shape[1])
            and rot.shape[-2:] == (3, 3)
        )
        if not valid_shape:
            raise ValueError("left_ee_rot must have shape [B, 3, 3] or [B, C, 3, 3].")
        return rot.expand(reference.shape[0], reference.shape[1], 3, 3)

    @staticmethod
    def _validate_indices(indices: Sequence[int], action_dim: int) -> None:
        if len(set(indices)) != len(indices):
            raise ValueError("action indices must be unique.")
        for index in indices:
            if index < 0 or index >= action_dim:
                raise ValueError(f"action index {index} is out of range for dim {action_dim}.")
