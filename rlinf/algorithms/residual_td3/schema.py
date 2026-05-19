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
"""Typed schema for residual TD3 data exchanged through nested dicts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch


class ResidualBaseActionSpace(str, Enum):
    """Supported env-space base action layouts for residual TD3."""

    ALOHA_QPOS14 = "aloha_qpos14"
    ROBOTWIN_ENDPOSE16 = "robotwin_endpose16"


class ResidualMode(str, Enum):
    """Residual action semantics."""

    JOINT_DELTA = "joint_delta"
    RIGHT_XYZ_DELTA = "right_xyz_delta"
    RIGHT_XYZ_LEFT_EE_FRAME = "right_xyz_left_ee_frame"
    RIGHT_XYZ_WORLD_FRAME = "right_xyz_world_frame"


class ResidualFrame(str, Enum):
    """Coordinate frame used by residual vectors."""

    ACTION = "action"
    WORLD = "world"
    LEFT_EE = "left_ee"


@dataclass
class ResidualObs:
    """Residual actor input.

    Shapes:
        vla_feature: ``[B, vla_feature_dim]`` when present.
        rel_state: ``[B, rel_state_dim]`` when present.
        base_action_chunk: ``[B, C, base_action_dim]``.
        residual_ref_chunk: ``[B, C, residual_action_dim]``.
        history: ``[B, history_dim]`` when present.
        gate: ``[B, 1]``, ``[B, 1, 1]``, or ``[B, C, 1]``.
        action_mask: ``[B, C, residual_action_dim]`` when present.
    """

    vla_feature: Optional[torch.Tensor]
    rel_state: Optional[torch.Tensor]
    base_action_chunk: torch.Tensor
    residual_ref_chunk: torch.Tensor
    history: Optional[torch.Tensor]
    gate: torch.Tensor
    action_mask: Optional[torch.Tensor]
    metadata: Optional[dict[str, Any]] = None

    def validate_shape(self) -> None:
        """Validate only lightweight batch/chunk consistency."""

        if self.base_action_chunk.ndim != 3:
            raise ValueError("base_action_chunk must have shape [B, C, D].")
        if self.residual_ref_chunk.ndim != 3:
            raise ValueError("residual_ref_chunk must have shape [B, C, K].")
        if self.base_action_chunk.shape[:2] != self.residual_ref_chunk.shape[:2]:
            raise ValueError("base_action_chunk and residual_ref_chunk must share [B, C].")
        batch_size, chunk_len = self.base_action_chunk.shape[:2]
        if self.gate.shape[0] != batch_size:
            raise ValueError("gate must share the batch dimension with base_action_chunk.")
        if self.action_mask is not None:
            if self.action_mask.shape != self.residual_ref_chunk.shape:
                raise ValueError("action_mask must match residual_ref_chunk shape.")
        for name, tensor in (
            ("vla_feature", self.vla_feature),
            ("rel_state", self.rel_state),
            ("history", self.history),
        ):
            if tensor is not None and tensor.shape[0] != batch_size:
                raise ValueError(f"{name} must share the batch dimension.")
        if self.gate.ndim == 3 and self.gate.shape[1] not in (1, chunk_len):
            raise ValueError("gate chunk dimension must be 1 or C.")


@dataclass
class ResidualTransition:
    """Replay transition schema for residual TD3."""

    curr_residual_obs: ResidualObs
    next_residual_obs: ResidualObs
    base_action_chunk: torch.Tensor
    residual_action_chunk: torch.Tensor
    executed_action_chunk: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    success: Optional[torch.Tensor] = None
    discount: Optional[torch.Tensor] = None
    info: Optional[dict[str, Any]] = None

    def validate_shape(self) -> None:
        """Validate lightweight replay tensor shape consistency."""

        if self.base_action_chunk.ndim != 3:
            raise ValueError("base_action_chunk must have shape [B, C, D].")
        if self.residual_action_chunk.ndim != 3:
            raise ValueError("residual_action_chunk must have shape [B, C, K].")
        if self.executed_action_chunk.shape != self.base_action_chunk.shape:
            raise ValueError("executed_action_chunk must match base_action_chunk shape.")
        if self.base_action_chunk.shape[:2] != self.residual_action_chunk.shape[:2]:
            raise ValueError("base_action_chunk and residual_action_chunk must share [B, C].")


@dataclass
class ExpertResidualSample:
    """BC sample schema for residual pretraining."""

    residual_obs: ResidualObs
    base_action_chunk: torch.Tensor
    expert_action_chunk: torch.Tensor
    bc_residual_target: torch.Tensor
    bc_mask: torch.Tensor
    task_name: Optional[str | list[str]] = None
    episode_id: Optional[torch.Tensor | list[int]] = None
    timestep: Optional[torch.Tensor] = None

    def validate_shape(self) -> None:
        """Validate lightweight BC tensor shape consistency."""

        if self.base_action_chunk.shape != self.expert_action_chunk.shape:
            raise ValueError("base_action_chunk and expert_action_chunk must match.")
        if self.bc_residual_target.shape != self.bc_mask.shape:
            raise ValueError("bc_residual_target and bc_mask must match.")
        if self.base_action_chunk.shape[:2] != self.bc_residual_target.shape[:2]:
            raise ValueError("BC tensors must share [B, C].")
