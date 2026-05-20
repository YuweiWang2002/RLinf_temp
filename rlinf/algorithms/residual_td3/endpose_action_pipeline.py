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
"""FK-backed qpos14 -> endpose16 residual action pipeline helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .action_adapter import ResidualActionAdapter


@dataclass(frozen=True)
class EndposeActionPipelineConfig:
    """Configuration for the FK-backed residual endpose action pipeline."""

    base_action_space: str = "aloha_qpos14"
    bridge_type: str = "fk"
    residual_chunk_len: int = 1
    env_action_chunk_len: int | None = None
    right_xyz_indices: tuple[int, int, int] = (8, 9, 10)
    robotwin_action_mode: str = "ee16"


@dataclass(frozen=True)
class EndposeActionPipelineOutput:
    """Tensors produced by a zero-residual endpose pipeline pass."""

    base_endpose16_chunk: torch.Tensor
    residual_ref_chunk: torch.Tensor
    zero_residual_chunk: torch.Tensor
    executed_endpose16_chunk: torch.Tensor


class Qpos14ToEndpose16Bridge(Protocol):
    """Bridge contract used by :class:`EndposeActionPipeline`."""

    def qpos14_to_endpose16(
        self,
        qpos14_chunk: torch.Tensor,
        current_obs=None,
    ) -> torch.Tensor:
        """Return a canonical ``[B, C, 16]`` endpose chunk."""


class EndposeActionPipeline:
    """Compose FK base actions with right-xyz residuals for RoboTwin ee16."""

    def __init__(
        self,
        fk_bridge: Qpos14ToEndpose16Bridge,
        residual_adapter: ResidualActionAdapter,
        config: EndposeActionPipelineConfig,
    ) -> None:
        self.fk_bridge = fk_bridge
        self.residual_adapter = residual_adapter
        self.config = config
        self.validate()

    def qpos14_chunk_to_base_endpose16(
        self,
        qpos14_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Convert ``[B, C, 14]`` qpos targets to ``[B, C, 16]`` endpose."""
        self._validate_qpos14_chunk(qpos14_chunk)
        out = self.fk_bridge.qpos14_to_endpose16(qpos14_chunk)
        self._validate_endpose16_chunk(out)
        return out.contiguous()

    def extract_right_xyz_ref(
        self,
        base_endpose16_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the right EE xyz reference chunk, shaped ``[B, C, 3]``."""
        self._validate_endpose16_chunk(base_endpose16_chunk)
        ref = self.residual_adapter.extract_residual_ref(base_endpose16_chunk)
        if ref.ndim != 3 or ref.shape[:2] != base_endpose16_chunk.shape[:2] or ref.shape[-1] != 3:
            raise ValueError(f"residual ref must have shape [B, C, 3], got {ref.shape}.")
        return ref.contiguous()

    def compose_executed_endpose16(
        self,
        base_endpose16_chunk: torch.Tensor,
        residual_xyz_chunk: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply right-xyz residuals through ``ResidualActionAdapter.compose_action``."""
        self._validate_endpose16_chunk(base_endpose16_chunk)
        self._validate_residual_xyz_chunk(base_endpose16_chunk, residual_xyz_chunk)
        executed = self.residual_adapter.compose_action(
            base_endpose16_chunk,
            residual_xyz_chunk,
            gate,
        )
        self._validate_endpose16_chunk(executed)
        return executed.contiguous()

    def build_zero_residual_action(
        self,
        qpos14_chunk: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> EndposeActionPipelineOutput:
        """Build FK base, right-xyz ref, zero residual, and executed ee16 chunk."""
        base = self.qpos14_chunk_to_base_endpose16(qpos14_chunk)
        ref = self.extract_right_xyz_ref(base)
        zero = torch.zeros_like(ref).contiguous()
        if gate is None:
            gate = torch.ones(
                base.shape[0],
                1,
                dtype=base.dtype,
                device=base.device,
            )
        executed = self.compose_executed_endpose16(base, zero, gate)
        return EndposeActionPipelineOutput(
            base_endpose16_chunk=base.contiguous(),
            residual_ref_chunk=ref.contiguous(),
            zero_residual_chunk=zero,
            executed_endpose16_chunk=executed.contiguous(),
        )

    def validate(self) -> None:
        """Validate that only the first supported pipeline mode is active."""
        cfg = self.config
        if cfg.base_action_space != "aloha_qpos14":
            raise NotImplementedError("EndposeActionPipeline only supports aloha_qpos14 input.")
        if cfg.bridge_type != "fk":
            raise NotImplementedError("EndposeActionPipeline only supports bridge_type='fk'.")
        if cfg.robotwin_action_mode not in ("ee16", "endpose16"):
            raise NotImplementedError("EndposeActionPipeline requires robotwin_action_mode='ee16'.")
        if cfg.right_xyz_indices != (8, 9, 10):
            raise NotImplementedError("EndposeActionPipeline only supports right_xyz_indices=(8, 9, 10).")
        if cfg.residual_chunk_len <= 0:
            raise ValueError("residual_chunk_len must be positive.")
        if cfg.env_action_chunk_len is not None and cfg.residual_chunk_len > cfg.env_action_chunk_len:
            raise ValueError("residual_chunk_len must be <= env_action_chunk_len.")

        spec = self.residual_adapter.spec
        if str(spec.base_action_space.value) != "robotwin_endpose16":
            raise NotImplementedError("residual_adapter must target robotwin_endpose16.")
        if str(spec.residual_mode.value) != "right_xyz_world_frame":
            raise NotImplementedError("residual_adapter must use right_xyz_world_frame.")
        if str(spec.residual_frame.value) != "world":
            raise NotImplementedError("residual_adapter must use residual_frame='world'.")
        if tuple(spec.right_xyz_indices) != cfg.right_xyz_indices:
            raise ValueError("pipeline and residual_adapter right_xyz_indices must match.")

    @staticmethod
    def _validate_qpos14_chunk(qpos14_chunk: torch.Tensor) -> None:
        if not isinstance(qpos14_chunk, torch.Tensor):
            raise TypeError("qpos14_chunk must be a torch.Tensor.")
        if qpos14_chunk.ndim != 3 or qpos14_chunk.shape[-1] != 14:
            raise ValueError(f"qpos14_chunk must have shape [B, C, 14], got {qpos14_chunk.shape}.")

    @staticmethod
    def _validate_endpose16_chunk(endpose16_chunk: torch.Tensor) -> None:
        if not isinstance(endpose16_chunk, torch.Tensor):
            raise TypeError("endpose16_chunk must be a torch.Tensor.")
        if endpose16_chunk.ndim != 3 or endpose16_chunk.shape[-1] != 16:
            raise ValueError(f"endpose16_chunk must have shape [B, C, 16], got {endpose16_chunk.shape}.")

    @staticmethod
    def _validate_residual_xyz_chunk(
        base_endpose16_chunk: torch.Tensor,
        residual_xyz_chunk: torch.Tensor,
    ) -> None:
        if not isinstance(residual_xyz_chunk, torch.Tensor):
            raise TypeError("residual_xyz_chunk must be a torch.Tensor.")
        if residual_xyz_chunk.ndim != 3:
            raise ValueError("residual_xyz_chunk must have shape [B, C, 3].")
        if residual_xyz_chunk.shape[:2] != base_endpose16_chunk.shape[:2]:
            raise ValueError("residual_xyz_chunk must share [B, C] with base_endpose16_chunk.")
        if residual_xyz_chunk.shape[-1] != 3:
            raise ValueError(f"residual_xyz_chunk must have last dim 3, got {residual_xyz_chunk.shape[-1]}.")
