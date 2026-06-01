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
"""Build residual BC targets in the base left-EE frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from .residual_ee_intervention import (
    ENDPOSE16_LEFT_GRIPPER,
    ENDPOSE16_LEFT_QUAT_WXYZ,
    ENDPOSE16_LEFT_XYZ,
    ENDPOSE16_RIGHT_GRIPPER,
    ENDPOSE16_RIGHT_QUAT_WXYZ,
    ENDPOSE16_RIGHT_XYZ,
    quat_wxyz_to_matrix,
)


class Qpos14ToEndpose16Bridge(Protocol):
    """Bridge contract needed by residual BC target construction."""

    def qpos14_to_endpose16(self, qpos14_chunk: torch.Tensor, current_obs=None) -> torch.Tensor:
        """Convert ``[B, C, 14]`` qpos chunks to ``[B, C, 16]`` endpose chunks."""


@dataclass(frozen=True)
class ResidualBCTargetConfig:
    """Configuration for residual BC target construction."""

    chunk_len: int = 50
    residual_horizon_k: int = 10
    target_horizon_offsets: tuple[int, ...] = (0, 1, 2)
    max_delta_local_xyz: float = 0.05
    gate_positive_only: bool = True
    min_positive_frames: int = 1
    base_action_source: str = "expert_as_base"


class ResidualBCTargetBuilder:
    """Construct right-xyz residual targets in the base left-EE frame."""

    def __init__(
        self,
        cfg: ResidualBCTargetConfig | None,
        fk_bridge: Qpos14ToEndpose16Bridge,
    ) -> None:
        self.cfg = cfg or ResidualBCTargetConfig()
        self.fk_bridge = fk_bridge
        self._validate_config()

    def build_sample(
        self,
        *,
        base_qpos_chunk,
        expert_qpos_chunk,
        gate_seq,
        current_state=None,
        episode_index: int | None = None,
        frame_index: int | None = None,
    ) -> dict[str, object] | None:
        """Return one residual BC sample, or ``None`` when gate filtering drops it."""
        base_qpos = self._as_qpos_tensor(base_qpos_chunk, "base_qpos_chunk")
        expert_qpos = self._as_qpos_tensor(expert_qpos_chunk, "expert_qpos_chunk")
        gate = self._as_gate_tensor(gate_seq)
        if not self.should_keep_gate_seq(gate):
            return None

        base_endpose = self._fk(base_qpos)
        expert_endpose = base_endpose.clone() if torch.equal(base_qpos, expert_qpos) else self._fk(expert_qpos)
        return self.build_sample_from_endpose(
            base_endpose16=base_endpose,
            expert_endpose16=expert_endpose,
            gate_seq=gate,
            current_state=current_state,
            episode_index=episode_index,
            frame_index=frame_index,
        )

    def build_sample_from_endpose(
        self,
        *,
        base_endpose16,
        expert_endpose16,
        gate_seq,
        current_state=None,
        episode_index: int | None = None,
        frame_index: int | None = None,
    ) -> dict[str, object] | None:
        """Return one residual BC sample from precomputed endpose chunks."""
        base_endpose = self._as_endpose_tensor(base_endpose16, "base_endpose16")
        expert_endpose = self._as_endpose_tensor(expert_endpose16, "expert_endpose16")
        gate = self._as_gate_tensor(gate_seq)
        if not self.should_keep_gate_seq(gate):
            return None
        targets = self.build_targets_for_offsets(
            base_endpose16=base_endpose,
            expert_endpose16=expert_endpose,
        )
        obs_features = self._build_obs_features(
            base_endpose16=base_endpose,
            gate_seq=gate,
            current_state=current_state,
        )
        metadata = {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "target_horizon_offsets": list(self.cfg.target_horizon_offsets),
            "gate_positive_count": int((gate > 0.0).sum().item()),
            "base_action_source": self.cfg.base_action_source,
        }
        return {
            "obs_features": obs_features,
            "target_delta_local_xyz": targets["delta_local_clipped"].detach().cpu().numpy(),
            "target_delta_local_xyz_unclipped": targets["delta_local"].detach().cpu().numpy(),
            "target_delta_world_xyz": targets["delta_world"].detach().cpu().numpy(),
            "base_endpose16": base_endpose.detach().cpu().numpy(),
            "expert_endpose16": expert_endpose.detach().cpu().numpy(),
            "gate_seq": gate.detach().cpu().numpy(),
            "clip_mask": targets["clip_mask"].detach().cpu().numpy(),
            "delta_local_norm_unclipped": targets["delta_local_norm"].detach().cpu().numpy(),
            "delta_local_norm_clipped": targets["delta_local_clipped_norm"].detach().cpu().numpy(),
            "metadata": metadata,
        }

    def build_targets_for_offsets(
        self,
        *,
        base_endpose16: torch.Tensor,
        expert_endpose16: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build target deltas for configured horizon offsets."""
        self._validate_endpose_chunk(base_endpose16, "base_endpose16")
        self._validate_endpose_chunk(expert_endpose16, "expert_endpose16")
        if base_endpose16.shape != expert_endpose16.shape:
            raise ValueError("base_endpose16 and expert_endpose16 must have the same shape.")

        delta_world = []
        delta_local = []
        delta_local_clipped = []
        clip_mask = []
        for offset in self.cfg.target_horizon_offsets:
            base_ee = base_endpose16[offset]
            expert_ee = expert_endpose16[offset]
            world = expert_ee[ENDPOSE16_RIGHT_XYZ] - base_ee[ENDPOSE16_RIGHT_XYZ]
            rotation = quat_wxyz_to_matrix(base_ee[ENDPOSE16_LEFT_QUAT_WXYZ]).to(
                device=base_ee.device,
                dtype=base_ee.dtype,
            )
            local = rotation.T @ world
            clipped = torch.clamp(
                local,
                min=-float(self.cfg.max_delta_local_xyz),
                max=float(self.cfg.max_delta_local_xyz),
            )
            delta_world.append(world)
            delta_local.append(local)
            delta_local_clipped.append(clipped)
            clip_mask.append(torch.ne(local, clipped))

        local_tensor = torch.stack(delta_local, dim=0)
        clipped_tensor = torch.stack(delta_local_clipped, dim=0)
        return {
            "delta_world": torch.stack(delta_world, dim=0),
            "delta_local": local_tensor,
            "delta_local_clipped": clipped_tensor,
            "clip_mask": torch.stack(clip_mask, dim=0),
            "delta_local_norm": torch.linalg.norm(local_tensor, dim=-1),
            "delta_local_clipped_norm": torch.linalg.norm(clipped_tensor, dim=-1),
        }

    def should_keep_gate_seq(self, gate_seq: torch.Tensor | np.ndarray) -> bool:
        """Return whether ``gate_seq`` passes the configured positive-frame filter."""
        gate = self._as_gate_tensor(gate_seq)
        if not self.cfg.gate_positive_only:
            return True
        return int((gate > 0.0).sum().item()) >= self.cfg.min_positive_frames

    def _build_obs_features(
        self,
        *,
        base_endpose16: torch.Tensor,
        gate_seq: torch.Tensor,
        current_state,
    ) -> dict[str, np.ndarray]:
        offsets = self.cfg.target_horizon_offsets
        rows: dict[str, list[np.ndarray | float]] = {
            "base_left_xyz": [],
            "base_left_quat_wxyz": [],
            "base_right_xyz": [],
            "base_right_quat_wxyz": [],
            "relative_right_xyz_in_left_frame": [],
            "left_gripper": [],
            "right_gripper": [],
            "gate_label": [],
            "horizon_offset": [],
        }
        for offset in offsets:
            base_ee = base_endpose16[offset]
            left_xyz = base_ee[ENDPOSE16_LEFT_XYZ]
            left_quat = base_ee[ENDPOSE16_LEFT_QUAT_WXYZ]
            right_xyz = base_ee[ENDPOSE16_RIGHT_XYZ]
            rotation = quat_wxyz_to_matrix(left_quat).to(device=base_ee.device, dtype=base_ee.dtype)
            relative_left = rotation.T @ (right_xyz - left_xyz)
            rows["base_left_xyz"].append(left_xyz.detach().cpu().numpy())
            rows["base_left_quat_wxyz"].append(left_quat.detach().cpu().numpy())
            rows["base_right_xyz"].append(right_xyz.detach().cpu().numpy())
            rows["base_right_quat_wxyz"].append(
                base_ee[ENDPOSE16_RIGHT_QUAT_WXYZ].detach().cpu().numpy()
            )
            rows["relative_right_xyz_in_left_frame"].append(relative_left.detach().cpu().numpy())
            rows["left_gripper"].append(float(base_ee[ENDPOSE16_LEFT_GRIPPER].detach().cpu()))
            rows["right_gripper"].append(float(base_ee[ENDPOSE16_RIGHT_GRIPPER].detach().cpu()))
            rows["gate_label"].append(float(gate_seq[offset].detach().cpu()))
            rows["horizon_offset"].append(float(offset))

        out = {
            key: np.asarray(values, dtype=np.float32)
            for key, values in rows.items()
        }
        if current_state is not None:
            out["current_qpos14_state"] = np.asarray(current_state, dtype=np.float32).reshape(-1)
        return out

    def _fk(self, qpos_chunk: torch.Tensor) -> torch.Tensor:
        endpose = self.fk_bridge.qpos14_to_endpose16(qpos_chunk.unsqueeze(0))[0]
        self._validate_endpose_chunk(endpose, "fk_endpose16")
        return endpose.to(device=qpos_chunk.device, dtype=qpos_chunk.dtype)

    def _as_qpos_tensor(self, value, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2 or tensor.shape[-1] != 14:
            raise ValueError(f"{name} must have shape [C, 14] or [1, C, 14], got {tuple(tensor.shape)}.")
        if tensor.shape[0] < self.cfg.chunk_len:
            raise ValueError(f"{name} length must be at least chunk_len={self.cfg.chunk_len}.")
        return tensor[: self.cfg.chunk_len].contiguous()

    def _as_gate_tensor(self, value) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if tensor.shape[0] < self.cfg.chunk_len:
            raise ValueError("gate_seq length must be at least chunk_len.")
        return tensor[: self.cfg.chunk_len].contiguous()

    def _validate_endpose_chunk(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.ndim != 2 or tensor.shape[-1] != 16:
            raise ValueError(f"{name} must have shape [C, 16], got {tuple(tensor.shape)}.")
        if tensor.shape[0] <= max(self.cfg.target_horizon_offsets):
            raise ValueError(f"{name} does not cover configured target_horizon_offsets.")

    def _as_endpose_tensor(self, value, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 2 or tensor.shape[-1] != 16:
            raise ValueError(f"{name} must have shape [C, 16] or [1, C, 16], got {tuple(tensor.shape)}.")
        if tensor.shape[0] < self.cfg.chunk_len:
            raise ValueError(f"{name} length must be at least chunk_len={self.cfg.chunk_len}.")
        return tensor[: self.cfg.chunk_len].contiguous()

    def _validate_config(self) -> None:
        cfg = self.cfg
        if cfg.chunk_len <= 0:
            raise ValueError("chunk_len must be positive.")
        if cfg.residual_horizon_k <= 0:
            raise ValueError("residual_horizon_k must be positive.")
        if not cfg.target_horizon_offsets:
            raise ValueError("target_horizon_offsets cannot be empty.")
        if any(offset < 0 for offset in cfg.target_horizon_offsets):
            raise ValueError("target_horizon_offsets must be non-negative.")
        if max(cfg.target_horizon_offsets) >= cfg.chunk_len:
            raise ValueError("target_horizon_offsets must be smaller than chunk_len.")
        if max(cfg.target_horizon_offsets) >= cfg.residual_horizon_k:
            raise ValueError("target_horizon_offsets must be smaller than residual_horizon_k.")
        if cfg.max_delta_local_xyz <= 0:
            raise ValueError("max_delta_local_xyz must be positive.")
        if cfg.min_positive_frames <= 0:
            raise ValueError("min_positive_frames must be positive.")
        if cfg.base_action_source not in ("expert_as_base", "pi05_cached_base"):
            raise ValueError("base_action_source must be expert_as_base or pi05_cached_base.")
