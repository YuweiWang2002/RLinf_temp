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
"""Safe EE residual intervention helpers for gated rollout smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .residual_actor import ResidualActorConfig, ZeroInitResidualActorMLP

QPOS14_LEFT_JOINTS = slice(0, 6)
QPOS14_LEFT_GRIPPER = 6
QPOS14_RIGHT_JOINTS = slice(7, 13)
QPOS14_RIGHT_GRIPPER = 13

ENDPOSE16_LEFT_XYZ = slice(0, 3)
ENDPOSE16_LEFT_QUAT_WXYZ = slice(3, 7)
ENDPOSE16_LEFT_GRIPPER = 7
ENDPOSE16_RIGHT_XYZ = slice(8, 11)
ENDPOSE16_RIGHT_QUAT_WXYZ = slice(11, 15)
ENDPOSE16_RIGHT_GRIPPER = 15


class Qpos14ToEndpose16Bridge(Protocol):
    """Bridge contract needed by residual EE intervention."""

    def qpos14_to_endpose16(self, qpos14_chunk: torch.Tensor, current_obs=None) -> torch.Tensor:
        """Convert ``[B, C, 14]`` qpos to ``[B, C, 16]`` endpose."""


@dataclass(frozen=True)
class ResidualEEInterventionConfig:
    """Configuration for K-step EE residual intervention."""

    horizon_k: int = 5
    target_horizon_offset: int = 0
    residual_frame: str = "left_ee"
    left_frame_source: str = "base_target"
    max_delta_local_xyz: float | tuple[float, float, float] = 0.02
    left_stabilization_mode: str = "none"
    left_deadband_xyz: float = 1e-4
    left_lowpass_alpha: float = 0.5


@dataclass(frozen=True)
class ResidualEEObservation:
    """Minimal residual actor observation for smoke-test actors."""

    left_xyz: torch.Tensor
    left_quat_wxyz: torch.Tensor
    right_xyz: torch.Tensor
    right_quat_wxyz: torch.Tensor
    relative_xyz_left_frame: torch.Tensor
    left_gripper: torch.Tensor
    right_gripper: torch.Tensor
    gate_score: float | None
    intervention_step_i: int


class ResidualEEActor(Protocol):
    """Residual actor interface used before TD3 exists."""

    def predict_delta_local_xyz(self, obs: ResidualEEObservation) -> torch.Tensor:
        """Return one local-frame right-xyz residual with shape ``[3]``."""


class ZeroResidualActor:
    """Actor that always returns zero local xyz residual."""

    def predict_delta_local_xyz(self, obs: ResidualEEObservation) -> torch.Tensor:
        del obs
        return torch.zeros(3, dtype=torch.float32)


@dataclass(frozen=True)
class ConstantResidualActor:
    """Actor that returns a fixed local xyz residual."""

    delta_local_xyz: tuple[float, float, float]

    def predict_delta_local_xyz(self, obs: ResidualEEObservation) -> torch.Tensor:
        del obs
        return torch.tensor(self.delta_local_xyz, dtype=torch.float32)


class ZeroInitResidualActor:
    """Tiny zero-initialized residual actor for inference-path validation."""

    def __init__(
        self,
        obs_dim: int = 21,
        hidden_dim: int = 128,
        chunk_len: int = 5,
        delta_max: float = 0.02,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = ZeroInitResidualActorMLP(
            ResidualActorConfig(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                chunk_len=chunk_len,
                residual_dim=3,
                delta_max=delta_max,
                zero_init_output=True,
            )
        ).to(self.device)

    def predict_delta_local_xyz(self, obs: ResidualEEObservation) -> torch.Tensor:
        vector = residual_ee_observation_to_tensor(obs).reshape(1, -1).to(self.device)
        with torch.no_grad():
            return self.model(vector)[0, 0]


@dataclass(frozen=True)
class ResidualEEInterventionResult:
    """Output and metadata for one intervention."""

    base_ee16_chunk: torch.Tensor
    exec_ee16_chunk: torch.Tensor
    records: list[dict[str, object]]
    metadata: dict[str, object]


class ResidualEEInterventionRunner:
    """Build K-step EE residual intervention chunks from pi05 qpos targets."""

    def __init__(
        self,
        bridge: Qpos14ToEndpose16Bridge,
        actor: ResidualEEActor | None = None,
        config: ResidualEEInterventionConfig | None = None,
    ) -> None:
        self.bridge = bridge
        self.actor = actor or ZeroResidualActor()
        self.config = config or ResidualEEInterventionConfig()
        self._validate_config()

    def run(
        self,
        qpos14_chunk: torch.Tensor,
        *,
        gate_score: float | None = None,
        gate_threshold: float | None = None,
        episode_id: int | None = None,
        env_step: int | None = None,
        intervention_id: int | None = None,
    ) -> ResidualEEInterventionResult:
        """Build an executable EE16 intervention chunk and per-step records."""
        self._validate_qpos14_chunk(qpos14_chunk)
        selected_indices = self._select_indices(qpos14_chunk.shape[1])
        selected_qpos = qpos14_chunk[:, selected_indices, :].contiguous()
        base_ee16 = self.bridge.qpos14_to_endpose16(selected_qpos).to(
            device=qpos14_chunk.device,
            dtype=qpos14_chunk.dtype,
        )
        if base_ee16.ndim != 3 or base_ee16.shape[:2] != selected_qpos.shape[:2] or base_ee16.shape[-1] != 16:
            raise ValueError(f"bridge must return shape [B, K, 16], got {tuple(base_ee16.shape)}.")

        exec_ee16 = base_ee16.clone()
        records: list[dict[str, object]] = []
        delta_norms = []
        stabilization_count = 0
        previous_left_pose: torch.Tensor | None = None
        max_delta = self._max_delta_tensor(qpos14_chunk.device, qpos14_chunk.dtype)

        for local_i, chunk_index in enumerate(selected_indices):
            base_step = base_ee16[0, local_i]
            residual_obs = self._build_observation(base_step, gate_score, local_i)
            delta_local = self.actor.predict_delta_local_xyz(residual_obs).to(
                device=qpos14_chunk.device,
                dtype=qpos14_chunk.dtype,
            )
            if delta_local.shape != (3,):
                raise ValueError(f"delta_local_xyz must have shape [3], got {tuple(delta_local.shape)}.")
            delta_local = torch.clamp(delta_local, min=-max_delta, max=max_delta)
            delta_world = quat_wxyz_to_matrix(base_step[ENDPOSE16_LEFT_QUAT_WXYZ]) @ delta_local
            exec_ee16[0, local_i, ENDPOSE16_RIGHT_XYZ] += delta_world
            stabilized, previous_left_pose = self._stabilize_left_pose(
                exec_ee16[0, local_i],
                previous_left_pose,
            )
            stabilization_count += int(stabilized)
            delta_norms.append(float(torch.linalg.norm(delta_local).detach().cpu()))
            records.append(
                {
                    "episode_id": episode_id,
                    "env_step": env_step,
                    "intervention_id": intervention_id,
                    "intervention_step_i": local_i,
                    "gate_score": gate_score,
                    "gate_threshold": gate_threshold,
                    "action_chunk_index": int(chunk_index),
                    "base_qpos14": selected_qpos[0, local_i].detach().cpu().tolist(),
                    "base_ee16": base_step.detach().cpu().tolist(),
                    "exec_ee16": exec_ee16[0, local_i].detach().cpu().tolist(),
                    "delta_local_xyz": delta_local.detach().cpu().tolist(),
                    "delta_world_xyz": delta_world.detach().cpu().tolist(),
                    "left_xyz": base_step[ENDPOSE16_LEFT_XYZ].detach().cpu().tolist(),
                    "left_quat": base_step[ENDPOSE16_LEFT_QUAT_WXYZ].detach().cpu().tolist(),
                    "right_xyz_base": base_step[ENDPOSE16_RIGHT_XYZ].detach().cpu().tolist(),
                    "right_xyz_exec": exec_ee16[0, local_i, ENDPOSE16_RIGHT_XYZ].detach().cpu().tolist(),
                    "residual_frame": self.config.residual_frame,
                    "left_frame_source": self.config.left_frame_source,
                    "d_LR": float(
                        torch.linalg.norm(
                            base_step[ENDPOSE16_RIGHT_XYZ] - base_step[ENDPOSE16_LEFT_XYZ]
                        ).detach().cpu()
                    ),
                    "left_gripper": float(base_step[ENDPOSE16_LEFT_GRIPPER].detach().cpu()),
                    "right_gripper": float(base_step[ENDPOSE16_RIGHT_GRIPPER].detach().cpu()),
                }
            )

        metadata = {
            "num_steps_executed": int(len(selected_indices)),
            "selected_indices": [int(index) for index in selected_indices],
            "max_delta_norm": float(max(delta_norms, default=0.0)),
            "mean_delta_norm": float(sum(delta_norms) / len(delta_norms)) if delta_norms else 0.0,
            "left_stabilization_count": int(stabilization_count),
            "records": records,
        }
        return ResidualEEInterventionResult(
            base_ee16_chunk=base_ee16.contiguous(),
            exec_ee16_chunk=exec_ee16.contiguous(),
            records=records,
            metadata=metadata,
        )

    def _select_indices(self, chunk_len: int) -> list[int]:
        start = self.config.target_horizon_offset
        end = min(chunk_len, start + self.config.horizon_k)
        if start >= chunk_len:
            raise ValueError(
                "target_horizon_offset must select at least one action; "
                f"offset={start}, chunk_len={chunk_len}."
            )
        return list(range(start, end))

    def _build_observation(
        self,
        base_step: torch.Tensor,
        gate_score: float | None,
        intervention_step_i: int,
    ) -> ResidualEEObservation:
        left_xyz = base_step[ENDPOSE16_LEFT_XYZ]
        left_quat = base_step[ENDPOSE16_LEFT_QUAT_WXYZ]
        right_xyz = base_step[ENDPOSE16_RIGHT_XYZ]
        rel_world = right_xyz - left_xyz
        rel_left = quat_wxyz_to_matrix(left_quat).T @ rel_world
        return ResidualEEObservation(
            left_xyz=left_xyz,
            left_quat_wxyz=left_quat,
            right_xyz=right_xyz,
            right_quat_wxyz=base_step[ENDPOSE16_RIGHT_QUAT_WXYZ],
            relative_xyz_left_frame=rel_left,
            left_gripper=base_step[ENDPOSE16_LEFT_GRIPPER],
            right_gripper=base_step[ENDPOSE16_RIGHT_GRIPPER],
            gate_score=gate_score,
            intervention_step_i=intervention_step_i,
        )

    def _stabilize_left_pose(
        self,
        exec_step: torch.Tensor,
        previous_left_pose: torch.Tensor | None,
    ) -> tuple[bool, torch.Tensor]:
        mode = self.config.left_stabilization_mode
        current = exec_step[0:7].clone()
        if previous_left_pose is None or mode == "none":
            return False, current
        if mode == "freeze":
            exec_step[0:7] = previous_left_pose
            return True, previous_left_pose
        if mode == "deadband":
            movement = torch.linalg.norm(current[0:3] - previous_left_pose[0:3])
            if float(movement.detach().cpu()) <= self.config.left_deadband_xyz:
                exec_step[0:7] = previous_left_pose
                return True, previous_left_pose
            return False, current
        alpha = self.config.left_lowpass_alpha
        blended = current.clone()
        blended[0:3] = alpha * current[0:3] + (1.0 - alpha) * previous_left_pose[0:3]
        exec_step[0:3] = blended[0:3]
        return True, blended

    def _max_delta_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = self.config.max_delta_local_xyz
        if isinstance(value, tuple):
            if len(value) != 3:
                raise ValueError("max_delta_local_xyz tuple must have length 3.")
            return torch.tensor(value, device=device, dtype=dtype)
        return torch.full((3,), float(value), device=device, dtype=dtype)

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.horizon_k <= 0:
            raise ValueError("horizon_k must be positive.")
        if cfg.target_horizon_offset < 0:
            raise ValueError("target_horizon_offset must be non-negative.")
        if cfg.residual_frame != "left_ee":
            raise NotImplementedError("Only residual_frame='left_ee' is supported.")
        if cfg.left_frame_source != "base_target":
            raise NotImplementedError("Only left_frame_source='base_target' is supported.")
        if cfg.left_stabilization_mode not in ("none", "deadband", "lowpass", "freeze"):
            raise ValueError("left_stabilization_mode must be none/deadband/lowpass/freeze.")
        if not 0.0 <= cfg.left_lowpass_alpha <= 1.0:
            raise ValueError("left_lowpass_alpha must be in [0, 1].")

    @staticmethod
    def _validate_qpos14_chunk(qpos14_chunk: torch.Tensor) -> None:
        if not isinstance(qpos14_chunk, torch.Tensor):
            raise TypeError("qpos14_chunk must be a torch.Tensor.")
        if qpos14_chunk.ndim != 3 or qpos14_chunk.shape[0] != 1 or qpos14_chunk.shape[-1] != 14:
            raise ValueError(f"qpos14_chunk must have shape [1, C, 14], got {tuple(qpos14_chunk.shape)}.")


def quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert one normalized-ish ``wxyz`` quaternion to a 3x3 rotation matrix."""
    if quat.shape != (4,):
        raise ValueError(f"quat must have shape [4], got {tuple(quat.shape)}.")
    quat = torch.nn.functional.normalize(quat.to(dtype=torch.float32), dim=0)
    w, x, y, z = quat.unbind()
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    ).to(device=quat.device)


def residual_ee_observation_to_tensor(obs: ResidualEEObservation) -> torch.Tensor:
    """Pack a residual EE observation into the default actor feature vector."""
    gate = torch.tensor(
        [0.0 if obs.gate_score is None else float(obs.gate_score)],
        dtype=torch.float32,
        device=obs.left_xyz.device,
    )
    step = torch.tensor(
        [float(obs.intervention_step_i)],
        dtype=torch.float32,
        device=obs.left_xyz.device,
    )
    return torch.cat(
        (
            obs.left_xyz.reshape(-1).to(dtype=torch.float32),
            obs.left_quat_wxyz.reshape(-1).to(dtype=torch.float32),
            obs.right_xyz.reshape(-1).to(dtype=torch.float32),
            obs.right_quat_wxyz.reshape(-1).to(dtype=torch.float32),
            obs.relative_xyz_left_frame.reshape(-1).to(dtype=torch.float32),
            obs.left_gripper.reshape(-1).to(dtype=torch.float32),
            obs.right_gripper.reshape(-1).to(dtype=torch.float32),
            gate,
            step,
        )
    )
