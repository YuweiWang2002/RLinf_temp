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
from pathlib import Path
from typing import Protocol

import torch

from .residual_actor import ResidualActorConfig, ZeroInitResidualActorMLP
from .residual_bc_training import (
    build_residual_actor_obs_features,
    build_residual_actor_obs_features_from_endpose,
    residual_actor_obs_features_to_tensor,
)

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
    residual_scale: float = 1.0
    dry_run: bool = False
    learned_residual_control_enabled: bool = False


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
    gate_label: float | None = None
    gate_positive_fraction: float | None = None


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


class RandomNoiseResidualActor:
    """Actor that samples clipped Gaussian local xyz residuals."""

    def __init__(
        self,
        noise_std: float = 0.001,
        max_delta_local_xyz: float | tuple[float, float, float] = 0.05,
        seed: int | None = None,
    ) -> None:
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        self.noise_std = float(noise_std)
        self.max_delta_local_xyz = max_delta_local_xyz
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))

    def predict_delta_local_xyz(self, obs: ResidualEEObservation) -> torch.Tensor:
        del obs
        delta = torch.randn(3, generator=self.generator, dtype=torch.float32) * self.noise_std
        return torch.clamp(delta, min=-self._max_delta_tensor(), max=self._max_delta_tensor())

    def _max_delta_tensor(self) -> torch.Tensor:
        if isinstance(self.max_delta_local_xyz, tuple):
            if len(self.max_delta_local_xyz) != 3:
                raise ValueError("max_delta_local_xyz tuple must have length 3.")
            return torch.tensor(self.max_delta_local_xyz, dtype=torch.float32)
        return torch.full((3,), float(self.max_delta_local_xyz), dtype=torch.float32)


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


class BCResidualActor:
    """Saved BC residual actor that predicts a full local-xyz delta chunk."""

    def __init__(
        self,
        model: ZeroInitResidualActorMLP,
        *,
        device: str | torch.device = "cpu",
        checkpoint_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.checkpoint_path = checkpoint_path

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "BCResidualActor":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = ZeroInitResidualActorMLP(ResidualActorConfig(**checkpoint["actor_config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, device=device, checkpoint_path=str(checkpoint_path))

    @property
    def chunk_len(self) -> int:
        return int(self.model.cfg.chunk_len)

    def predict_delta_chunk_local_xyz(self, obs_vector: torch.Tensor) -> torch.Tensor:
        """Return ``[K, 3]`` predicted local residuals for one intervention."""

        if obs_vector.ndim != 1 or obs_vector.shape[0] != self.model.cfg.obs_dim:
            raise ValueError(
                f"obs_vector must have shape [{self.model.cfg.obs_dim}], got {tuple(obs_vector.shape)}."
            )
        with torch.no_grad():
            return self.model(obs_vector.reshape(1, -1).to(self.device))[0].detach()

    def predict_delta_local_xyz(self, obs: ResidualEEObservation) -> torch.Tensor:
        return self.predict_delta_chunk_local_xyz(residual_ee_observation_to_tensor(obs))[0]


class TD3ResidualActor(BCResidualActor):
    """Offline TD3 residual actor with the same inference contract as BC."""


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
        intervention_stage: str = "handover",
        trigger_source: str = "handover_gate",
        trigger_metadata: dict[str, object] | None = None,
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
        gate_label = float(gate_score is not None and gate_threshold is not None and gate_score >= gate_threshold)
        gate_positive_fraction = resolve_gate_positive_fraction_online(gate_score, gate_label)
        obs_features = build_residual_actor_obs_features_from_endpose(
            base_ee16[0, 0],
            gate_label=gate_label,
            gate_positive_fraction=gate_positive_fraction,
        )
        obs_vector = residual_actor_obs_features_to_tensor(obs_features)
        pred_delta_chunk = self._predict_delta_chunk_if_supported(obs_vector, base_ee16.shape[1])
        if pred_delta_chunk is not None:
            pred_delta_chunk = pred_delta_chunk.to(device=qpos14_chunk.device, dtype=qpos14_chunk.dtype)
        records: list[dict[str, object]] = []
        applied_delta_norms = []
        pred_delta_norms = []
        saturation_count = 0
        nan_inf_count = 0
        stabilization_count = 0
        previous_left_pose: torch.Tensor | None = None
        max_delta = self._max_delta_tensor(qpos14_chunk.device, qpos14_chunk.dtype)
        trigger_metadata = dict(trigger_metadata or {})

        for local_i, chunk_index in enumerate(selected_indices):
            base_step = base_ee16[0, local_i]
            if pred_delta_chunk is None:
                residual_obs = self._build_observation(
                    base_step,
                    gate_score,
                    local_i,
                    gate_label=gate_label,
                    gate_positive_fraction=gate_positive_fraction,
                )
                pred_delta_local = self.actor.predict_delta_local_xyz(residual_obs).to(
                    device=qpos14_chunk.device,
                    dtype=qpos14_chunk.dtype,
                )
            else:
                pred_delta_local = pred_delta_chunk[local_i]
            if pred_delta_local.shape != (3,):
                raise ValueError(
                    f"pred_delta_local_xyz must have shape [3], got {tuple(pred_delta_local.shape)}."
                )
            has_nan_or_inf = not bool(torch.isfinite(pred_delta_local).all().detach().cpu())
            clipped_delta_local = torch.clamp(pred_delta_local, min=-max_delta, max=max_delta)
            saturated = bool(torch.ne(pred_delta_local, clipped_delta_local).any().detach().cpu())
            exec_before_residual = exec_ee16[0, local_i].clone()
            if self.config.dry_run:
                applied_delta_local = torch.zeros_like(clipped_delta_local)
            elif self.config.learned_residual_control_enabled:
                applied_delta_local = clipped_delta_local * float(self.config.residual_scale)
            else:
                applied_delta_local = torch.zeros_like(clipped_delta_local)
            pred_delta_world = quat_wxyz_to_matrix(base_step[ENDPOSE16_LEFT_QUAT_WXYZ]) @ pred_delta_local
            applied_delta_world = quat_wxyz_to_matrix(base_step[ENDPOSE16_LEFT_QUAT_WXYZ]) @ applied_delta_local
            if self.config.learned_residual_control_enabled and not self.config.dry_run:
                exec_ee16[0, local_i, ENDPOSE16_RIGHT_XYZ] += applied_delta_world
            stabilized, previous_left_pose = self._stabilize_left_pose(
                exec_ee16[0, local_i],
                previous_left_pose,
            )
            stabilization_count += int(stabilized)
            pred_delta_norm = float(torch.linalg.norm(pred_delta_local).detach().cpu())
            applied_delta_norm = float(torch.linalg.norm(applied_delta_local).detach().cpu())
            pred_delta_norms.append(pred_delta_norm)
            applied_delta_norms.append(applied_delta_norm)
            saturation_count += int(saturated)
            nan_inf_count += int(has_nan_or_inf)
            obs_features_log = {key: value.detach().cpu().tolist() for key, value in obs_features.items()}
            records.append(
                {
                    "episode_id": episode_id,
                    "env_step": env_step,
                    "intervention_id": intervention_id,
                    "intervention_stage": intervention_stage,
                    "trigger_source": trigger_source,
                    "intervention_step_i": local_i,
                    "gate_score": gate_score,
                    "gate_threshold": gate_threshold,
                    "action_chunk_index": int(chunk_index),
                    "gate_label_online": gate_label,
                    "gate_positive_fraction_online": gate_positive_fraction,
                    "gate_positive_fraction_source": "gate_score" if gate_score is not None else "gate_label",
                    "base_qpos14": selected_qpos[0, local_i].detach().cpu().tolist(),
                    "base_ee16": base_step.detach().cpu().tolist(),
                    "obs_features": obs_features_log,
                    "obs_vector": obs_vector.detach().cpu().tolist(),
                    "obs_feature_reference_step": 0,
                    "pred_delta_local_xyz": pred_delta_local.detach().cpu().tolist(),
                    "pred_delta_world_xyz": pred_delta_world.detach().cpu().tolist(),
                    "pred_delta_norm": pred_delta_norm,
                    "residual_scale": float(self.config.residual_scale),
                    "noise_std": float(getattr(self.actor, "noise_std", 0.0)),
                    "dry_run": bool(self.config.dry_run),
                    "learned_residual_control_enabled": bool(self.config.learned_residual_control_enabled),
                    "applied_delta_local_xyz": applied_delta_local.detach().cpu().tolist(),
                    "applied_delta_world_xyz": applied_delta_world.detach().cpu().tolist(),
                    "applied_delta_norm": applied_delta_norm,
                    "exec_ee16_before_residual": exec_before_residual.detach().cpu().tolist(),
                    "exec_ee16_after_residual": exec_ee16[0, local_i].detach().cpu().tolist(),
                    "exec_ee16": exec_ee16[0, local_i].detach().cpu().tolist(),
                    "delta_local_xyz": applied_delta_local.detach().cpu().tolist(),
                    "delta_world_xyz": applied_delta_world.detach().cpu().tolist(),
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
                    "saturation": int(saturated),
                    "has_nan_or_inf": int(has_nan_or_inf),
                }
            )

        metadata = {
            "num_steps_executed": int(len(selected_indices)),
            "selected_indices": [int(index) for index in selected_indices],
            "max_delta_norm": float(max(applied_delta_norms, default=0.0)),
            "mean_delta_norm": float(sum(applied_delta_norms) / len(applied_delta_norms))
            if applied_delta_norms
            else 0.0,
            "pred_delta_norm_max": float(max(pred_delta_norms, default=0.0)),
            "pred_delta_norm_mean": float(sum(pred_delta_norms) / len(pred_delta_norms))
            if pred_delta_norms
            else 0.0,
            "applied_delta_norm_max": float(max(applied_delta_norms, default=0.0)),
            "applied_delta_norm_mean": float(sum(applied_delta_norms) / len(applied_delta_norms))
            if applied_delta_norms
            else 0.0,
            "saturation_count": int(saturation_count),
            "nan_inf_count": int(nan_inf_count),
            "dry_run": bool(self.config.dry_run),
            "learned_residual_control_enabled": bool(self.config.learned_residual_control_enabled),
            "residual_scale": float(self.config.residual_scale),
            "left_stabilization_count": int(stabilization_count),
            "gate_label_online": gate_label,
            "gate_positive_fraction_online": gate_positive_fraction,
            "gate_positive_fraction_source": "gate_score" if gate_score is not None else "gate_label",
            "obs_feature_reference_step": 0,
            "intervention_stage": intervention_stage,
            "trigger_source": trigger_source,
            **trigger_metadata,
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

    def _predict_delta_chunk_if_supported(
        self,
        obs_vector: torch.Tensor,
        chunk_len: int,
    ) -> torch.Tensor | None:
        if not hasattr(self.actor, "predict_delta_chunk_local_xyz"):
            return None
        pred = self.actor.predict_delta_chunk_local_xyz(obs_vector)
        if pred.ndim != 2 or pred.shape[-1] != 3:
            raise ValueError(f"predicted residual chunk must have shape [K, 3], got {tuple(pred.shape)}.")
        if pred.shape[0] < chunk_len:
            raise ValueError(f"predicted residual chunk length {pred.shape[0]} is smaller than {chunk_len}.")
        return pred[:chunk_len].to(dtype=torch.float32)

    def _build_observation(
        self,
        base_step: torch.Tensor,
        gate_score: float | None,
        intervention_step_i: int,
        *,
        gate_label: float | None = None,
        gate_positive_fraction: float | None = None,
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
            gate_label=gate_label,
            gate_positive_fraction=gate_positive_fraction,
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
        if cfg.residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative.")
        if cfg.dry_run and cfg.learned_residual_control_enabled:
            raise ValueError("dry_run and learned_residual_control_enabled cannot both be true.")

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
    features = build_residual_actor_obs_features(
        base_left_xyz=obs.left_xyz,
        base_left_quat_wxyz=obs.left_quat_wxyz,
        base_right_xyz=obs.right_xyz,
        base_right_quat_wxyz=obs.right_quat_wxyz,
        left_gripper=obs.left_gripper,
        right_gripper=obs.right_gripper,
        gate_label=float(obs.gate_label) if obs.gate_label is not None else 0.0,
        gate_positive_fraction=(
            float(obs.gate_positive_fraction)
            if obs.gate_positive_fraction is not None
            else resolve_gate_positive_fraction_online(obs.gate_score, obs.gate_label)
        ),
    )
    expected_relative = torch.as_tensor(obs.relative_xyz_left_frame, dtype=torch.float32).reshape(-1)
    actual_relative = features["relative_right_xyz_in_left_frame"]
    if not torch.allclose(actual_relative, expected_relative.to(device=actual_relative.device), atol=1e-6):
        raise ValueError("ResidualEEObservation relative_xyz_left_frame is inconsistent with pose/quaternion inputs.")
    return residual_actor_obs_features_to_tensor(features)


def resolve_gate_positive_fraction_online(
    gate_score: float | None,
    gate_label: float | None,
) -> float:
    """Resolve online gate_positive_fraction when no future gate sequence exists."""

    if gate_score is not None:
        return float(gate_score)
    return 0.0 if gate_label is None else float(gate_label)
