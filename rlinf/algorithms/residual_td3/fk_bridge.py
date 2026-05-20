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
"""ALOHA qpos14 -> canonical RoboTwin endpose16 FK bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import yaml

ALOHA_LEFT_JOINT_NAMES = (
    "fl_joint1",
    "fl_joint2",
    "fl_joint3",
    "fl_joint4",
    "fl_joint5",
    "fl_joint6",
)
ALOHA_RIGHT_JOINT_NAMES = (
    "fr_joint1",
    "fr_joint2",
    "fr_joint3",
    "fr_joint4",
    "fr_joint5",
    "fr_joint6",
)
ALOHA_ROOT_POSE_WXYZ = (0.0, -0.65, 0.0, 0.707, 0.0, 0.0, 0.707)


@dataclass(frozen=True)
class AlohaFKBridgeConfig:
    """Runtime configuration for CuRobo-backed ALOHA FK."""

    robotwin_path: str | None = None
    left_curobo_yml_path: str | None = None
    right_curobo_yml_path: str | None = None
    device: str = "cuda:0"
    left_root_pose_wxyz: tuple[float, float, float, float, float, float, float] = (
        ALOHA_ROOT_POSE_WXYZ
    )
    right_root_pose_wxyz: tuple[float, float, float, float, float, float, float] = (
        ALOHA_ROOT_POSE_WXYZ
    )


class ArmFKBackend(Protocol):
    """Backend contract for one arm's 6D qpos FK."""

    def qpos6_to_pose7(self, qpos6: torch.Tensor) -> torch.Tensor:
        """Return ``[B, 7]`` pose in ``[xyz, quat_wxyz]`` order."""


class AlohaFKBridge:
    """Convert ALOHA qpos14 action chunks to canonical RoboTwin endpose16.

    Canonical endpose16 layout is:
    ``[left xyz3, left quat_wxyz4, left gripper1,
    right xyz3, right quat_wxyz4, right gripper1]``.
    """

    def __init__(
        self,
        config: AlohaFKBridgeConfig | None = None,
        *,
        left_backend: ArmFKBackend | None = None,
        right_backend: ArmFKBackend | None = None,
    ) -> None:
        self.config = config or AlohaFKBridgeConfig()
        if (left_backend is None) != (right_backend is None):
            raise ValueError("left_backend and right_backend must be provided together.")
        if left_backend is None or right_backend is None:
            left_backend, right_backend = self._build_curobo_backends(self.config)
        self.left_backend = left_backend
        self.right_backend = right_backend

    @classmethod
    def from_robotwin_task(
        cls,
        task,
        *,
        device: str = "cuda:0",
    ) -> "AlohaFKBridge":
        """Build the bridge from a live RoboTwin ALOHA task."""
        robot = task.robot
        cfg = AlohaFKBridgeConfig(
            robotwin_path=os.environ.get("ROBOTWIN_PATH") or os.environ.get("ASSETS_PATH"),
            device=device,
            left_root_pose_wxyz=_pose_to_tuple(robot.left_entity_origion_pose),
            right_root_pose_wxyz=_pose_to_tuple(robot.right_entity_origion_pose),
        )
        return cls(cfg)

    def qpos14_to_endpose16(
        self,
        qpos14_chunk: torch.Tensor,
        current_obs=None,
    ) -> torch.Tensor:
        """Convert ``[B, C, 14]`` ALOHA qpos chunk to ``[B, C, 16]`` endpose."""
        del current_obs
        if not isinstance(qpos14_chunk, torch.Tensor):
            raise TypeError("qpos14_chunk must be a torch.Tensor.")
        if qpos14_chunk.ndim != 3 or qpos14_chunk.shape[-1] != 14:
            raise ValueError(f"qpos14_chunk must have shape [B, C, 14], got {qpos14_chunk.shape}.")
        batch_size, chunk_len, _ = qpos14_chunk.shape
        flat = qpos14_chunk.reshape(batch_size * chunk_len, 14)
        out = self.qpos14_step_to_endpose16(flat)
        return out.reshape(batch_size, chunk_len, 16)

    def qpos14_step_to_endpose16(self, qpos14: torch.Tensor) -> torch.Tensor:
        """Convert ``[B, 14]`` ALOHA qpos targets to ``[B, 16]`` endpose."""
        if not isinstance(qpos14, torch.Tensor):
            raise TypeError("qpos14 must be a torch.Tensor.")
        if qpos14.ndim != 2 or qpos14.shape[-1] != 14:
            raise ValueError(f"qpos14 must have shape [B, 14], got {qpos14.shape}.")

        left_pose = self.left_backend.qpos6_to_pose7(qpos14[..., 0:6])
        right_pose = self.right_backend.qpos6_to_pose7(qpos14[..., 7:13])
        left_pose = left_pose.to(device=qpos14.device, dtype=qpos14.dtype)
        right_pose = right_pose.to(device=qpos14.device, dtype=qpos14.dtype)
        return torch.cat(
            (
                left_pose[..., 0:7],
                qpos14[..., 6:7],
                right_pose[..., 0:7],
                qpos14[..., 13:14],
            ),
            dim=-1,
        )

    @staticmethod
    def _build_curobo_backends(
        config: AlohaFKBridgeConfig,
    ) -> tuple[ArmFKBackend, ArmFKBackend]:
        try:
            left_path, right_path = _resolve_curobo_paths(config)
            left = _CuRoboArmFK(
                yml_path=left_path,
                arm_joint_names=ALOHA_LEFT_JOINT_NAMES,
                root_pose_wxyz=config.left_root_pose_wxyz,
                device=config.device,
            )
            right = _CuRoboArmFK(
                yml_path=right_path,
                arm_joint_names=ALOHA_RIGHT_JOINT_NAMES,
                root_pose_wxyz=config.right_root_pose_wxyz,
                device=config.device,
            )
        except Exception as exc:  # noqa: BLE001
            msg = (
                "AlohaFKBridge requires a working CuRobo FK backend and ALOHA "
                "curobo_left.yml/curobo_right.yml configs. It will not fall back "
                "to a learned bridge silently."
            )
            raise RuntimeError(msg) from exc
        return left, right


class _CuRoboArmFK:
    """CuRobo FK backend with RoboTwin ALOHA frame conversion."""

    def __init__(
        self,
        *,
        yml_path: Path,
        arm_joint_names: tuple[str, ...],
        root_pose_wxyz: tuple[float, float, float, float, float, float, float],
        device: str,
    ) -> None:
        from curobo.cuda_robot_model.cuda_robot_model import (
            CudaRobotModel,
            CudaRobotModelConfig,
        )
        from curobo.types.base import TensorDeviceType

        tensor_args = TensorDeviceType(device=torch.device(device))
        cfg = CudaRobotModelConfig.from_robot_yaml_file(str(yml_path), tensor_args=tensor_args)
        self.model = CudaRobotModel(cfg)
        self.yml_path = yml_path
        self.arm_joint_names = arm_joint_names
        self.curobo_joint_names = tuple(self.model.kinematics_config.joint_names)
        self.device = torch.device(device)
        root = torch.tensor(root_pose_wxyz, dtype=torch.float32, device=self.device)
        self.root_position = root[0:3]
        self.root_quaternion = _normalize_quat(root[3:7])
        self.root_rotation = _quat_to_matrix(self.root_quaternion)
        self.frame_bias = torch.tensor(
            _read_frame_bias(yml_path),
            dtype=torch.float32,
            device=self.device,
        )

    def qpos6_to_pose7(self, qpos6: torch.Tensor) -> torch.Tensor:
        q = self._pack_qpos(qpos6)
        state = self.model.get_state(q)
        raw_pos = state.ee_position.to(dtype=q.dtype)
        raw_quat = _normalize_quat(state.ee_quaternion.to(dtype=q.dtype))
        world_pos = self.root_position.to(dtype=q.dtype) + (
            raw_pos - self.frame_bias.to(dtype=q.dtype)
        ) @ self.root_rotation.to(dtype=q.dtype).T
        world_quat = _quat_mul(
            self.root_quaternion.to(dtype=q.dtype).expand_as(raw_quat),
            raw_quat,
        )
        return torch.cat((world_pos, _normalize_quat(world_quat)), dim=-1)

    def _pack_qpos(self, qpos6: torch.Tensor) -> torch.Tensor:
        qpos6 = qpos6.to(device=self.device, dtype=torch.float32)
        values_by_name = {
            name: qpos6[..., index] for index, name in enumerate(self.arm_joint_names)
        }
        values = []
        for name in self.curobo_joint_names:
            if name in values_by_name:
                values.append(values_by_name[name])
            elif name.endswith(("joint7", "joint8")):
                values.append(torch.full_like(qpos6[..., 0], 0.04))
            else:
                values.append(torch.zeros_like(qpos6[..., 0]))
        return torch.stack(values, dim=-1)


def _resolve_curobo_paths(config: AlohaFKBridgeConfig) -> tuple[Path, Path]:
    robotwin_path = config.robotwin_path or os.environ.get("ROBOTWIN_PATH") or os.environ.get("ASSETS_PATH")
    if robotwin_path is None and (
        config.left_curobo_yml_path is None or config.right_curobo_yml_path is None
    ):
        raise FileNotFoundError("ROBOTWIN_PATH or explicit ALOHA CuRobo config paths are required.")

    base = Path(robotwin_path) if robotwin_path is not None else None
    left = Path(config.left_curobo_yml_path) if config.left_curobo_yml_path else (
        base / "assets" / "embodiments" / "aloha-agilex" / "curobo_left.yml"
    )
    right = Path(config.right_curobo_yml_path) if config.right_curobo_yml_path else (
        base / "assets" / "embodiments" / "aloha-agilex" / "curobo_right.yml"
    )
    if not left.exists():
        raise FileNotFoundError(f"left CuRobo config not found: {left}")
    if not right.exists():
        raise FileNotFoundError(f"right CuRobo config not found: {right}")
    return left, right


def _read_frame_bias(path: Path) -> tuple[float, float, float]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return tuple(data.get("planner", {}).get("frame_bias", (0.0, 0.0, 0.0)))


def _pose_to_tuple(pose) -> tuple[float, float, float, float, float, float, float]:
    return tuple(float(x) for x in (*pose.p, *pose.q))


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).eps)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quat(q)
    w, x, y, z = q.unbind(dim=-1)
    two = torch.tensor(2.0, dtype=q.dtype, device=q.device)
    return torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(3, 3)
