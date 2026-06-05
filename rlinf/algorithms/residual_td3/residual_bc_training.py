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
"""Offline BC utilities for residual right-xyz actors."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_NAMES = (
    "base_left_xyz",
    "base_left_quat_wxyz",
    "base_right_xyz",
    "base_right_quat_wxyz",
    "relative_right_xyz_in_left_frame",
    "left_gripper",
    "right_gripper",
    "gate_label",
    "gate_positive_fraction",
)
FEATURE_DIMS = {
    "base_left_xyz": 3,
    "base_left_quat_wxyz": 4,
    "base_right_xyz": 3,
    "base_right_quat_wxyz": 4,
    "relative_right_xyz_in_left_frame": 3,
    "left_gripper": 1,
    "right_gripper": 1,
    "gate_label": 1,
    "gate_positive_fraction": 1,
}


@dataclass(frozen=True)
class ResidualBCDatasetConfig:
    """Configuration for loading residual BC target npz files."""

    target_horizon_k: int | None = None
    max_delta_local_xyz: float = 0.05
    include_current_qpos14: bool = False


class ResidualBCNpzDataset(Dataset):
    """Torch dataset for residual BC targets saved by build_residual_bc_targets.py.

    Each item contains an ``obs_vector`` used by the MLP actor, the structured
    ``obs_features`` that produced it, ``target_delta_local_xyz`` shaped
    ``[K, 3]``, and episode/frame metadata.
    """

    def __init__(
        self,
        path: str | Path,
        cfg: ResidualBCDatasetConfig | None = None,
        indices: np.ndarray | list[int] | None = None,
    ) -> None:
        self.path = Path(path)
        self.cfg = cfg or ResidualBCDatasetConfig()
        if not self.path.exists():
            raise FileNotFoundError(f"Residual BC target npz not found: {self.path}")
        with np.load(self.path, allow_pickle=True) as data:
            self.arrays = {key: data[key] for key in data.files}
        self._validate_arrays()
        if indices is None:
            self.indices = np.arange(self.arrays["target_delta_local_xyz"].shape[0], dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
        self.obs_vectors = self._build_obs_vectors()

    @property
    def obs_dim(self) -> int:
        """Dimension of the flattened actor input vector."""

        return int(self.obs_vectors.shape[-1])

    @property
    def target_horizon_k(self) -> int:
        """Number of target residual steps exposed by the dataset."""

        return int(self.arrays["target_delta_local_xyz"].shape[1])

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        row = int(self.indices[index])
        obs_features = self._structured_obs_features(row)
        return {
            "obs_vector": torch.from_numpy(self.obs_vectors[row]).to(dtype=torch.float32),
            "obs_features": obs_features,
            "target_delta_local_xyz": torch.from_numpy(
                self.arrays["target_delta_local_xyz"][row]
            ).to(dtype=torch.float32),
            "episode_index": int(self.arrays["episode_index"][row]),
            "frame_index": int(self.arrays["frame_index"][row]),
        }

    def _validate_arrays(self) -> None:
        required = (
            "target_delta_local_xyz",
            "episode_index",
            "frame_index",
            "obs_base_left_xyz",
            "obs_base_left_quat_wxyz",
            "obs_base_right_xyz",
            "obs_base_right_quat_wxyz",
            "obs_relative_right_xyz_in_left_frame",
            "obs_left_gripper",
            "obs_right_gripper",
            "obs_gate_label",
            "gate_seq",
        )
        missing = [key for key in required if key not in self.arrays]
        if missing:
            raise KeyError(f"Residual BC target npz missing required arrays: {missing}")
        target = self.arrays["target_delta_local_xyz"]
        if target.ndim != 3 or target.shape[-1] != 3:
            raise ValueError(
                "target_delta_local_xyz must have shape [N, K, 3], "
                f"got {tuple(target.shape)}."
            )
        if self.cfg.target_horizon_k is not None:
            if self.cfg.target_horizon_k <= 0:
                raise ValueError("target_horizon_k must be positive.")
            if target.shape[1] < self.cfg.target_horizon_k:
                target = self._rebuild_targets_from_endpose(self.cfg.target_horizon_k)
            self.arrays["target_delta_local_xyz"] = target[:, : self.cfg.target_horizon_k, :]
        size = self.arrays["target_delta_local_xyz"].shape[0]
        for key, value in self.arrays.items():
            if value.shape and value.shape[0] != size:
                raise ValueError(f"Array {key} has mismatched first dimension {value.shape[0]} != {size}.")

    def _build_obs_vectors(self) -> np.ndarray:
        parts = [
            self._first_offset("obs_base_left_xyz"),
            self._first_offset("obs_base_left_quat_wxyz"),
            self._first_offset("obs_base_right_xyz"),
            self._first_offset("obs_base_right_quat_wxyz"),
            self._first_offset("obs_relative_right_xyz_in_left_frame"),
            self._first_offset("obs_left_gripper"),
            self._first_offset("obs_right_gripper"),
            self._first_offset("obs_gate_label"),
            (self.arrays["gate_seq"] > 0.0).mean(axis=1, keepdims=True).astype(np.float32),
        ]
        if self.cfg.include_current_qpos14:
            if "obs_current_qpos14_state" not in self.arrays:
                raise KeyError("include_current_qpos14=True requires obs_current_qpos14_state.")
            parts.append(np.asarray(self.arrays["obs_current_qpos14_state"], dtype=np.float32))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _rebuild_targets_from_endpose(self, target_horizon_k: int) -> np.ndarray:
        if "base_endpose16" not in self.arrays or "expert_endpose16" not in self.arrays:
            raise ValueError(
                "Requested target_horizon_k exceeds saved targets and base/expert endpose chunks "
                "are unavailable. Rebuild targets with --target-horizon-offsets covering the requested K."
            )
        base = np.asarray(self.arrays["base_endpose16"], dtype=np.float32)
        expert = np.asarray(self.arrays["expert_endpose16"], dtype=np.float32)
        if base.ndim != 3 or expert.ndim != 3 or base.shape != expert.shape or base.shape[-1] != 16:
            raise ValueError("base_endpose16 and expert_endpose16 must have matching shape [N, C, 16].")
        if base.shape[1] < target_horizon_k:
            raise ValueError(
                f"base/expert endpose chunks length {base.shape[1]} is smaller than target_horizon_k "
                f"{target_horizon_k}."
            )
        targets = []
        for offset in range(target_horizon_k):
            world = expert[:, offset, 8:11] - base[:, offset, 8:11]
            rotation = quat_wxyz_to_matrix_np(base[:, offset, 3:7])
            local = np.einsum("nij,nj->ni", np.swapaxes(rotation, 1, 2), world)
            targets.append(
                np.clip(
                    local,
                    -float(self.cfg.max_delta_local_xyz),
                    float(self.cfg.max_delta_local_xyz),
                )
            )
        return np.stack(targets, axis=1).astype(np.float32)

    def _first_offset(self, key: str) -> np.ndarray:
        value = np.asarray(self.arrays[key], dtype=np.float32)
        if value.ndim == 1:
            return value.reshape(-1, 1)
        if value.ndim == 2:
            if key in {"obs_left_gripper", "obs_right_gripper", "obs_gate_label", "obs_horizon_offset"}:
                return value[:, :1]
            return value[:, :1] if value.shape[1] == 1 else value
        return value[:, 0].reshape(value.shape[0], -1)

    def _structured_obs_features(self, row: int) -> dict[str, np.ndarray | float]:
        return {
            "base_left_xyz": self._first_offset_row("obs_base_left_xyz", row),
            "base_left_quat_wxyz": self._first_offset_row("obs_base_left_quat_wxyz", row),
            "base_right_xyz": self._first_offset_row("obs_base_right_xyz", row),
            "base_right_quat_wxyz": self._first_offset_row("obs_base_right_quat_wxyz", row),
            "relative_right_xyz_in_left_frame": self._first_offset_row(
                "obs_relative_right_xyz_in_left_frame", row
            ),
            "left_gripper": float(self._first_offset("obs_left_gripper")[row, 0]),
            "right_gripper": float(self._first_offset("obs_right_gripper")[row, 0]),
            "gate_label": float(self._first_offset("obs_gate_label")[row, 0]),
            "gate_positive_fraction": float((self.arrays["gate_seq"][row] > 0.0).mean()),
        }

    def _first_offset_row(self, key: str, row: int) -> np.ndarray:
        return self._first_offset(key)[row].astype(np.float32)


def split_by_episode(
    episode_index: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and validation row indices without episode leakage."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")
    episodes = np.unique(np.asarray(episode_index, dtype=np.int64))
    if episodes.size < 2:
        raise ValueError("Need at least two episodes for episode-level train/val split.")
    rng = np.random.default_rng(seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(episodes.size * val_ratio)))
    val_count = min(val_count, episodes.size - 1)
    val_episodes = {int(ep) for ep in shuffled[:val_count]}
    val_mask = np.asarray([int(ep) in val_episodes for ep in episode_index], dtype=bool)
    train_indices = np.where(~val_mask)[0].astype(np.int64)
    val_indices = np.where(val_mask)[0].astype(np.int64)
    if train_indices.size == 0 or val_indices.size == 0:
        raise ValueError("Episode split produced an empty train or validation set.")
    return train_indices, val_indices


def quat_wxyz_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions shaped [N, 4] to rotation matrices [N, 3, 3]."""

    q = np.asarray(quat, dtype=np.float32)
    if q.ndim != 2 or q.shape[-1] != 4:
        raise ValueError(f"quat must have shape [N, 4], got {tuple(q.shape)}.")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(norm, 1e-8, None)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrix = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - z * w)
    matrix[:, 0, 2] = 2.0 * (x * z + y * w)
    matrix[:, 1, 0] = 2.0 * (x * y + z * w)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - x * w)
    matrix[:, 2, 0] = 2.0 * (x * z - y * w)
    matrix[:, 2, 1] = 2.0 * (y * z + x * w)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def build_residual_actor_obs_features_from_endpose(
    base_endpose16: torch.Tensor,
    *,
    gate_label: float,
    gate_positive_fraction: float,
) -> dict[str, torch.Tensor]:
    """Build the canonical 21D residual actor observation from one EE16 target.

    ``base_endpose16`` uses ``[left xyz, left quat_wxyz, left gripper,
    right xyz, right quat_wxyz, right gripper]``. The resulting feature order
    exactly matches ``FEATURE_NAMES`` and ``ResidualBCNpzDataset``.
    """

    if base_endpose16.shape != (16,):
        raise ValueError(f"base_endpose16 must have shape [16], got {tuple(base_endpose16.shape)}.")
    return build_residual_actor_obs_features(
        base_left_xyz=base_endpose16[0:3],
        base_left_quat_wxyz=base_endpose16[3:7],
        base_right_xyz=base_endpose16[8:11],
        base_right_quat_wxyz=base_endpose16[11:15],
        left_gripper=base_endpose16[7:8],
        right_gripper=base_endpose16[15:16],
        gate_label=gate_label,
        gate_positive_fraction=gate_positive_fraction,
    )


def build_residual_actor_obs_features(
    *,
    base_left_xyz: torch.Tensor,
    base_left_quat_wxyz: torch.Tensor,
    base_right_xyz: torch.Tensor,
    base_right_quat_wxyz: torch.Tensor,
    left_gripper: torch.Tensor | float,
    right_gripper: torch.Tensor | float,
    gate_label: float,
    gate_positive_fraction: float,
) -> dict[str, torch.Tensor]:
    """Build structured residual actor features in the canonical 21D order."""

    left_xyz = torch.as_tensor(base_left_xyz, dtype=torch.float32).reshape(-1)
    left_quat = torch.as_tensor(base_left_quat_wxyz, dtype=torch.float32).reshape(-1)
    right_xyz = torch.as_tensor(base_right_xyz, dtype=torch.float32).reshape(-1)
    right_quat = torch.as_tensor(base_right_quat_wxyz, dtype=torch.float32).reshape(-1)
    left_gripper_tensor = torch.as_tensor(left_gripper, dtype=torch.float32).reshape(-1)
    right_gripper_tensor = torch.as_tensor(right_gripper, dtype=torch.float32).reshape(-1)
    if left_xyz.numel() != 3:
        raise ValueError(f"base_left_xyz must have 3 values, got {left_xyz.numel()}.")
    if left_quat.numel() != 4:
        raise ValueError(f"base_left_quat_wxyz must have 4 values, got {left_quat.numel()}.")
    if right_xyz.numel() != 3:
        raise ValueError(f"base_right_xyz must have 3 values, got {right_xyz.numel()}.")
    if right_quat.numel() != 4:
        raise ValueError(f"base_right_quat_wxyz must have 4 values, got {right_quat.numel()}.")
    if left_gripper_tensor.numel() != 1:
        raise ValueError(f"left_gripper must have 1 value, got {left_gripper_tensor.numel()}.")
    if right_gripper_tensor.numel() != 1:
        raise ValueError(f"right_gripper must have 1 value, got {right_gripper_tensor.numel()}.")
    device = left_xyz.device
    rotation = quat_wxyz_to_matrix_torch(left_quat).to(device=device)
    relative_left = rotation.T @ (right_xyz.to(device=device) - left_xyz)
    return {
        "base_left_xyz": left_xyz,
        "base_left_quat_wxyz": left_quat.to(device=device),
        "base_right_xyz": right_xyz.to(device=device),
        "base_right_quat_wxyz": right_quat.to(device=device),
        "relative_right_xyz_in_left_frame": relative_left,
        "left_gripper": left_gripper_tensor.to(device=device),
        "right_gripper": right_gripper_tensor.to(device=device),
        "gate_label": torch.tensor([float(gate_label)], dtype=torch.float32, device=device),
        "gate_positive_fraction": torch.tensor(
            [float(gate_positive_fraction)],
            dtype=torch.float32,
            device=device,
        ),
    }


def residual_actor_obs_features_to_tensor(
    features: dict[str, torch.Tensor | np.ndarray | float],
) -> torch.Tensor:
    """Pack structured residual actor features into the canonical 21D tensor."""

    parts = []
    for name in FEATURE_NAMES:
        value = features[name]
        tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        expected = FEATURE_DIMS[name]
        if tensor.numel() != expected:
            raise ValueError(f"{name} must have {expected} values, got {tensor.numel()}.")
        parts.append(tensor)
    return torch.cat(parts, dim=0)


def quat_wxyz_to_matrix_torch(quat: torch.Tensor) -> torch.Tensor:
    """Convert one ``wxyz`` quaternion to a 3x3 rotation matrix."""

    if quat.shape != (4,):
        raise ValueError(f"quat must have shape [4], got {tuple(quat.shape)}.")
    q = torch.nn.functional.normalize(quat.to(dtype=torch.float32), dim=0)
    w, x, y, z = q.unbind()
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    ).to(device=quat.device)


def iter_numpy_batches(
    dataset: ResidualBCNpzDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield obs and target tensors from a dataset without multiprocessing overhead."""

    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, order.shape[0], batch_size):
        batch = order[start : start + batch_size]
        obs = torch.from_numpy(dataset.obs_vectors[batch]).to(dtype=torch.float32)
        target = torch.from_numpy(dataset.arrays["target_delta_local_xyz"][batch]).to(dtype=torch.float32)
        yield obs, target
