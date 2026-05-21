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
"""Binary residual gate pseudo-labeling utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class BinaryGateLabelerConfig:
    """Configuration for binary residual gate labeling."""

    moving_eps: float = 1e-3
    distance_small_threshold: float | None = None
    distance_stable_eps: float = 1e-4
    gripper_closed_threshold: float | None = None
    gripper_closed_is_high: bool = True
    contact_dilation: int = 3
    min_positive_segment_len: int = 3


class BinaryResidualGateLabeler:
    """Builds binary residual gate labels from end-effector traces."""

    def __init__(self, cfg: BinaryGateLabelerConfig):
        self.cfg = cfg

    def label_episode(
        self,
        *,
        left_ee_pos: np.ndarray,
        right_ee_pos: np.ndarray,
        left_gripper: np.ndarray,
        right_gripper: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Labels one episode.

        Args:
            left_ee_pos: Left end-effector positions with shape ``[T, 3]``.
            right_ee_pos: Right end-effector positions with shape ``[T, 3]``.
            left_gripper: Left gripper scalar trace with shape ``[T]``.
            right_gripper: Right gripper scalar trace with shape ``[T]``.
            timestamps: Optional timestamps with shape ``[T]``. Currently used
                only for validation and passthrough.

        Returns:
            A dictionary containing ``w_binary`` and intermediate debug signals.
        """
        left_pos = _as_float_array(left_ee_pos, "left_ee_pos", ndim=2)
        right_pos = _as_float_array(right_ee_pos, "right_ee_pos", ndim=2)
        left_grip = _as_float_array(left_gripper, "left_gripper", ndim=1)
        right_grip = _as_float_array(right_gripper, "right_gripper", ndim=1)
        _validate_positions(left_pos, right_pos)
        _validate_lengths(left_pos.shape[0], left_grip, right_grip, timestamps)

        v_left = _framewise_norm_delta(left_pos)
        v_right = _framewise_norm_delta(right_pos)
        distance_lr = np.linalg.norm(left_pos - right_pos, axis=-1)
        delta_distance = np.zeros_like(distance_lr)
        delta_distance[1:] = distance_lr[1:] - distance_lr[:-1]

        moving_left = v_left > self.cfg.moving_eps
        moving_right = v_right > self.cfg.moving_eps
        both_moving = moving_left & moving_right

        left_threshold = _resolve_gripper_threshold(
            left_grip,
            self.cfg.gripper_closed_threshold,
        )
        right_threshold = _resolve_gripper_threshold(
            right_grip,
            self.cfg.gripper_closed_threshold,
        )
        if self.cfg.gripper_closed_is_high:
            left_closed = left_grip > left_threshold
            right_closed = right_grip > right_threshold
        else:
            left_closed = left_grip < left_threshold
            right_closed = right_grip < right_threshold
        both_closed = left_closed & right_closed

        distance_threshold = _resolve_distance_threshold(
            distance_lr,
            self.cfg.distance_small_threshold,
        )
        relative_stable = np.abs(delta_distance) < self.cfg.distance_stable_eps
        close_and_moving = both_moving & (distance_lr < distance_threshold)
        stable_close_and_moving = close_and_moving & relative_stable
        w_raw = both_closed | close_and_moving | stable_close_and_moving
        w_binary = _remove_short_positive_segments(
            _dilate_binary(w_raw, self.cfg.contact_dilation),
            self.cfg.min_positive_segment_len,
        )

        thresholds = {
            "distance_small_threshold": float(distance_threshold),
            "left_gripper_closed_threshold": float(left_threshold),
            "right_gripper_closed_threshold": float(right_threshold),
            "gripper_closed_is_high": bool(self.cfg.gripper_closed_is_high),
        }
        return {
            "w_binary": w_binary.astype(np.int64),
            "w_raw": w_raw.astype(np.int64),
            "v_L": v_left,
            "v_R": v_right,
            "d_LR": distance_lr,
            "delta_d": delta_distance,
            "moving_L": moving_left,
            "moving_R": moving_right,
            "both_moving": both_moving,
            "left_closed": left_closed,
            "right_closed": right_closed,
            "both_closed": both_closed,
            "relative_stable": relative_stable,
            "thresholds": thresholds,
            "config": asdict(self.cfg),
        }


def _as_float_array(value: np.ndarray, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim {ndim}, got shape {array.shape}.")
    return array


def _validate_positions(left_pos: np.ndarray, right_pos: np.ndarray) -> None:
    if left_pos.shape != right_pos.shape:
        raise ValueError(
            "left_ee_pos and right_ee_pos must have the same shape, "
            f"got {left_pos.shape} and {right_pos.shape}."
        )
    if left_pos.shape[-1] != 3:
        raise ValueError(f"end-effector positions must have shape [T, 3], got {left_pos.shape}.")


def _validate_lengths(
    length: int,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    timestamps: np.ndarray | None,
) -> None:
    if left_gripper.shape[0] != length or right_gripper.shape[0] != length:
        raise ValueError("gripper traces must have the same length as end-effector positions.")
    if timestamps is not None and np.asarray(timestamps).shape[0] != length:
        raise ValueError("timestamps must have the same length as end-effector positions.")


def _framewise_norm_delta(positions: np.ndarray) -> np.ndarray:
    velocity = np.zeros(positions.shape[0], dtype=np.float64)
    if positions.shape[0] > 1:
        velocity[1:] = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    return velocity


def _resolve_gripper_threshold(values: np.ndarray, configured: float | None) -> float:
    if configured is not None:
        return float(configured)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot estimate gripper threshold from non-finite values.")
    return float((np.min(finite) + np.max(finite)) / 2.0)


def _resolve_distance_threshold(values: np.ndarray, configured: float | None) -> float:
    if configured is not None:
        return float(configured)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot estimate distance threshold from non-finite values.")
    return float(np.quantile(finite, 0.3))


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError(f"contact_dilation must be >= 0, got {radius}.")
    out = np.asarray(mask, dtype=bool).copy()
    if radius == 0 or out.size == 0:
        return out
    positive = np.flatnonzero(out)
    for index in positive:
        start = max(0, index - radius)
        stop = min(out.size, index + radius + 1)
        out[start:stop] = True
    return out


def _remove_short_positive_segments(mask: np.ndarray, min_len: int) -> np.ndarray:
    if min_len < 1:
        raise ValueError(f"min_positive_segment_len must be >= 1, got {min_len}.")
    out = np.asarray(mask, dtype=bool).copy()
    start = None
    for index, value in enumerate(np.r_[out, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start < min_len:
                out[start:index] = False
            start = None
    return out
