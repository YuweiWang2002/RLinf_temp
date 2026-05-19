"""Alignment reward for RoboTwin handover residual control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


ERROR_SHAPE = (4,)


@dataclass(frozen=True)
class HandoverAlignmentRewardConfig:
    """Configuration for handover alignment reward shaping."""

    xy_weight: float = 1.0
    z_weight: float = 1.0
    yaw_weight: float = 0.25
    residual_penalty_weight: float = 0.01
    residual_delta_penalty_weight: float = 0.0
    success_bonus: float = 5.0
    collision_penalty: float = 2.0
    timeout_penalty: float = 0.0
    stuck_penalty: float = 0.0
    success_xy_threshold: float = 0.02
    success_z_threshold: float = 0.02
    success_yaw_threshold: float = 0.087
    reward_clip: tuple[float, float] | None = None


def compute_handover_alignment_reward(
    prev_error: np.ndarray,
    curr_error: np.ndarray,
    residual: np.ndarray,
    prev_residual: np.ndarray | None = None,
    info: Mapping[str, Any] | None = None,
    config: HandoverAlignmentRewardConfig | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute reward from current and previous alignment errors.

    Args:
        prev_error: Previous step error as ``[ex, ey, ez, eyaw]``.
        curr_error: Current step error as ``[ex, ey, ez, eyaw]``.
        residual: Current residual action as ``[dx, dy, dz, dyaw]``.
        prev_residual: Optional previous residual action.
        info: Optional current-step metadata.
        config: Reward shaping configuration.

    Returns:
        A scalar reward and normalized reward diagnostics.

    Raises:
        ValueError: If any error or residual input is not shaped ``(4,)``.
    """
    cfg = config or HandoverAlignmentRewardConfig()
    metadata = info or {}

    prev_error_arr = _as_vector4(prev_error, "prev_error")
    curr_error_arr = _as_vector4(curr_error, "curr_error")
    residual_arr = _as_vector4(residual, "residual")
    prev_residual_arr = (
        _as_vector4(prev_residual, "prev_residual")
        if prev_residual is not None
        else None
    )

    prev_components = _error_components(prev_error_arr)
    curr_components = _error_components(curr_error_arr)
    progress_xy = prev_components["e_xy"] - curr_components["e_xy"]
    progress_z = prev_components["e_z"] - curr_components["e_z"]
    progress_yaw = prev_components["e_yaw"] - curr_components["e_yaw"]

    residual_norm = float(np.linalg.norm(residual_arr))
    residual_delta_norm = (
        float(np.linalg.norm(residual_arr - prev_residual_arr))
        if prev_residual_arr is not None
        else 0.0
    )

    success = _standard_bool(
        metadata,
        keys=("success", "is_success", "succeeded"),
        default=_within_success_threshold(curr_components, cfg),
    )
    collision = _standard_bool(
        metadata,
        keys=("collision", "has_collision", "collided"),
        default=False,
    )
    timeout = _standard_bool(
        metadata,
        keys=("timeout", "time_out", "timed_out", "truncated"),
        default=False,
    )
    stuck = _standard_bool(
        metadata,
        keys=("stuck", "is_stuck"),
        default=False,
    )

    reward = (
        cfg.xy_weight * progress_xy
        + cfg.z_weight * progress_z
        + cfg.yaw_weight * progress_yaw
        - cfg.residual_penalty_weight * residual_norm**2
        - cfg.residual_delta_penalty_weight * residual_delta_norm**2
    )
    if success:
        reward += cfg.success_bonus
    if collision:
        reward -= cfg.collision_penalty
    if timeout:
        reward -= cfg.timeout_penalty
    if stuck:
        reward -= cfg.stuck_penalty

    reward_before_clip = float(reward)
    if cfg.reward_clip is not None:
        reward = float(np.clip(reward, cfg.reward_clip[0], cfg.reward_clip[1]))
    else:
        reward = float(reward)

    reward_info = {
        "e_xy": curr_components["e_xy"],
        "e_z": curr_components["e_z"],
        "e_yaw": curr_components["e_yaw"],
        "prev_e_xy": prev_components["e_xy"],
        "prev_e_z": prev_components["e_z"],
        "prev_e_yaw": prev_components["e_yaw"],
        "progress_xy": float(progress_xy),
        "progress_z": float(progress_z),
        "progress_yaw": float(progress_yaw),
        "residual_norm": residual_norm,
        "residual_delta_norm": residual_delta_norm,
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "stuck": stuck,
        "reward_before_clip": reward_before_clip,
    }
    return reward, reward_info


class HandoverAlignmentReward:
    """Callable wrapper for handover alignment reward computation."""

    def __init__(
        self,
        config: HandoverAlignmentRewardConfig | None = None,
    ) -> None:
        self.config = config or HandoverAlignmentRewardConfig()

    def __call__(
        self,
        prev_error: np.ndarray,
        curr_error: np.ndarray,
        residual: np.ndarray,
        prev_residual: np.ndarray | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Compute a scalar reward and diagnostics."""
        return compute_handover_alignment_reward(
            prev_error=prev_error,
            curr_error=curr_error,
            residual=residual,
            prev_residual=prev_residual,
            info=info,
            config=self.config,
        )


def _as_vector4(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != ERROR_SHAPE:
        msg = f"{name} must have shape {ERROR_SHAPE}, got {arr.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return arr


def _error_components(error: np.ndarray) -> dict[str, float]:
    return {
        "e_xy": float(np.linalg.norm(error[:2])),
        "e_z": float(abs(error[2])),
        "e_yaw": float(abs(error[3])),
    }


def _within_success_threshold(
    components: Mapping[str, float],
    config: HandoverAlignmentRewardConfig,
) -> bool:
    return bool(
        components["e_xy"] <= config.success_xy_threshold
        and components["e_z"] <= config.success_z_threshold
        and components["e_yaw"] <= config.success_yaw_threshold
    )


def _standard_bool(
    info: Mapping[str, Any],
    keys: tuple[str, ...],
    default: bool,
) -> bool:
    for key in keys:
        if key in info:
            return bool(info[key])
    return bool(default)
