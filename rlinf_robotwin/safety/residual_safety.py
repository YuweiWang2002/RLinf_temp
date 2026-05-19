"""Residual action safety filter for RoboTwin handover control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


RESIDUAL_SHAPE = (4,)
DEFAULT_RESIDUAL_LOW = (-0.01, -0.01, -0.01, -0.087)
DEFAULT_RESIDUAL_HIGH = (0.01, 0.01, 0.01, 0.087)


@dataclass(frozen=True)
class ResidualSafetyConfig:
    """Configuration for residual action safety constraints."""

    per_dim_low: Sequence[float] = field(default_factory=lambda: DEFAULT_RESIDUAL_LOW)
    per_dim_high: Sequence[float] = field(default_factory=lambda: DEFAULT_RESIDUAL_HIGH)
    max_norm: float | None = 0.1
    workspace_low: Sequence[float] | None = None
    workspace_high: Sequence[float] | None = None


def apply_residual_safety_filter(
    residual: np.ndarray,
    config: ResidualSafetyConfig | None = None,
    current_position: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Clip a residual action to configured safety limits.

    Args:
        residual: Residual action shaped ``(4,)``.
        config: Safety configuration.
        current_position: Optional current pose shaped ``(4,)`` for workspace bounds.

    Returns:
        The clipped residual and diagnostics.

    Raises:
        ValueError: If vector inputs are not shaped ``(4,)`` or contain non-finite
            values.
    """
    cfg = config or ResidualSafetyConfig()
    residual_arr = _as_vector4(residual, "residual")
    low = _as_vector4(np.asarray(cfg.per_dim_low), "per_dim_low")
    high = _as_vector4(np.asarray(cfg.per_dim_high), "per_dim_high")
    if np.any(low > high):
        raise ValueError("per_dim_low must be less than or equal to per_dim_high")

    clipped = np.clip(residual_arr, low, high)
    per_dim_clipped = bool(not np.allclose(clipped, residual_arr))

    norm_clipped = False
    if cfg.max_norm is not None:
        if cfg.max_norm < 0:
            raise ValueError("max_norm must be non-negative")
        norm = float(np.linalg.norm(clipped))
        if norm > cfg.max_norm and norm > 0.0:
            clipped = clipped * (cfg.max_norm / norm)
            norm_clipped = True

    workspace_clipped = False
    if cfg.workspace_low is not None or cfg.workspace_high is not None:
        if current_position is None:
            raise ValueError("current_position is required for workspace bounds")
        if cfg.workspace_low is None or cfg.workspace_high is None:
            raise ValueError("workspace_low and workspace_high must be set together")
        position = _as_vector4(current_position, "current_position")
        workspace_low = _as_vector4(np.asarray(cfg.workspace_low), "workspace_low")
        workspace_high = _as_vector4(np.asarray(cfg.workspace_high), "workspace_high")
        if np.any(workspace_low > workspace_high):
            raise ValueError(
                "workspace_low must be less than or equal to workspace_high"
            )
        target = np.clip(position + clipped, workspace_low, workspace_high)
        workspace_adjusted = target - position
        workspace_clipped = bool(not np.allclose(workspace_adjusted, clipped))
        clipped = workspace_adjusted

    safety_info = {
        "input_norm": float(np.linalg.norm(residual_arr)),
        "output_norm": float(np.linalg.norm(clipped)),
        "per_dim_clipped": per_dim_clipped,
        "norm_clipped": norm_clipped,
        "workspace_clipped": workspace_clipped,
        "clip_delta_norm": float(np.linalg.norm(clipped - residual_arr)),
    }
    return clipped, safety_info


class ResidualSafetyFilter:
    """Callable residual safety filter."""

    def __init__(self, config: ResidualSafetyConfig | None = None) -> None:
        self.config = config or ResidualSafetyConfig()

    def __call__(
        self,
        residual: np.ndarray,
        current_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Apply configured residual safety limits."""
        return apply_residual_safety_filter(
            residual=residual,
            config=self.config,
            current_position=current_position,
        )


def _as_vector4(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != RESIDUAL_SHAPE:
        msg = f"{name} must have shape {RESIDUAL_SHAPE}, got {arr.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return arr
