"""Left-gripper chunk trigger for pre-grasp EE execution switching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

GripperCloseDirection = Literal["lower", "higher"]
InterventionStage = Literal["none", "pregrasp", "handover"]


@dataclass(frozen=True)
class PregraspTriggerConfig:
    """Configuration for chunk-level left-gripper closing detection."""

    enabled: bool = False
    close_direction: GripperCloseDirection | None = None
    close_threshold: float | None = None
    closing_delta_threshold: float = 0.15
    cooldown_steps: int = 50


@dataclass(frozen=True)
class PregraspTriggerResult:
    """Result and diagnostics for one pre-grasp trigger decision."""

    triggered: bool
    score: float
    first_closing_chunk_index: int | None
    left_gripper_start: float
    left_gripper_end: float
    left_gripper_min: float
    left_gripper_max: float
    trigger_reason: str
    in_cooldown: bool = False


def detect_pregrasp_trigger(
    action_chunk: np.ndarray,
    *,
    env_step: int,
    config: PregraspTriggerConfig,
    last_trigger_step: int | None = None,
) -> PregraspTriggerResult:
    """Detect whether a qpos14 chunk predicts left-gripper closing."""

    if not config.enabled:
        return _empty_result(action_chunk, "disabled")
    if config.close_direction is None:
        raise ValueError("--gripper-close-direction is required when pregrasp intervention is enabled.")
    gripper = np.asarray(action_chunk, dtype=np.float32)
    if gripper.ndim == 3:
        gripper = gripper[0]
    if gripper.ndim != 2 or gripper.shape[-1] < 7:
        raise ValueError(f"action_chunk must have shape [C, 14] or [1, C, 14], got {tuple(gripper.shape)}.")
    seq = gripper[:, 6].astype(np.float32)
    if seq.size == 0:
        return _empty_result(action_chunk, "empty_chunk")
    stats = _stats(seq)
    if _in_cooldown(env_step, last_trigger_step, config.cooldown_steps):
        return PregraspTriggerResult(False, 0.0, None, *stats, "cooldown", True)

    trend_score = _closing_trend_score(seq, config.close_direction)
    threshold_hit_index = _threshold_hit_index(seq, config)
    threshold_triggered = threshold_hit_index is not None
    trend_triggered = trend_score >= float(config.closing_delta_threshold)
    triggered = bool(threshold_triggered or trend_triggered)
    first_index = threshold_hit_index if threshold_hit_index is not None else _first_delta_index(seq, config)
    reason = "none"
    if threshold_triggered and trend_triggered:
        reason = "close_threshold_and_delta"
    elif threshold_triggered:
        reason = "close_threshold"
    elif trend_triggered:
        reason = "closing_delta"
    return PregraspTriggerResult(
        triggered=triggered,
        score=float(trend_score),
        first_closing_chunk_index=first_index if triggered else None,
        left_gripper_start=stats[0],
        left_gripper_end=stats[1],
        left_gripper_min=stats[2],
        left_gripper_max=stats[3],
        trigger_reason=reason,
        in_cooldown=False,
    )


def choose_intervention_stage(
    *,
    handover_gate: bool,
    pregrasp_trigger: bool,
    enable_handover: bool,
    enable_pregrasp: bool,
) -> InterventionStage:
    """Resolve stage priority for one replan chunk."""

    if enable_handover and handover_gate:
        return "handover"
    if enable_pregrasp and pregrasp_trigger:
        return "pregrasp"
    return "none"


def _empty_result(action_chunk: np.ndarray, reason: str) -> PregraspTriggerResult:
    seq = np.asarray(action_chunk, dtype=np.float32)
    if seq.ndim == 3:
        seq = seq[0]
    if seq.ndim == 2 and seq.shape[-1] >= 7 and seq.shape[0] > 0:
        stats = _stats(seq[:, 6])
    else:
        stats = (0.0, 0.0, 0.0, 0.0)
    return PregraspTriggerResult(False, 0.0, None, *stats, reason, False)


def _stats(seq: np.ndarray) -> tuple[float, float, float, float]:
    return (float(seq[0]), float(seq[-1]), float(seq.min()), float(seq.max()))


def _in_cooldown(env_step: int, last_trigger_step: int | None, cooldown_steps: int) -> bool:
    return last_trigger_step is not None and cooldown_steps > 0 and env_step - last_trigger_step < cooldown_steps


def _closing_trend_score(seq: np.ndarray, direction: GripperCloseDirection) -> float:
    if direction == "lower":
        return float(seq[0] - seq.min())
    return float(seq.max() - seq[0])


def _threshold_hit_index(seq: np.ndarray, config: PregraspTriggerConfig) -> int | None:
    if config.close_threshold is None:
        return None
    if config.close_direction == "lower":
        if seq[0] <= float(config.close_threshold):
            return None
        hits = np.nonzero(seq <= float(config.close_threshold))[0]
    else:
        if seq[0] >= float(config.close_threshold):
            return None
        hits = np.nonzero(seq >= float(config.close_threshold))[0]
    return int(hits[0]) if hits.size else None


def _first_delta_index(seq: np.ndarray, config: PregraspTriggerConfig) -> int | None:
    threshold = float(config.closing_delta_threshold)
    if config.close_direction == "lower":
        hits = np.nonzero(seq[0] - seq >= threshold)[0]
    else:
        hits = np.nonzero(seq - seq[0] >= threshold)[0]
    return int(hits[0]) if hits.size else None
