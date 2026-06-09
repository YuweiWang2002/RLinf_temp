"""Pure stage and execution-plan decisions for residual rollout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InterventionStage(str, Enum):
    """Execution stage selected for one replan chunk."""

    QPOS = "qpos"
    PREGRASP = "pregrasp"
    HANDOVER = "handover"


class TriggerSource(str, Enum):
    """Source that selected an intervention stage."""

    NONE = "none"
    PREGRASP_GRIPPER_CHUNK = "pregrasp_gripper_chunk"
    HANDOVER_GATE = "handover_gate"


@dataclass(frozen=True)
class StageDecision:
    """Binary stage signals before resolving priority."""

    handover_gate_binary: bool
    pregrasp_trigger_binary: bool
    enable_handover: bool
    enable_pregrasp: bool


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved action execution plan for one rollout replan."""

    stage: InterventionStage
    trigger_source: TriggerSource
    execution_mode: str
    residual_horizon_k: int
    target_horizon_offset: int
    selected_action_chunk_indices: list[int]
    env_action_mode: str

    @property
    def ee16_binary(self) -> bool:
        return self.stage in {InterventionStage.PREGRASP, InterventionStage.HANDOVER}

    @property
    def gate_binary(self) -> bool:
        return self.stage == InterventionStage.HANDOVER

    @property
    def intervention_stage_name(self) -> str:
        if self.stage == InterventionStage.QPOS:
            return "none"
        return self.stage.value

    @property
    def runtime_trigger_source(self) -> str:
        if self.trigger_source == TriggerSource.PREGRASP_GRIPPER_CHUNK:
            return "gripper_chunk"
        return self.trigger_source.value


def handover_intervention_enabled_for_mode(execution_mode: str) -> bool:
    """Return whether an execution mode allows handover EE intervention."""

    return execution_mode in {
        "gate_controlled_hybrid",
        "handover_only_k50",
        "pregrasp_plus_handover_k50",
    }


def pregrasp_intervention_enabled_for_mode(
    execution_mode: str,
    *,
    enable_pregrasp_intervention: bool = False,
) -> bool:
    """Return whether an execution mode allows pregrasp EE intervention."""

    return bool(enable_pregrasp_intervention) or execution_mode in {
        "pregrasp_only_k50",
        "pregrasp_plus_handover_k50",
    }


def decide_stage(decision: StageDecision) -> tuple[InterventionStage, TriggerSource]:
    """Resolve stage priority for one replan chunk."""

    if decision.enable_handover and decision.handover_gate_binary:
        return InterventionStage.HANDOVER, TriggerSource.HANDOVER_GATE
    if decision.enable_pregrasp and decision.pregrasp_trigger_binary:
        return InterventionStage.PREGRASP, TriggerSource.PREGRASP_GRIPPER_CHUNK
    return InterventionStage.QPOS, TriggerSource.NONE


def decide_execution_plan(
    *,
    execution_mode: str,
    gate_prob: float,
    gate_threshold: float,
    pregrasp_triggered: bool,
    enable_pregrasp_intervention: bool = False,
    residual_horizon_k: int = 50,
    target_horizon_offset: int = 0,
    action_chunk_len: int | None = None,
    handover_interventions_so_far: int = 0,
    max_handover_interventions_per_episode: int | None = None,
) -> ExecutionPlan:
    """Build a pure execution plan from trigger signals and rollout flags."""

    enable_handover = handover_intervention_enabled_for_mode(execution_mode)
    if (
        max_handover_interventions_per_episode is not None
        and handover_interventions_so_far >= max_handover_interventions_per_episode
    ):
        enable_handover = False
    enable_pregrasp = pregrasp_intervention_enabled_for_mode(
        execution_mode,
        enable_pregrasp_intervention=enable_pregrasp_intervention,
    )
    decision = StageDecision(
        handover_gate_binary=bool(enable_handover and gate_prob >= gate_threshold),
        pregrasp_trigger_binary=bool(pregrasp_triggered),
        enable_handover=enable_handover,
        enable_pregrasp=enable_pregrasp,
    )
    stage, trigger_source = decide_stage(decision)
    ee16_binary = stage in {InterventionStage.PREGRASP, InterventionStage.HANDOVER}
    selected_indices = _selected_action_indices(
        residual_horizon_k=residual_horizon_k,
        target_horizon_offset=target_horizon_offset,
        action_chunk_len=action_chunk_len,
    )
    return ExecutionPlan(
        stage=stage,
        trigger_source=trigger_source,
        execution_mode="ee16_zero_residual" if ee16_binary else "qpos14",
        residual_horizon_k=int(residual_horizon_k),
        target_horizon_offset=int(target_horizon_offset),
        selected_action_chunk_indices=selected_indices if ee16_binary else [],
        env_action_mode="ee16" if ee16_binary else "qpos14",
    )


def _selected_action_indices(
    *,
    residual_horizon_k: int,
    target_horizon_offset: int,
    action_chunk_len: int | None,
) -> list[int]:
    start = max(0, int(target_horizon_offset))
    end = start + max(0, int(residual_horizon_k))
    if action_chunk_len is not None:
        end = min(end, int(action_chunk_len))
    return list(range(start, end))
