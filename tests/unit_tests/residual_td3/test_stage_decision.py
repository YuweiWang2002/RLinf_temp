from rlinf.algorithms.residual_td3.stage_decision import (
    InterventionStage,
    TriggerSource,
    decide_execution_plan,
)


def test_handover_only_k50_gate_trigger_selects_handover_plan():
    plan = decide_execution_plan(
        execution_mode="handover_only_k50",
        gate_prob=0.9,
        gate_threshold=0.5,
        pregrasp_triggered=False,
        residual_horizon_k=50,
        target_horizon_offset=0,
        action_chunk_len=60,
    )

    assert plan.stage == InterventionStage.HANDOVER
    assert plan.trigger_source == TriggerSource.HANDOVER_GATE
    assert plan.execution_mode == "ee16_zero_residual"
    assert plan.gate_binary is True
    assert plan.ee16_binary is True
    assert plan.selected_action_chunk_indices == list(range(50))


def test_handover_only_k50_gate_not_triggered_selects_qpos_plan():
    plan = decide_execution_plan(
        execution_mode="handover_only_k50",
        gate_prob=0.1,
        gate_threshold=0.5,
        pregrasp_triggered=False,
        residual_horizon_k=50,
        action_chunk_len=60,
    )

    assert plan.stage == InterventionStage.QPOS
    assert plan.trigger_source == TriggerSource.NONE
    assert plan.execution_mode == "qpos14"
    assert plan.gate_binary is False
    assert plan.ee16_binary is False
    assert plan.selected_action_chunk_indices == []


def test_pregrasp_only_k50_trigger_selects_pregrasp_plan():
    plan = decide_execution_plan(
        execution_mode="pregrasp_only_k50",
        gate_prob=0.0,
        gate_threshold=0.5,
        pregrasp_triggered=True,
        residual_horizon_k=50,
        action_chunk_len=60,
    )

    assert plan.stage == InterventionStage.PREGRASP
    assert plan.trigger_source == TriggerSource.PREGRASP_GRIPPER_CHUNK
    assert plan.runtime_trigger_source == "gripper_chunk"
    assert plan.intervention_stage_name == "pregrasp"


def test_pregrasp_plus_handover_prioritizes_handover_when_both_trigger():
    plan = decide_execution_plan(
        execution_mode="pregrasp_plus_handover_k50",
        gate_prob=0.9,
        gate_threshold=0.5,
        pregrasp_triggered=True,
        residual_horizon_k=50,
        action_chunk_len=60,
    )

    assert plan.stage == InterventionStage.HANDOVER
    assert plan.trigger_source == TriggerSource.HANDOVER_GATE
    assert plan.intervention_stage_name == "handover"


def test_selected_indices_respect_offset_k_and_chunk_len():
    plan = decide_execution_plan(
        execution_mode="handover_only_k50",
        gate_prob=1.0,
        gate_threshold=0.5,
        pregrasp_triggered=False,
        residual_horizon_k=4,
        target_horizon_offset=3,
        action_chunk_len=6,
    )

    assert plan.residual_horizon_k == 4
    assert plan.target_horizon_offset == 3
    assert plan.selected_action_chunk_indices == [3, 4, 5]


def test_default_mode_does_not_enable_pregrasp_without_flag():
    plan = decide_execution_plan(
        execution_mode="gate_controlled_hybrid",
        gate_prob=0.0,
        gate_threshold=0.5,
        pregrasp_triggered=True,
        enable_pregrasp_intervention=False,
        residual_horizon_k=50,
        action_chunk_len=60,
    )

    assert plan.stage == InterventionStage.QPOS


def test_default_mode_can_enable_pregrasp_with_explicit_flag():
    plan = decide_execution_plan(
        execution_mode="gate_controlled_hybrid",
        gate_prob=0.0,
        gate_threshold=0.5,
        pregrasp_triggered=True,
        enable_pregrasp_intervention=True,
        residual_horizon_k=50,
        action_chunk_len=60,
    )

    assert plan.stage == InterventionStage.PREGRASP


def test_cooldown_blocked_pregrasp_result_keeps_qpos_plan():
    plan = decide_execution_plan(
        execution_mode="pregrasp_only_k50",
        gate_prob=0.0,
        gate_threshold=0.5,
        pregrasp_triggered=False,
        residual_horizon_k=50,
        action_chunk_len=60,
    )

    assert plan.stage == InterventionStage.QPOS
