import json

import numpy as np

from rlinf.algorithms.residual_td3.episode_sink import (
    EpisodeSummary,
    EvalSink,
    InterventionEvent,
    ReplaySink,
    StepEvent,
)


def _step_event() -> StepEvent:
    return StepEvent(
        episode_id=0,
        seed=123,
        env_step=50,
        action_mode="ee16_zero_residual",
        execution_stage="handover",
        trigger_source="handover_gate",
        gate_score=1.0,
        selected_action_chunk_indices=[0, 1],
        has_intervention=True,
        reward=0.0,
        done=False,
        metadata={"env_step_start": 0, "env_step_end": 50, "replan_id": 0},
    )


def _intervention_event(step_i: int = 0) -> InterventionEvent:
    obs = np.linspace(0.0, 1.0, 21, dtype=np.float32) + step_i
    return InterventionEvent(
        episode_id=0,
        seed=123,
        env_step=50 + step_i,
        intervention_id=0,
        intervention_step_i=step_i,
        stage="handover",
        trigger_source="handover_gate",
        action_source="zero",
        pred_delta_local_xyz=[0.0, 0.0, 0.0],
        applied_delta_local_xyz=[0.0, 0.0, 0.0],
        applied_delta_world_xyz=[0.0, 0.0, 0.0],
        base_ee16=[0.0] * 16,
        exec_ee16=[0.0] * 16,
        saturation=0,
        has_nan_or_inf=0,
        metadata={"obs_vector": obs.tolist(), "gate_score": 1.0},
    )


def _summary() -> EpisodeSummary:
    return EpisodeSummary(
        episode_id=0,
        seed=123,
        success=False,
        failure_reason="mock_timeout",
        episode_len=100,
        num_interventions=1,
        total_ee_steps=2,
        num_pregrasp_interventions=0,
        num_handover_interventions=1,
        first_pregrasp_trigger_step=None,
        first_handover_trigger_step=50,
        mock_runtime=True,
        metadata={"return": 0.0},
    )


def test_event_schemas_are_json_serializable():
    json.dumps(_step_event().to_dict())
    json.dumps(_intervention_event().to_dict())
    json.dumps(_summary().to_dict())


def test_eval_sink_writes_summary_and_logs(tmp_path):
    sink = EvalSink(
        tmp_path,
        execution_mode="handover_only_k50",
        mock_runtime=True,
        real_env=False,
        real_model=False,
    )
    sink.on_episode_start(0, 123)
    sink.on_step(_step_event())
    sink.on_intervention(_intervention_event())
    sink.on_episode_end(_summary())

    summary = sink.close()

    assert summary["mock_runtime"] is True
    assert summary["real_env"] is False
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "hybrid_log_episode_0000.csv").exists()
    assert (tmp_path / "intervention_records_episode_0000.csv").exists()


def test_replay_sink_writes_minimal_replay(tmp_path):
    sink = ReplaySink(
        tmp_path,
        action_source="zero",
        mock_runtime=True,
        real_env=False,
        real_model=False,
    )
    sink.on_episode_start(0, 123)
    sink.on_intervention(_intervention_event(0))
    sink.on_intervention(_intervention_event(1))
    sink.on_episode_end(_summary())

    summary = sink.close()

    assert summary["reward_summary"]["mock_runtime"] is True
    assert (tmp_path / "replay.npz").exists()
    assert (tmp_path / "episodes_summary.csv").exists()
    assert (tmp_path / "reward_summary.json").exists()
    with np.load(tmp_path / "replay.npz") as replay:
        assert replay["obs"].shape == (2, 21)
        assert replay["action"].shape == (2, 3)
        assert replay["reward"].shape == (2,)
        assert replay["next_obs"].shape == (2, 21)
        assert replay["done"].shape == (2,)
