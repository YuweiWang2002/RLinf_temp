import numpy as np
import pytest
import torch

from rlinf.algorithms.residual_td3.pregrasp_trigger import (
    PregraspTriggerConfig,
    choose_intervention_stage,
    detect_pregrasp_trigger,
)
from rlinf.algorithms.residual_td3.residual_ee_intervention import (
    ResidualEEInterventionConfig,
    ResidualEEInterventionRunner,
    ZeroResidualActor,
)
from scripts.rollout_pi05_gate_controlled_hybrid_zero_residual import parse_args


class FakeBridge:
    def qpos14_to_endpose16(self, qpos14_chunk, current_obs=None):
        del current_obs
        batch, chunk, _ = qpos14_chunk.shape
        out = torch.zeros(batch, chunk, 16, dtype=qpos14_chunk.dtype, device=qpos14_chunk.device)
        out[..., 0:3] = qpos14_chunk[..., 0:3]
        out[..., 3] = 1.0
        out[..., 7] = qpos14_chunk[..., 6]
        out[..., 8:11] = qpos14_chunk[..., 7:10]
        out[..., 11] = 1.0
        out[..., 15] = qpos14_chunk[..., 13]
        return out


def _qpos_chunk(chunk_len=6):
    qpos = torch.zeros(1, chunk_len, 14)
    for idx in range(chunk_len):
        qpos[0, idx, 0:3] = torch.tensor([float(idx), 0.0, 0.0])
        qpos[0, idx, 7:10] = torch.tensor([10.0 + idx, 20.0 + idx, 30.0 + idx])
        qpos[0, idx, 6] = 0.5
        qpos[0, idx, 13] = 0.75
    return qpos


def _chunk(left_gripper):
    chunk = np.zeros((len(left_gripper), 14), dtype=np.float32)
    chunk[:, 6] = np.asarray(left_gripper, dtype=np.float32)
    return chunk


def test_pregrasp_trigger_detects_lower_closing_delta():
    result = detect_pregrasp_trigger(
        _chunk([1.0, 0.95, 0.7, 0.65]),
        env_step=100,
        config=PregraspTriggerConfig(
            enabled=True,
            close_direction="lower",
            closing_delta_threshold=0.2,
        ),
    )

    assert result.triggered is True
    assert result.trigger_reason == "closing_delta"
    assert result.first_closing_chunk_index == 2
    assert result.left_gripper_start == pytest.approx(1.0)
    assert result.left_gripper_min == pytest.approx(0.65)


def test_pregrasp_trigger_detects_higher_closing_threshold():
    result = detect_pregrasp_trigger(
        _chunk([0.0, 0.1, 0.25, 0.8]),
        env_step=100,
        config=PregraspTriggerConfig(
            enabled=True,
            close_direction="higher",
            close_threshold=0.75,
            closing_delta_threshold=0.5,
        ),
    )

    assert result.triggered is True
    assert result.trigger_reason == "close_threshold_and_delta"
    assert result.first_closing_chunk_index == 3
    assert result.left_gripper_max == pytest.approx(0.8)


def test_pregrasp_trigger_does_not_fire_without_closing_event():
    result = detect_pregrasp_trigger(
        _chunk([1.0, 0.98, 0.99, 0.97]),
        env_step=100,
        config=PregraspTriggerConfig(
            enabled=True,
            close_direction="lower",
            closing_delta_threshold=0.2,
        ),
    )

    assert result.triggered is False
    assert result.trigger_reason == "none"


def test_pregrasp_trigger_lower_threshold_requires_crossing_from_open_side():
    result = detect_pregrasp_trigger(
        _chunk([0.0, 0.02, 0.01, 0.03]),
        env_step=100,
        config=PregraspTriggerConfig(
            enabled=True,
            close_direction="lower",
            close_threshold=0.25,
            closing_delta_threshold=0.2,
        ),
    )

    assert result.triggered is False
    assert result.trigger_reason == "none"


def test_pregrasp_trigger_higher_threshold_requires_crossing_from_open_side():
    result = detect_pregrasp_trigger(
        _chunk([1.0, 0.98, 0.99, 0.97]),
        env_step=100,
        config=PregraspTriggerConfig(
            enabled=True,
            close_direction="higher",
            close_threshold=0.75,
            closing_delta_threshold=0.2,
        ),
    )

    assert result.triggered is False
    assert result.trigger_reason == "none"


def test_pregrasp_trigger_cooldown_blocks_event():
    result = detect_pregrasp_trigger(
        _chunk([1.0, 0.5, 0.4]),
        env_step=125,
        last_trigger_step=100,
        config=PregraspTriggerConfig(
            enabled=True,
            close_direction="lower",
            closing_delta_threshold=0.2,
            cooldown_steps=50,
        ),
    )

    assert result.triggered is False
    assert result.in_cooldown is True
    assert result.trigger_reason == "cooldown"


def test_pregrasp_trigger_requires_direction_when_enabled():
    with pytest.raises(ValueError, match="gripper-close-direction"):
        detect_pregrasp_trigger(
            _chunk([1.0, 0.5]),
            env_step=0,
            config=PregraspTriggerConfig(enabled=True),
        )


def test_pregrasp_intervention_stage_metadata_is_recorded():
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        ZeroResidualActor(),
        ResidualEEInterventionConfig(horizon_k=2),
    )

    result = runner.run(
        _qpos_chunk(chunk_len=2),
        intervention_stage="pregrasp",
        trigger_source="gripper_chunk",
        trigger_metadata={"pregrasp_trigger_score": 0.5},
    )

    assert result.metadata["intervention_stage"] == "pregrasp"
    assert result.metadata["trigger_source"] == "gripper_chunk"
    assert result.metadata["pregrasp_trigger_score"] == pytest.approx(0.5)
    assert result.records[0]["intervention_stage"] == "pregrasp"
    assert result.records[0]["trigger_source"] == "gripper_chunk"


def test_handover_gate_has_priority_over_pregrasp_trigger():
    assert (
        choose_intervention_stage(
            handover_gate=True,
            pregrasp_trigger=True,
            enable_handover=True,
            enable_pregrasp=True,
        )
        == "handover"
    )


def test_default_cli_does_not_enable_pregrasp():
    args = parse_args(
        [
            "--checkpoint",
            "model.safetensors",
            "--save-dir",
            "out",
        ]
    )

    assert args.execution_mode == "gate_controlled_hybrid"
    assert args.enable_pregrasp_intervention is False
