import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.algorithms.residual_td3.residual_actor import (
    ResidualActorConfig,
    ZeroInitResidualActorMLP,
)
from rlinf.algorithms.residual_td3.residual_ee_intervention import (
    ENDPOSE16_LEFT_XYZ,
    ENDPOSE16_RIGHT_XYZ,
    BCResidualActor,
    ConstantResidualActor,
    ResidualEEInterventionConfig,
    ResidualEEInterventionRunner,
    ZeroInitResidualActor,
    ZeroResidualActor,
    residual_ee_observation_to_tensor,
)


class FakeBridge:
    def __init__(self, left_quat=(1.0, 0.0, 0.0, 0.0)):
        self.left_quat = torch.tensor(left_quat, dtype=torch.float32)

    def qpos14_to_endpose16(self, qpos14_chunk, current_obs=None):
        del current_obs
        batch, chunk, _ = qpos14_chunk.shape
        out = torch.zeros(batch, chunk, 16, dtype=qpos14_chunk.dtype, device=qpos14_chunk.device)
        out[..., 0:3] = qpos14_chunk[..., 0:3]
        out[..., 3:7] = self.left_quat.to(device=qpos14_chunk.device, dtype=qpos14_chunk.dtype)
        out[..., 7] = qpos14_chunk[..., 6]
        out[..., 8:11] = qpos14_chunk[..., 7:10]
        out[..., 11] = 1.0
        out[..., 15] = qpos14_chunk[..., 13]
        return out


class FakeChunkActor:
    def __init__(self, chunk):
        self.chunk = torch.as_tensor(chunk, dtype=torch.float32)

    def predict_delta_chunk_local_xyz(self, obs_vector):
        assert obs_vector.shape == (21,)
        return self.chunk

    def predict_delta_local_xyz(self, obs):
        del obs
        return self.chunk[0]


def _qpos_chunk(chunk_len=6):
    qpos = torch.zeros(1, chunk_len, 14)
    for idx in range(chunk_len):
        qpos[0, idx, 0:3] = torch.tensor([float(idx), 0.0, 0.0])
        qpos[0, idx, 7:10] = torch.tensor([10.0 + idx, 20.0 + idx, 30.0 + idx])
        qpos[0, idx, 6] = 0.5
        qpos[0, idx, 13] = 0.75
    return qpos


def test_zero_residual_leaves_right_xyz_unchanged_from_base_ee16():
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        ZeroResidualActor(),
        ResidualEEInterventionConfig(horizon_k=3),
    )

    result = runner.run(_qpos_chunk())

    np.testing.assert_allclose(
        result.exec_ee16_chunk[..., ENDPOSE16_RIGHT_XYZ].numpy(),
        result.base_ee16_chunk[..., ENDPOSE16_RIGHT_XYZ].numpy(),
    )


def test_constant_local_residual_identity_rotation_maps_to_world_xyz():
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        ConstantResidualActor((0.01, -0.02, 0.03)),
        ResidualEEInterventionConfig(
            horizon_k=1,
            max_delta_local_xyz=1.0,
            learned_residual_control_enabled=True,
        ),
    )

    result = runner.run(_qpos_chunk())
    delta = result.exec_ee16_chunk[0, 0, ENDPOSE16_RIGHT_XYZ] - result.base_ee16_chunk[
        0, 0, ENDPOSE16_RIGHT_XYZ
    ]

    np.testing.assert_allclose(delta.numpy(), np.array([0.01, -0.02, 0.03]), atol=1e-6)


def test_constant_local_residual_rotates_by_left_ee_quaternion():
    quat_z90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    runner = ResidualEEInterventionRunner(
        FakeBridge(left_quat=quat_z90),
        ConstantResidualActor((0.01, 0.0, 0.0)),
        ResidualEEInterventionConfig(
            horizon_k=1,
            max_delta_local_xyz=1.0,
            learned_residual_control_enabled=True,
        ),
    )

    result = runner.run(_qpos_chunk())
    delta = result.exec_ee16_chunk[0, 0, ENDPOSE16_RIGHT_XYZ] - result.base_ee16_chunk[
        0, 0, ENDPOSE16_RIGHT_XYZ
    ]

    np.testing.assert_allclose(delta.numpy(), np.array([0.0, 0.01, 0.0]), atol=1e-6)


def test_clipping_applies_in_local_frame():
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        ConstantResidualActor((0.5, -0.5, 0.01)),
        ResidualEEInterventionConfig(
            horizon_k=1,
            max_delta_local_xyz=0.02,
            learned_residual_control_enabled=True,
        ),
    )

    result = runner.run(_qpos_chunk())

    assert result.records[0]["delta_local_xyz"] == pytest.approx([0.02, -0.02, 0.01])


def test_target_horizon_offset_selects_expected_chunk_index():
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        ZeroResidualActor(),
        ResidualEEInterventionConfig(horizon_k=2, target_horizon_offset=3),
    )

    result = runner.run(_qpos_chunk())

    assert result.metadata["selected_indices"] == [3, 4]
    assert [row["action_chunk_index"] for row in result.records] == [3, 4]


def test_left_deadband_keeps_previous_left_target_when_movement_is_small():
    qpos = _qpos_chunk(chunk_len=2)
    qpos[0, 1, 0:3] = torch.tensor([1e-5, 0.0, 0.0])
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        ZeroResidualActor(),
        ResidualEEInterventionConfig(
            horizon_k=2,
            left_stabilization_mode="deadband",
            left_deadband_xyz=1e-4,
        ),
    )

    result = runner.run(qpos)

    assert result.metadata["left_stabilization_count"] == 1
    np.testing.assert_allclose(
        result.exec_ee16_chunk[0, 1, ENDPOSE16_LEFT_XYZ].numpy(),
        result.exec_ee16_chunk[0, 0, ENDPOSE16_LEFT_XYZ].numpy(),
    )


def test_metadata_contains_selected_indices_and_delta_norms():
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        ConstantResidualActor((0.01, 0.0, 0.0)),
        ResidualEEInterventionConfig(
            horizon_k=2,
            max_delta_local_xyz=1.0,
            learned_residual_control_enabled=True,
        ),
    )

    result = runner.run(_qpos_chunk(), gate_score=0.9, gate_threshold=0.6)

    assert result.metadata["num_steps_executed"] == 2
    assert result.metadata["selected_indices"] == [0, 1]
    assert result.metadata["max_delta_norm"] == pytest.approx(0.01)
    assert result.metadata["mean_delta_norm"] == pytest.approx(0.01)
    assert result.records[0]["gate_score"] == 0.9
    assert result.records[0]["gate_threshold"] == 0.6


def test_zero_init_residual_actor_outputs_bounded_zero_chunk():
    actor = ZeroInitResidualActor(obs_dim=21, hidden_dim=16, chunk_len=4, delta_max=0.02)
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        actor,
        ResidualEEInterventionConfig(horizon_k=4, max_delta_local_xyz=0.02),
    )

    result = runner.run(_qpos_chunk(chunk_len=4))

    np.testing.assert_allclose(
        result.exec_ee16_chunk[..., ENDPOSE16_RIGHT_XYZ].numpy(),
        result.base_ee16_chunk[..., ENDPOSE16_RIGHT_XYZ].numpy(),
        atol=0.0,
    )
    assert result.metadata["max_delta_norm"] == 0.0


def test_zero_init_residual_actor_predict_delta_uses_residual_observation():
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        ZeroResidualActor(),
        ResidualEEInterventionConfig(horizon_k=1),
    )
    base = FakeBridge().qpos14_to_endpose16(_qpos_chunk(chunk_len=1))[0, 0]
    obs = runner._build_observation(base, gate_score=1.0, intervention_step_i=0)
    actor = ZeroInitResidualActor(chunk_len=3)

    vector = residual_ee_observation_to_tensor(obs)
    delta = actor.predict_delta_local_xyz(obs)

    assert vector.shape == (21,)
    assert float(vector[-2]) == 0.0
    assert float(vector[-1]) == 1.0
    np.testing.assert_allclose(delta.numpy(), np.zeros(3), atol=0.0)


def test_bc_actor_dry_run_logs_prediction_but_applies_zero_delta():
    actor = FakeChunkActor([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        actor,
        ResidualEEInterventionConfig(horizon_k=2, max_delta_local_xyz=0.05, dry_run=True),
    )

    result = runner.run(_qpos_chunk(chunk_len=2), gate_score=0.9, gate_threshold=0.6)

    np.testing.assert_allclose(
        result.exec_ee16_chunk.numpy(),
        result.base_ee16_chunk.numpy(),
        atol=0.0,
    )
    assert result.records[0]["dry_run"] is True
    assert result.records[0]["pred_delta_local_xyz"] == pytest.approx([0.01, 0.0, 0.0])
    assert result.records[0]["applied_delta_local_xyz"] == pytest.approx([0.0, 0.0, 0.0])
    assert result.metadata["pred_delta_norm_max"] == pytest.approx(0.02)
    assert result.metadata["max_delta_norm"] == 0.0


def test_bc_actor_dry_run_exec_matches_zero_residual_exec():
    qpos = _qpos_chunk(chunk_len=2)
    dry_run_result = ResidualEEInterventionRunner(
        FakeBridge(),
        FakeChunkActor([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]]),
        ResidualEEInterventionConfig(horizon_k=2, max_delta_local_xyz=0.05, dry_run=True),
    ).run(qpos, gate_score=0.9, gate_threshold=0.6)
    zero_result = ResidualEEInterventionRunner(
        FakeBridge(),
        ZeroResidualActor(),
        ResidualEEInterventionConfig(horizon_k=2, max_delta_local_xyz=0.05, dry_run=True),
    ).run(qpos, gate_score=0.9, gate_threshold=0.6)

    np.testing.assert_allclose(dry_run_result.exec_ee16_chunk.numpy(), zero_result.exec_ee16_chunk.numpy(), atol=0.0)
    assert dry_run_result.records[0]["obs_feature_reference_step"] == 0
    assert dry_run_result.records[0]["gate_positive_fraction_source"] == "gate_score"


def test_learned_control_applies_scaled_delta_and_logs_metadata():
    actor = FakeChunkActor([[0.01, 0.0, 0.0]])
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        actor,
        ResidualEEInterventionConfig(
            horizon_k=1,
            max_delta_local_xyz=0.05,
            residual_scale=0.25,
            learned_residual_control_enabled=True,
        ),
    )

    result = runner.run(_qpos_chunk(chunk_len=1), gate_score=0.9, gate_threshold=0.6)

    assert result.records[0]["learned_residual_control_enabled"] is True
    assert result.records[0]["dry_run"] is False
    assert result.records[0]["applied_delta_local_xyz"] == pytest.approx([0.0025, 0.0, 0.0])
    assert result.records[0]["applied_delta_norm"] == pytest.approx(0.0025)
    assert result.records[0]["residual_scale"] == pytest.approx(0.25)
    assert result.metadata["applied_delta_norm_mean"] == pytest.approx(0.0025)
    assert result.metadata["residual_scale"] == pytest.approx(0.25)


def test_learned_control_exec_right_xyz_uses_world_delta():
    quat_z90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    actor = FakeChunkActor([[0.01, 0.0, 0.0]])
    runner = ResidualEEInterventionRunner(
        FakeBridge(left_quat=quat_z90),
        actor,
        ResidualEEInterventionConfig(
            horizon_k=1,
            max_delta_local_xyz=0.05,
            residual_scale=0.5,
            learned_residual_control_enabled=True,
        ),
    )

    result = runner.run(_qpos_chunk(chunk_len=1))

    assert result.records[0]["applied_delta_world_xyz"] == pytest.approx([0.0, 0.005, 0.0], abs=1e-6)
    assert result.records[0]["exec_ee16_before_residual"][8:11] == pytest.approx([10.0, 20.0, 30.0])
    assert result.records[0]["exec_ee16_after_residual"][8:11] == pytest.approx([10.0, 20.005, 30.0], abs=1e-6)


def test_learned_control_applies_clipping_before_scaling():
    actor = FakeChunkActor([[0.5, -0.5, 0.01]])
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        actor,
        ResidualEEInterventionConfig(
            horizon_k=1,
            max_delta_local_xyz=0.02,
            residual_scale=0.25,
            learned_residual_control_enabled=True,
        ),
    )

    result = runner.run(_qpos_chunk(chunk_len=1))

    assert result.records[0]["applied_delta_local_xyz"] == pytest.approx([0.005, -0.005, 0.0025])


def test_dry_run_and_learned_control_cannot_both_be_true():
    with pytest.raises(ValueError, match="cannot both be true"):
        ResidualEEInterventionRunner(
            FakeBridge(),
            ZeroResidualActor(),
            ResidualEEInterventionConfig(dry_run=True, learned_residual_control_enabled=True),
        )


def test_saturation_and_nan_flags_are_recorded():
    actor = FakeChunkActor([[0.5, float("inf"), 0.0]])
    runner = ResidualEEInterventionRunner(
        FakeBridge(),
        actor,
        ResidualEEInterventionConfig(horizon_k=1, max_delta_local_xyz=0.02, dry_run=True),
    )

    result = runner.run(_qpos_chunk(chunk_len=1), gate_score=1.0, gate_threshold=0.6)

    assert result.records[0]["saturation"] == 1
    assert result.records[0]["has_nan_or_inf"] == 1
    assert result.metadata["saturation_count"] == 1
    assert result.metadata["nan_inf_count"] == 1


def test_bc_actor_checkpoint_can_load_and_predict_chunk(tmp_path):
    model = ZeroInitResidualActorMLP(
        ResidualActorConfig(obs_dim=21, hidden_dim=8, chunk_len=4, delta_max=0.05, zero_init_output=False)
    )
    checkpoint_path = Path(tmp_path) / "residual_actor_bc.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "actor_config": model.cfg.__dict__,
            "metrics": {"train_episodes": [0, 1], "val_episodes": [2]},
        },
        checkpoint_path,
    )

    actor = BCResidualActor.load_from_checkpoint(checkpoint_path, device="cpu")
    pred = actor.predict_delta_chunk_local_xyz(torch.zeros(21, dtype=torch.float32))

    assert actor.chunk_len == 4
    assert pred.shape == (4, 3)
