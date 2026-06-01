import math

import numpy as np
import torch

from rlinf.algorithms.residual_td3.residual_bc_dataset import (
    ResidualBCTargetBuilder,
    ResidualBCTargetConfig,
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


def _cfg(**kwargs):
    values = {
        "chunk_len": 5,
        "residual_horizon_k": 3,
        "target_horizon_offsets": (0, 1, 2),
        "max_delta_local_xyz": 0.05,
        "gate_positive_only": True,
        "min_positive_frames": 1,
    }
    values.update(kwargs)
    return ResidualBCTargetConfig(**values)


def _qpos_chunk(chunk_len=5):
    qpos = torch.zeros(chunk_len, 14)
    for idx in range(chunk_len):
        qpos[idx, 0:3] = torch.tensor([float(idx), 0.0, 0.0])
        qpos[idx, 7:10] = torch.tensor([1.0 + idx, 2.0 + idx, 3.0 + idx])
        qpos[idx, 6] = 0.5
        qpos[idx, 13] = 0.75
    return qpos


def test_zero_target_when_expert_equals_base():
    builder = ResidualBCTargetBuilder(_cfg(), FakeBridge())
    qpos = _qpos_chunk()

    sample = builder.build_sample(
        base_qpos_chunk=qpos,
        expert_qpos_chunk=qpos,
        gate_seq=np.ones(5, dtype=np.float32),
    )

    np.testing.assert_allclose(sample["target_delta_local_xyz"], np.zeros((3, 3)), atol=0.0)
    assert sample["clip_mask"].sum() == 0


def test_identity_left_rotation_maps_world_delta_to_local():
    builder = ResidualBCTargetBuilder(_cfg(target_horizon_offsets=(0,)), FakeBridge())
    base = _qpos_chunk()
    expert = base.clone()
    expert[0, 7:10] += torch.tensor([0.01, -0.02, 0.03])

    sample = builder.build_sample(
        base_qpos_chunk=base,
        expert_qpos_chunk=expert,
        gate_seq=np.ones(5, dtype=np.float32),
    )

    np.testing.assert_allclose(sample["target_delta_world_xyz"][0], [0.01, -0.02, 0.03], atol=1e-6)
    np.testing.assert_allclose(sample["target_delta_local_xyz"][0], [0.01, -0.02, 0.03], atol=1e-6)


def test_left_rotation_uses_base_left_quaternion():
    quat_z90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    builder = ResidualBCTargetBuilder(
        _cfg(target_horizon_offsets=(0,), max_delta_local_xyz=1.0),
        FakeBridge(left_quat=quat_z90),
    )
    base = _qpos_chunk()
    expert = base.clone()
    expert[0, 7:10] += torch.tensor([0.0, 0.01, 0.0])

    sample = builder.build_sample(
        base_qpos_chunk=base,
        expert_qpos_chunk=expert,
        gate_seq=np.ones(5, dtype=np.float32),
    )

    np.testing.assert_allclose(sample["target_delta_local_xyz"][0], [0.01, 0.0, 0.0], atol=1e-6)


def test_clipping_applies_in_local_frame():
    builder = ResidualBCTargetBuilder(
        _cfg(target_horizon_offsets=(0,), max_delta_local_xyz=0.02),
        FakeBridge(),
    )
    base = _qpos_chunk()
    expert = base.clone()
    expert[0, 7:10] += torch.tensor([0.5, -0.5, 0.01])

    sample = builder.build_sample(
        base_qpos_chunk=base,
        expert_qpos_chunk=expert,
        gate_seq=np.ones(5, dtype=np.float32),
    )

    np.testing.assert_allclose(sample["target_delta_local_xyz"][0], [0.02, -0.02, 0.01], atol=1e-6)
    np.testing.assert_array_equal(sample["clip_mask"][0], [True, True, False])


def test_gate_positive_filtering_drops_all_zero_gate_seq():
    builder = ResidualBCTargetBuilder(_cfg(), FakeBridge())
    qpos = _qpos_chunk()

    assert builder.build_sample(
        base_qpos_chunk=qpos,
        expert_qpos_chunk=qpos,
        gate_seq=np.zeros(5, dtype=np.float32),
    ) is None

    assert builder.build_sample(
        base_qpos_chunk=qpos,
        expert_qpos_chunk=qpos,
        gate_seq=np.array([0, 0, 1, 0, 0], dtype=np.float32),
    ) is not None


def test_horizon_offsets_select_expected_chunk_indices():
    builder = ResidualBCTargetBuilder(_cfg(target_horizon_offsets=(0, 1, 2)), FakeBridge())
    base = _qpos_chunk()
    expert = base.clone()
    expert[0, 7] += 0.01
    expert[1, 8] += 0.02
    expert[2, 9] += 0.03

    sample = builder.build_sample(
        base_qpos_chunk=base,
        expert_qpos_chunk=expert,
        gate_seq=np.ones(5, dtype=np.float32),
    )

    np.testing.assert_allclose(
        sample["target_delta_local_xyz"],
        [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03]],
        atol=1e-6,
    )


def test_output_shapes_and_observation_keys_exist():
    builder = ResidualBCTargetBuilder(_cfg(), FakeBridge())
    qpos = _qpos_chunk()

    sample = builder.build_sample(
        base_qpos_chunk=qpos,
        expert_qpos_chunk=qpos,
        gate_seq=np.ones(5, dtype=np.float32),
        current_state=np.arange(14, dtype=np.float32),
        episode_index=3,
        frame_index=7,
    )

    assert sample["target_delta_local_xyz"].shape == (3, 3)
    assert sample["base_endpose16"].shape == (5, 16)
    assert sample["expert_endpose16"].shape == (5, 16)
    assert sample["metadata"]["episode_index"] == 3
    obs = sample["obs_features"]
    for key in (
        "base_left_xyz",
        "base_left_quat_wxyz",
        "base_right_xyz",
        "base_right_quat_wxyz",
        "relative_right_xyz_in_left_frame",
        "left_gripper",
        "right_gripper",
        "gate_label",
        "horizon_offset",
        "current_qpos14_state",
    ):
        assert key in obs
