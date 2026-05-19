import pytest
import torch

from rlinf.algorithms.residual_td3.action_adapter import (
    ResidualActionAdapter,
    ResidualActionSpec,
)


def test_aloha_qpos14_compose_changes_only_selected_indices():
    batch_size, chunk_len = 2, 5
    base = torch.zeros(batch_size, chunk_len, 14)
    residual = torch.ones(batch_size, chunk_len, 7)
    gate = torch.ones(batch_size, 1)
    adapter = ResidualActionAdapter(
        ResidualActionSpec(
            base_action_space="aloha_qpos14",
            residual_mode="joint_delta",
            base_action_dim=14,
            residual_action_indices=[7, 8, 9, 10, 11, 12, 13],
            residual_chunk_len=chunk_len,
            env_action_chunk_len=chunk_len,
        )
    )

    executed = adapter.compose_action(base, residual, gate)

    assert executed.shape == (batch_size, chunk_len, 14)
    assert torch.allclose(executed[..., :7], base[..., :7])
    assert torch.allclose(executed[..., 7:14], torch.ones(batch_size, chunk_len, 7))


def test_aloha_qpos14_right_xyz_mode_raises():
    with pytest.raises(ValueError, match="aloha_qpos14 only supports joint_delta"):
        ResidualActionAdapter(
            ResidualActionSpec(
                base_action_space="aloha_qpos14",
                residual_mode="right_xyz_delta",
                base_action_dim=14,
                residual_action_indices=[7, 8, 9, 10, 11, 12, 13],
            )
        )


def test_robotwin_endpose16_compose_changes_only_right_xyz():
    batch_size, chunk_len = 2, 5
    base = torch.arange(batch_size * chunk_len * 16, dtype=torch.float32).reshape(
        batch_size, chunk_len, 16
    )
    residual = torch.ones(batch_size, chunk_len, 3)
    gate = torch.ones(batch_size, chunk_len, 1)
    adapter = ResidualActionAdapter(
        ResidualActionSpec(
            base_action_space="robotwin_endpose16",
            residual_mode="right_xyz_delta",
            residual_frame="action",
            base_action_dim=16,
            residual_chunk_len=chunk_len,
            env_action_chunk_len=chunk_len,
        )
    )

    executed = adapter.compose_action(base, residual, gate)

    assert executed.shape == (batch_size, chunk_len, 16)
    assert torch.allclose(executed[..., 8:11], base[..., 8:11] + 1.0)
    unchanged_indices = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15]
    assert torch.allclose(executed[..., unchanged_indices], base[..., unchanged_indices])


def test_robotwin_left_ee_frame_requires_obs_rotation():
    batch_size, chunk_len = 2, 5
    base = torch.zeros(batch_size, chunk_len, 16)
    residual = torch.ones(batch_size, chunk_len, 3)
    gate = torch.ones(batch_size, 1, 1)
    adapter = ResidualActionAdapter(
        ResidualActionSpec(
            base_action_space="robotwin_endpose16",
            residual_mode="right_xyz_left_ee_frame",
            residual_frame="left_ee",
            base_action_dim=16,
            residual_chunk_len=chunk_len,
            env_action_chunk_len=chunk_len,
        )
    )

    with pytest.raises(ValueError, match="left_ee frame requires"):
        adapter.compose_action(base, residual, gate, obs=None)
    with pytest.raises(ValueError, match="left_ee frame requires"):
        adapter.compose_action(base, residual, gate, obs={})


def test_residual_chunk_len_greater_than_env_action_chunk_len_raises():
    with pytest.raises(ValueError, match="residual_chunk_len must be <= env_action_chunk_len"):
        ResidualActionAdapter(
            ResidualActionSpec(
                base_action_space="robotwin_endpose16",
                residual_mode="right_xyz_delta",
                base_action_dim=16,
                residual_chunk_len=6,
                env_action_chunk_len=5,
            )
        )


def test_compose_clamps_float_delta_max():
    base = torch.zeros(1, 2, 16)
    residual = torch.full((1, 2, 3), 10.0)
    gate = torch.ones(1, 1)
    adapter = ResidualActionAdapter(
        ResidualActionSpec(
            base_action_space="robotwin_endpose16",
            residual_mode="right_xyz_delta",
            base_action_dim=16,
            delta_max=0.25,
            clamp_residual=True,
        )
    )

    executed = adapter.compose_action(base, residual, gate)

    assert torch.allclose(executed[..., 8:11], torch.full((1, 2, 3), 0.25))
