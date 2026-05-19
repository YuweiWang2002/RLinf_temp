import torch

from rlinf.algorithms.residual_td3.action_adapter import (
    ResidualActionAdapter,
    ResidualActionSpec,
)


def test_robotwin_bc_target_world_frame_is_expert_minus_base_right_xyz():
    batch_size, chunk_len = 2, 5
    base = torch.zeros(batch_size, chunk_len, 16)
    expert = torch.zeros(batch_size, chunk_len, 16)
    expert[..., 8:11] = torch.tensor([1.0, 2.0, 3.0])
    adapter = ResidualActionAdapter(
        ResidualActionSpec(
            base_action_space="robotwin_endpose16",
            residual_mode="right_xyz_world_frame",
            residual_frame="world",
            base_action_dim=16,
            residual_chunk_len=chunk_len,
            env_action_chunk_len=chunk_len,
        )
    )

    target = adapter.compute_bc_target(base, expert)

    assert target.shape == (batch_size, chunk_len, 3)
    assert torch.allclose(target, expert[..., 8:11] - base[..., 8:11])


def test_robotwin_bc_target_left_ee_identity_matches_world_delta():
    batch_size, chunk_len = 2, 5
    base = torch.zeros(batch_size, chunk_len, 16)
    expert = torch.zeros(batch_size, chunk_len, 16)
    expert[..., 8:11] = torch.tensor([0.5, -0.25, 1.5])
    left_ee_rot = torch.eye(3).repeat(batch_size, chunk_len, 1, 1)
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

    target = adapter.compute_bc_target(base, expert, obs={"left_ee_rot": left_ee_rot})

    assert target.shape == (batch_size, chunk_len, 3)
    assert torch.allclose(target, expert[..., 8:11] - base[..., 8:11])
