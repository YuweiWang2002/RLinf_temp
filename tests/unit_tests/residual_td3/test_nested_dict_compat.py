import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.utils.nested_dict_process import split_dict_to_chunk


def _residual_obs(batch_size: int, chunk_len: int, base_dim: int, residual_dim: int):
    return {
        "states": torch.zeros(batch_size, 4),
        "residual_td3": {
            "vla_feature": torch.zeros(batch_size, 256),
            "rel_state": torch.zeros(batch_size, 6),
            "base_action_chunk": torch.zeros(batch_size, chunk_len, base_dim),
            "residual_ref_chunk": torch.zeros(batch_size, chunk_len, residual_dim),
            "gate": torch.ones(batch_size, 1),
        },
    }


def _robotwin_residual_obs(batch_size: int, chunk_len: int):
    obs = _residual_obs(batch_size, chunk_len, base_dim=16, residual_dim=3)
    obs["residual_td3"]["left_ee_rot"] = torch.eye(3).repeat(batch_size, 1, 1)
    return obs


def _forward_inputs(
    batch_size: int,
    chunk_len: int,
    base_dim: int,
    residual_dim: int,
):
    return {
        "action": torch.zeros(batch_size, chunk_len * base_dim),
        "residual_td3": {
            "base_action_chunk": torch.zeros(batch_size, chunk_len, base_dim),
            "residual_action_chunk": torch.zeros(batch_size, chunk_len, residual_dim),
            "executed_action_chunk": torch.zeros(batch_size, chunk_len, base_dim),
        },
    }


def _rollout_with_residual_nested_dict(
    *,
    base_dim: int,
    residual_dim: int,
    obs_factory,
    batch_size: int = 4,
    chunk_len: int = 5,
    steps: int = 3,
) -> EmbodiedRolloutResult:
    rollout = EmbodiedRolloutResult(max_episode_length=steps)
    for _ in range(steps):
        rollout.append_step_result(
            ChunkStepResult(
                actions=torch.zeros(batch_size, base_dim),
                rewards=torch.zeros(batch_size, 1),
                dones=torch.zeros(batch_size, 1, dtype=torch.bool),
                terminations=torch.zeros(batch_size, 1, dtype=torch.bool),
                truncations=torch.zeros(batch_size, 1, dtype=torch.bool),
                prev_logprobs=torch.zeros(batch_size, base_dim),
                prev_values=torch.zeros(batch_size, 1),
                versions=torch.zeros(batch_size, 1),
                forward_inputs=_forward_inputs(
                    batch_size, chunk_len, base_dim, residual_dim
                ),
            )
        )
        rollout.append_transitions(
            curr_obs=obs_factory(batch_size, chunk_len),
            next_obs=obs_factory(batch_size, chunk_len),
        )
    return rollout


def _assert_residual_shapes(
    batch,
    *,
    batch_size: int,
    chunk_len: int,
    base_dim: int,
    residual_dim: int,
):
    curr = batch["curr_obs"]["residual_td3"]
    next_obs = batch["next_obs"]["residual_td3"]
    forward_inputs = batch["forward_inputs"]["residual_td3"]
    assert curr["base_action_chunk"].shape == (batch_size, chunk_len, base_dim)
    assert curr["residual_ref_chunk"].shape == (batch_size, chunk_len, residual_dim)
    assert curr["gate"].shape == (batch_size, 1)
    assert next_obs["base_action_chunk"].shape == (batch_size, chunk_len, base_dim)
    assert forward_inputs["base_action_chunk"].shape == (
        batch_size,
        chunk_len,
        base_dim,
    )
    assert forward_inputs["residual_action_chunk"].shape == (
        batch_size,
        chunk_len,
        residual_dim,
    )
    assert forward_inputs["executed_action_chunk"].shape == (
        batch_size,
        chunk_len,
        base_dim,
    )


def test_aloha_residual_nested_dict_survives_stack_split_and_replay_sample():
    rollout = _rollout_with_residual_nested_dict(
        base_dim=14,
        residual_dim=7,
        obs_factory=lambda batch_size, chunk_len: _residual_obs(
            batch_size, chunk_len, base_dim=14, residual_dim=7
        ),
    )
    trajectory = rollout.to_trajectory()

    assert "residual_td3" not in Trajectory.__dataclass_fields__
    assert trajectory.curr_obs["residual_td3"]["base_action_chunk"].shape == (
        3,
        4,
        5,
        14,
    )

    split = split_dict_to_chunk(trajectory.curr_obs, split_size=2, dim=1)
    assert split[0]["residual_td3"]["base_action_chunk"].shape == (3, 2, 5, 14)

    split_trajectories = rollout.to_splited_trajectories(split_size=2)
    assert split_trajectories[0].curr_obs["residual_td3"][
        "base_action_chunk"
    ].shape == (3, 2, 5, 14)

    batch = convert_trajectories_to_batch([trajectory])
    assert batch["curr_obs"]["residual_td3"]["base_action_chunk"].shape == (
        3,
        4,
        5,
        14,
    )

    replay = TrajectoryReplayBuffer(seed=0, auto_save=False, enable_cache=True)
    replay.add_trajectories([trajectory])
    sample = replay.sample(num_chunks=6)

    _assert_residual_shapes(
        sample,
        batch_size=6,
        chunk_len=5,
        base_dim=14,
        residual_dim=7,
    )


def test_robotwin_residual_nested_dict_survives_stack_split_and_replay_sample():
    rollout = _rollout_with_residual_nested_dict(
        base_dim=16,
        residual_dim=3,
        obs_factory=_robotwin_residual_obs,
    )
    trajectory = rollout.to_trajectory()

    split_trajectories = rollout.to_splited_trajectories(split_size=2)
    assert split_trajectories[0].curr_obs["residual_td3"]["left_ee_rot"].shape == (
        3,
        2,
        3,
        3,
    )
    chunks = split_dict_to_chunk(trajectory.forward_inputs, split_size=2, dim=1)
    assert chunks[0]["residual_td3"]["residual_action_chunk"].shape == (3, 2, 5, 3)

    replay = TrajectoryReplayBuffer(seed=0, auto_save=False, enable_cache=True)
    replay.add_trajectories([trajectory])
    sample = replay.sample(num_chunks=6)

    _assert_residual_shapes(
        sample,
        batch_size=6,
        chunk_len=5,
        base_dim=16,
        residual_dim=3,
    )
    assert sample["curr_obs"]["residual_td3"]["left_ee_rot"].shape == (6, 3, 3)
