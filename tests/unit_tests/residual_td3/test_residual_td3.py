import argparse

import numpy as np
import torch

from rlinf.algorithms.residual_td3.residual_actor import (
    ResidualActorConfig,
    ZeroInitResidualActorMLP,
)
from rlinf.algorithms.residual_td3.residual_td3 import (
    NpzReplayDataset,
    ResidualCritic,
    ResidualTD3Config,
    actor_action,
    clone_module,
    save_td3_checkpoints,
    td3_update,
)
from scripts.train_residual_td3_from_replay import train_residual_td3_from_replay


def make_replay_npz(path, n: int = 16) -> None:
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        obs=rng.normal(size=(n, 21)).astype(np.float32),
        action=np.clip(rng.normal(scale=0.001, size=(n, 3)), -0.05, 0.05).astype(np.float32),
        reward=rng.normal(size=n).astype(np.float32),
        next_obs=rng.normal(size=(n, 21)).astype(np.float32),
        done=np.zeros(n, dtype=np.bool_),
        action_source=np.asarray(["unit"] * n, dtype="U32"),
        seed=np.arange(n, dtype=np.int64),
        episode_id=np.zeros(n, dtype=np.int64),
        success=np.ones(n, dtype=np.bool_),
        failure_reason=np.asarray([""] * n, dtype="U64"),
    )


def make_actor_checkpoint(path) -> None:
    cfg = ResidualActorConfig(
        obs_dim=21,
        hidden_dim=32,
        chunk_len=4,
        residual_dim=3,
        delta_max=0.05,
        zero_init_output=False,
    )
    actor = ZeroInitResidualActorMLP(cfg)
    torch.save(
        {
            "model_state_dict": actor.state_dict(),
            "actor_config": cfg.__dict__,
            "feature_names": (),
        },
        path,
    )


def test_replay_dataset_sample_batch_shape(tmp_path):
    replay_path = tmp_path / "replay.npz"
    make_replay_npz(replay_path)
    dataset = NpzReplayDataset.load(replay_path)
    batch = dataset.sample(8, rng=np.random.default_rng(1), device="cpu")

    assert batch["obs"].shape == (8, 21)
    assert batch["action"].shape == (8, 3)
    assert batch["reward"].shape == (8, 1)


def test_critic_forward_outputs_column_vector():
    critic = ResidualCritic(obs_dim=21, action_dim=3, hidden_dim=16)

    q = critic(torch.zeros(5, 21), torch.zeros(5, 3))

    assert q.shape == (5, 1)


def test_td3_update_backpropagates_and_clips_actor_action(tmp_path):
    replay_path = tmp_path / "replay.npz"
    make_replay_npz(replay_path)
    dataset = NpzReplayDataset.load(replay_path)
    actor = ZeroInitResidualActorMLP(
        ResidualActorConfig(obs_dim=21, hidden_dim=16, chunk_len=2, delta_max=0.02)
    )
    actor_target = clone_module(actor)
    critic1 = ResidualCritic(21, 3, 16)
    critic2 = ResidualCritic(21, 3, 16)
    critic1_target = clone_module(critic1)
    critic2_target = clone_module(critic2)
    batch = dataset.sample(8, rng=np.random.default_rng(2), device="cpu")
    cfg = ResidualTD3Config(hidden_dim=16, delta_max=0.02, policy_delay=1)

    row = td3_update(
        actor=actor,
        actor_target=actor_target,
        critic1=critic1,
        critic2=critic2,
        critic1_target=critic1_target,
        critic2_target=critic2_target,
        actor_optimizer=torch.optim.AdamW(actor.parameters(), lr=1e-4),
        critic_optimizer=torch.optim.AdamW(
            list(critic1.parameters()) + list(critic2.parameters()),
            lr=1e-3,
        ),
        batch=batch,
        cfg=cfg,
        update_step=1,
    )

    with torch.no_grad():
        action = actor_action(actor, batch["obs"])
    assert np.isfinite(row["critic_loss"])
    assert action.abs().max() <= 0.020001


def test_checkpoint_save_load_and_training_smoke(tmp_path):
    replay_path = tmp_path / "replay.npz"
    bc_path = tmp_path / "bc.pt"
    out_dir = tmp_path / "td3"
    make_replay_npz(replay_path, n=32)
    make_actor_checkpoint(bc_path)

    summary = train_residual_td3_from_replay(
        argparse.Namespace(
            replay=str(replay_path),
            bc_actor_checkpoint=str(bc_path),
            output_dir=str(out_dir),
            obs_dim=21,
            action_dim=3,
            hidden_dim=16,
            delta_max=0.05,
            device="cpu",
            updates=2,
            batch_size=8,
            gamma=0.99,
            tau=0.005,
            policy_delay=1,
        actor_lr=1e-4,
        critic_lr=1e-3,
        target_noise=0.001,
        target_noise_clip=0.003,
        actor_delta_max=0.01,
        actor_bc_weight=10.0,
        actor_l2_weight=0.0,
        actor_bc_mode="all",
        awbc_temperature=1.0,
        q_filter_margin=0.0,
        seed=0,
        log_interval=100,
    )
    )

    actor_path = out_dir / "actor_td3.pt"
    critic_path = out_dir / "critic_td3.pt"
    assert actor_path.exists()
    assert critic_path.exists()
    assert summary["actor_smoke"]["finite"]
    checkpoint = torch.load(actor_path, map_location="cpu")
    actor = ZeroInitResidualActorMLP(ResidualActorConfig(**checkpoint["actor_config"]))
    actor.load_state_dict(checkpoint["model_state_dict"])
    assert actor(torch.zeros(1, 21)).shape == (1, 4, 3)
    assert checkpoint["actor_config"]["delta_max"] == 0.01


def test_save_td3_checkpoints_contains_eval_actor_format(tmp_path):
    actor = ZeroInitResidualActorMLP(ResidualActorConfig(obs_dim=21, hidden_dim=16))
    critic1 = ResidualCritic(21, 3, 16)
    critic2 = ResidualCritic(21, 3, 16)

    paths = save_td3_checkpoints(
        tmp_path,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        cfg=ResidualTD3Config(hidden_dim=16),
        metrics={},
    )

    checkpoint = torch.load(paths["actor_path"], map_location="cpu")
    assert checkpoint["algorithm"] == "offline_td3"
    assert "actor_config" in checkpoint
    assert "model_state_dict" in checkpoint
