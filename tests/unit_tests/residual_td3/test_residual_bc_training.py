import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.rollout_pi05_gate_controlled_hybrid_zero_residual import (
    resolve_residual_control_flags,
)
from scripts.train_residual_actor_bc import baseline_comparison, train_residual_actor_bc
from rlinf.algorithms.residual_td3.residual_actor import (
    ResidualActorConfig,
    ZeroInitResidualActorMLP,
)
from rlinf.algorithms.residual_td3.residual_bc_training import (
    ResidualBCDatasetConfig,
    ResidualBCNpzDataset,
    build_residual_actor_obs_features_from_endpose,
    residual_actor_obs_features_to_tensor,
    split_by_episode,
)


def _write_npz(path: Path, *, num_samples: int = 8, horizon: int = 3, zero_target: bool = False):
    rng = np.random.default_rng(0)
    episodes = np.repeat(np.arange(max(2, num_samples // 2)), 2)[:num_samples]
    target = np.zeros((num_samples, horizon, 3), dtype=np.float32)
    if not zero_target:
        target = rng.normal(0.0, 0.002, size=(num_samples, horizon, 3)).astype(np.float32)
    arrays = {
        "target_delta_local_xyz": target,
        "episode_index": episodes.astype(np.int64),
        "frame_index": np.arange(num_samples, dtype=np.int64),
        "gate_seq": np.ones((num_samples, 50), dtype=np.float32),
        "obs_base_left_xyz": rng.normal(size=(num_samples, horizon, 3)).astype(np.float32),
        "obs_base_left_quat_wxyz": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (num_samples, horizon, 1),
        ),
        "obs_base_right_xyz": rng.normal(size=(num_samples, horizon, 3)).astype(np.float32),
        "obs_base_right_quat_wxyz": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (num_samples, horizon, 1),
        ),
        "obs_relative_right_xyz_in_left_frame": rng.normal(size=(num_samples, horizon, 3)).astype(
            np.float32
        ),
        "obs_left_gripper": rng.normal(size=(num_samples, horizon)).astype(np.float32),
        "obs_right_gripper": rng.normal(size=(num_samples, horizon)).astype(np.float32),
        "obs_gate_label": np.ones((num_samples, horizon), dtype=np.float32),
        "base_endpose16": np.zeros((num_samples, 10, 16), dtype=np.float32),
        "expert_endpose16": np.zeros((num_samples, 10, 16), dtype=np.float32),
    }
    arrays["base_endpose16"][:, :, 3] = 1.0
    arrays["base_endpose16"][:, :, 11] = 1.0
    arrays["expert_endpose16"][:] = arrays["base_endpose16"]
    arrays["expert_endpose16"][:, :, 8] += 0.001
    np.savez_compressed(path, **arrays)
    return arrays


def test_dataset_reads_target_npz(tmp_path):
    path = tmp_path / "targets.npz"
    arrays = _write_npz(path)

    dataset = ResidualBCNpzDataset(path, ResidualBCDatasetConfig(target_horizon_k=3))
    item = dataset[0]

    assert len(dataset) == arrays["target_delta_local_xyz"].shape[0]
    assert dataset.obs_dim == 21
    assert item["obs_vector"].shape == (21,)
    assert item["target_delta_local_xyz"].shape == (3, 3)
    assert item["episode_index"] == 0


def test_zero_predictor_baseline_metrics_are_computed_on_same_targets():
    target = np.asarray(
        [
            [[0.001, 0.0, 0.0], [0.0, 0.002, 0.0]],
            [[0.0, 0.0, 0.003], [0.004, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    pred = target.copy()

    comparison = baseline_comparison(pred, target, delta_max=0.05)

    assert comparison["trained_bc_actor"]["mse"] == 0.0
    assert comparison["zero_predictor"]["mse"] == pytest.approx(float(np.mean(target**2)))
    assert comparison["zero_predictor"]["error_norm_mean"] == pytest.approx(
        float(np.linalg.norm(target, axis=-1).mean())
    )
    assert comparison["improvement"]["mse_improvement_pct"] == pytest.approx(100.0)


def test_online_endpose_feature_builder_matches_dataset_feature_order(tmp_path):
    path = tmp_path / "targets.npz"
    arrays = _write_npz(path, num_samples=4, horizon=3)
    dataset = ResidualBCNpzDataset(path, ResidualBCDatasetConfig(target_horizon_k=3))
    base = torch.from_numpy(arrays["base_endpose16"][0, 0])
    base[0:3] = torch.from_numpy(arrays["obs_base_left_xyz"][0, 0])
    base[3:7] = torch.from_numpy(arrays["obs_base_left_quat_wxyz"][0, 0])
    base[7] = torch.as_tensor(arrays["obs_left_gripper"][0, 0])
    base[8:11] = torch.from_numpy(arrays["obs_base_right_xyz"][0, 0])
    base[11:15] = torch.from_numpy(arrays["obs_base_right_quat_wxyz"][0, 0])
    base[15] = torch.as_tensor(arrays["obs_right_gripper"][0, 0])

    features = build_residual_actor_obs_features_from_endpose(
        base,
        gate_label=float(arrays["obs_gate_label"][0, 0]),
        gate_positive_fraction=float((arrays["gate_seq"][0] > 0).mean()),
    )
    vector = residual_actor_obs_features_to_tensor(features).numpy()

    expected = dataset.obs_vectors[0].copy()
    expected[14:17] = vector[14:17]
    np.testing.assert_allclose(vector, expected, atol=1e-6)


def test_residual_actor_input_output_shape_and_mse_backprop(tmp_path):
    path = tmp_path / "targets.npz"
    _write_npz(path)
    dataset = ResidualBCNpzDataset(path, ResidualBCDatasetConfig(target_horizon_k=3))
    actor = ZeroInitResidualActorMLP(
        ResidualActorConfig(obs_dim=dataset.obs_dim, hidden_dim=16, chunk_len=3, zero_init_output=False)
    )
    item = dataset[0]
    obs = item["obs_vector"].unsqueeze(0)
    target = item["target_delta_local_xyz"].unsqueeze(0)

    pred = actor(obs)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    assert pred.shape == (1, 3, 3)
    assert actor.net[0].weight.grad is not None


def test_zero_target_zero_init_actor_outputs_zero(tmp_path):
    path = tmp_path / "zero_targets.npz"
    _write_npz(path, zero_target=True)
    dataset = ResidualBCNpzDataset(path, ResidualBCDatasetConfig(target_horizon_k=3))
    actor = ZeroInitResidualActorMLP(
        ResidualActorConfig(obs_dim=dataset.obs_dim, hidden_dim=16, chunk_len=3, delta_max=0.05)
    )
    obs = torch.from_numpy(dataset.obs_vectors[:4])
    target = torch.from_numpy(dataset.arrays["target_delta_local_xyz"][:4])

    pred = actor(obs)
    loss = torch.nn.functional.mse_loss(pred, target)

    assert float(loss) == 0.0
    assert torch.count_nonzero(pred) == 0


def test_dataset_can_rebuild_longer_targets_from_endpose(tmp_path):
    path = tmp_path / "targets.npz"
    _write_npz(path, horizon=3)

    dataset = ResidualBCNpzDataset(path, ResidualBCDatasetConfig(target_horizon_k=10))

    assert dataset.target_horizon_k == 10
    np.testing.assert_allclose(
        dataset.arrays["target_delta_local_xyz"][:, :, 0],
        np.full((len(dataset), 10), 0.001, dtype=np.float32),
        atol=1e-7,
    )


def test_train_val_episode_split_has_no_leakage():
    episode_index = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    train_indices, val_indices = split_by_episode(episode_index, val_ratio=0.25, seed=3)

    train_eps = set(episode_index[train_indices].tolist())
    val_eps = set(episode_index[val_indices].tolist())

    assert train_eps
    assert val_eps
    assert train_eps.isdisjoint(val_eps)


def test_checkpoint_save_load_predictions_match(tmp_path):
    path = tmp_path / "targets.npz"
    _write_npz(path, num_samples=12, horizon=2)
    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        target_dataset=str(path),
        output_dir=str(out_dir),
        residual_horizon_k=2,
        delta_max=0.05,
        hidden_dim=16,
        batch_size=4,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        val_ratio=0.25,
        seed=7,
        device="cpu",
        sample_limit=None,
        include_current_qpos14=False,
        checkpoint=None,
        eval_only=False,
        compute_zero_baseline=True,
    )

    metrics = train_residual_actor_bc(args)
    checkpoint = torch.load(metrics["checkpoint_path"], map_location="cpu")
    actor_a = ZeroInitResidualActorMLP(ResidualActorConfig(**checkpoint["actor_config"]))
    actor_b = ZeroInitResidualActorMLP(ResidualActorConfig(**checkpoint["actor_config"]))
    actor_a.load_state_dict(checkpoint["model_state_dict"])
    actor_b.load_state_dict(checkpoint["model_state_dict"])
    obs = torch.randn(3, checkpoint["actor_config"]["obs_dim"])

    np.testing.assert_allclose(
        actor_a(obs).detach().numpy(),
        actor_b(obs).detach().numpy(),
        atol=0.0,
    )
    assert (out_dir / "baseline_comparison.json").exists()


def test_eval_only_checkpoint_loads_and_writes_baseline_comparison(tmp_path):
    path = tmp_path / "targets.npz"
    _write_npz(path, num_samples=12, horizon=2)
    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        target_dataset=str(path),
        output_dir=str(out_dir),
        residual_horizon_k=2,
        delta_max=0.05,
        hidden_dim=16,
        batch_size=4,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        val_ratio=0.25,
        seed=7,
        device="cpu",
        sample_limit=None,
        include_current_qpos14=False,
        checkpoint=None,
        eval_only=False,
        compute_zero_baseline=False,
    )
    train_metrics = train_residual_actor_bc(args)
    args.eval_only = True
    args.checkpoint = train_metrics["checkpoint_path"]
    args.compute_zero_baseline = True

    eval_metrics = train_residual_actor_bc(args)

    assert eval_metrics["checkpoint_path"] == train_metrics["checkpoint_path"]
    assert "zero_predictor" in eval_metrics["baseline_comparison"]
    assert eval_metrics["split_source"] == "checkpoint_episode_split"
    assert (out_dir / "baseline_comparison.json").exists()


def test_eval_only_reuses_checkpoint_episode_split_even_if_seed_changes(tmp_path):
    path = tmp_path / "targets.npz"
    _write_npz(path, num_samples=16, horizon=2)
    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        target_dataset=str(path),
        output_dir=str(out_dir),
        residual_horizon_k=2,
        delta_max=0.05,
        hidden_dim=16,
        batch_size=4,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        val_ratio=0.25,
        seed=7,
        device="cpu",
        sample_limit=None,
        include_current_qpos14=False,
        checkpoint=None,
        eval_only=False,
        compute_zero_baseline=False,
    )
    train_metrics = train_residual_actor_bc(args)
    train_val_episodes = train_metrics["val_episodes"]
    args.eval_only = True
    args.checkpoint = train_metrics["checkpoint_path"]
    args.seed = 999
    args.val_ratio = 0.49

    eval_metrics = train_residual_actor_bc(args)

    assert eval_metrics["val_episodes"] == train_val_episodes
    assert eval_metrics["split_source"] == "checkpoint_episode_split"


def test_residual_control_flags_require_explicit_enable():
    args = argparse.Namespace(residual_dry_run=False, enable_learned_residual_control=False)
    assert resolve_residual_control_flags(args) == {
        "dry_run": False,
        "learned_residual_control_enabled": False,
    }

    args = argparse.Namespace(residual_dry_run=True, enable_learned_residual_control=False)
    assert resolve_residual_control_flags(args) == {
        "dry_run": True,
        "learned_residual_control_enabled": False,
    }

    args = argparse.Namespace(residual_dry_run=False, enable_learned_residual_control=True)
    assert resolve_residual_control_flags(args) == {
        "dry_run": False,
        "learned_residual_control_enabled": True,
    }

    with pytest.raises(ValueError, match="cannot be enabled together"):
        resolve_residual_control_flags(
            argparse.Namespace(residual_dry_run=True, enable_learned_residual_control=True)
        )
