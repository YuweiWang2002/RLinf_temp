from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.algorithms.residual_td3.residual_ee_intervention import (
    RandomNoiseResidualActor,
)
from rlinf.algorithms.residual_td3.residual_replay import (
    ReplayRewardConfig,
    build_replay_from_rollout,
    episode_reward,
    load_baseline_success_by_seed,
    save_replay_artifacts,
)


def _record(step: int, action=(0.0, 0.0, 0.0), *, intervention_id: int = 0):
    base_ee16 = np.zeros(16, dtype=np.float32)
    exec_ee16 = base_ee16.copy()
    exec_ee16[8:11] = np.asarray(action, dtype=np.float32)
    return {
        "obs_vector": (np.arange(21, dtype=np.float32) + step).tolist(),
        "pred_delta_local_xyz": list(action),
        "applied_delta_local_xyz": list(action),
        "applied_delta_world_xyz": list(action),
        "base_ee16": base_ee16.tolist(),
        "exec_ee16": exec_ee16.tolist(),
        "env_step": step,
        "intervention_id": intervention_id,
        "intervention_step_i": step,
        "gate_score": 0.8,
        "residual_scale": 0.25,
        "noise_std": 0.0,
        "saturation": 0,
        "has_nan_or_inf": 0,
    }


def _summary(success=True):
    return {
        "success_rate": 1.0 if success else 0.0,
        "episodes": [
            {
                "episode_id": 0,
                "seed": 100100000,
                "success": success,
                "failure_reason": None if success else "timeout",
            }
        ],
    }


def test_sparse_and_baseline_comparison_rewards():
    assert episode_reward(mode="sparse_success", residual_success=True, seed=1) == 1.0
    assert episode_reward(mode="sparse_success", residual_success=False, seed=1) == 0.0
    assert (
        episode_reward(
            mode="baseline_comparison",
            residual_success=True,
            seed=1,
            baseline_success_by_seed={1: False},
        )
        == 10.0
    )
    assert (
        episode_reward(
            mode="baseline_comparison",
            residual_success=False,
            seed=1,
            baseline_success_by_seed={1: True},
        )
        == -10.0
    )


def test_baseline_summary_missing_raises():
    with pytest.raises(ValueError, match="baseline-summary"):
        load_baseline_success_by_seed(None)


def test_build_replay_shapes_and_zero_action():
    replay, episodes, reward_summary = build_replay_from_rollout(
        _summary(success=True),
        {0: [_record(0), _record(1)]},
        action_source="zero",
        reward_config=ReplayRewardConfig(mode="sparse_success"),
    )

    assert replay["obs"].shape == (2, 21)
    assert replay["action"].shape == (2, 3)
    assert replay["next_obs"].shape == (2, 21)
    np.testing.assert_allclose(replay["action"], np.zeros((2, 3)))
    assert episodes[0]["num_transitions"] == 2
    assert reward_summary["num_transitions"] == 2
    assert reward_summary["reward_mean"] == pytest.approx(1.0)


def test_bc_action_source_keeps_scaled_applied_delta():
    replay, _, reward_summary = build_replay_from_rollout(
        _summary(success=True),
        {0: [_record(0, action=(0.0025, 0.0, 0.0))]},
        action_source="bc",
        reward_config=ReplayRewardConfig(mode="sparse_success"),
    )

    np.testing.assert_allclose(replay["action"][0], np.asarray([0.0025, 0.0, 0.0], dtype=np.float32))
    assert replay["residual_scale"][0] == pytest.approx(0.25)
    assert reward_summary["action_norm_max"] == pytest.approx(0.0025)


def test_random_noise_actor_is_clipped():
    actor = RandomNoiseResidualActor(noise_std=10.0, max_delta_local_xyz=0.002, seed=0)
    delta = actor.predict_delta_local_xyz(None)

    assert torch.all(torch.abs(delta) <= 0.002)


def test_replay_npz_save_load(tmp_path):
    replay, episodes, reward_summary = build_replay_from_rollout(
        _summary(success=True),
        {0: [_record(0, action=(0.001, 0.0, 0.0))]},
        action_source="bc",
        reward_config=ReplayRewardConfig(mode="sparse_success"),
    )

    paths = save_replay_artifacts(tmp_path, replay, episodes, reward_summary, {"action_source": "bc"})
    loaded = np.load(paths["npz_path"])

    assert loaded["obs"].shape == (1, 21)
    assert loaded["action"].shape == (1, 3)
    assert Path(paths["reward_summary_path"]).exists()


def test_collect_residual_replay_contains_no_optimizer_update():
    source = Path("scripts/collect_residual_replay.py").read_text(encoding="utf-8").lower()

    assert "optimizer" not in source
    assert ".backward(" not in source
    assert "critic" not in source
