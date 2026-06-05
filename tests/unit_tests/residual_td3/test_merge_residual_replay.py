import numpy as np

from scripts.merge_residual_replay import load_replay, merge_replays, replay_summary


def make_replay(n: int, source: str) -> dict[str, np.ndarray]:
    return {
        "obs": np.zeros((n, 21), dtype=np.float32),
        "action": np.full((n, 3), 0.001, dtype=np.float32),
        "reward": np.linspace(0.0, 1.0, n, dtype=np.float32),
        "next_obs": np.ones((n, 21), dtype=np.float32),
        "done": np.asarray([False] * (n - 1) + [True], dtype=np.bool_),
        "action_source": np.asarray([source] * n, dtype="U32"),
        "seed": np.arange(n, dtype=np.int64),
        "episode_id": np.zeros(n, dtype=np.int64),
        "success": np.ones(n, dtype=np.bool_),
        "failure_reason": np.asarray([""] * n, dtype="U64"),
        "saturation": np.zeros(n, dtype=np.int64),
        "has_nan_or_inf": np.zeros(n, dtype=np.int64),
    }


def test_merge_replay_preserves_shapes_and_metadata(tmp_path):
    first = make_replay(3, "bc_s05")
    second = make_replay(2, "zero")
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    np.savez_compressed(first_path, **first)
    np.savez_compressed(second_path, **second)

    first_loaded = load_replay(first_path, delta_max=0.05)
    second_loaded = load_replay(second_path, delta_max=0.05)
    merged = merge_replays([first_loaded["arrays"], second_loaded["arrays"]])
    summary = replay_summary(
        merged,
        source_summaries=[first_loaded["summary"], second_loaded["summary"]],
        delta_max=0.05,
    )

    assert merged["obs"].shape == (5, 21)
    assert merged["action"].shape == (5, 3)
    assert set(merged["action_source"].tolist()) == {"bc_s05", "zero"}
    assert summary["merged"]["transitions"] == 5
    assert summary["merged"]["episodes"] == 5
