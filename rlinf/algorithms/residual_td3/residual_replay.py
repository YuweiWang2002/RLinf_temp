"""Residual replay utilities for online intervention logs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReplayRewardConfig:
    """Reward assignment for collected residual replay."""

    mode: str = "sparse_success"
    discount_to_steps: str = "all"
    baseline_summary_path: str | None = None


def load_rollout_summary(path: str | Path) -> dict[str, Any]:
    """Load a rollout summary from a summary path or rollout directory."""

    summary_path = resolve_summary_path(Path(path))
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_summary_path(path: Path) -> Path:
    """Resolve ``path`` to a ``summary.json`` file."""

    return path if path.name == "summary.json" else path / "summary.json"


def load_baseline_success_by_seed(path: str | Path | None) -> dict[int, bool]:
    """Return ``seed -> success`` from a baseline rollout summary."""

    if path is None:
        raise ValueError("--baseline-summary is required for reward-mode=baseline_comparison.")
    summary_path = resolve_summary_path(Path(path))
    if not summary_path.exists():
        raise FileNotFoundError(f"baseline summary not found: {summary_path}")
    summary = load_rollout_summary(summary_path)
    return {int(ep["seed"]): bool(ep.get("success", False)) for ep in summary.get("episodes", [])}


def episode_reward(
    *,
    mode: str,
    residual_success: bool,
    seed: int,
    baseline_success_by_seed: dict[int, bool] | None = None,
) -> float:
    """Compute the episode-level residual replay reward."""

    if mode == "sparse_success":
        return 1.0 if residual_success else 0.0
    if mode != "baseline_comparison":
        raise ValueError(f"unknown reward mode: {mode}")
    if baseline_success_by_seed is None:
        raise ValueError("baseline_success_by_seed is required for baseline_comparison.")
    baseline_success = bool(baseline_success_by_seed.get(seed, False))
    if residual_success and not baseline_success:
        return 10.0
    if residual_success and baseline_success:
        return 1.0
    if not residual_success and baseline_success:
        return -10.0
    return 0.0


def load_intervention_records(path: str | Path) -> list[dict[str, Any]]:
    """Load intervention records from parquet."""

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()


def build_replay_from_rollout(
    summary: dict[str, Any],
    records_by_episode: dict[int, list[dict[str, Any]]],
    *,
    action_source: str,
    reward_config: ReplayRewardConfig,
    baseline_success_by_seed: dict[int, bool] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Build one-step residual replay transitions from rollout intervention records."""

    if reward_config.mode == "baseline_comparison" and baseline_success_by_seed is None:
        baseline_success_by_seed = load_baseline_success_by_seed(reward_config.baseline_summary_path)
    if reward_config.discount_to_steps not in ("all", "last"):
        raise ValueError("reward_discount_to_steps must be all or last.")

    transitions: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for episode in summary.get("episodes", []):
        episode_id = int(episode["episode_id"])
        seed = int(episode["seed"])
        records = sorted(
            records_by_episode.get(episode_id, []),
            key=lambda row: (int(row.get("intervention_id", 0)), int(row.get("intervention_step_i", 0))),
        )
        reward_value = episode_reward(
            mode=reward_config.mode,
            residual_success=bool(episode.get("success", False)),
            seed=seed,
            baseline_success_by_seed=baseline_success_by_seed,
        )
        episode_rows.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "success": int(bool(episode.get("success", False))),
                "failure_reason": episode.get("failure_reason") or "",
                "num_transitions": len(records),
                "episode_reward": reward_value,
            }
        )
        for idx, record in enumerate(records):
            is_last_episode_record = idx == len(records) - 1
            next_record = records[idx + 1] if not is_last_episode_record else record
            done_intervention = is_last_episode_record or record.get("intervention_id") != next_record.get(
                "intervention_id"
            )
            reward = reward_value if reward_config.discount_to_steps == "all" or is_last_episode_record else 0.0
            transition = build_transition(
                record,
                next_record,
                episode,
                action_source=action_source,
                reward=reward,
                done_episode=is_last_episode_record,
                done_intervention=bool(done_intervention),
            )
            transitions.append(transition)

    replay = transitions_to_arrays(transitions)
    return replay, episode_rows, replay_summary(replay, episode_rows, summary)


def build_transition(
    record: dict[str, Any],
    next_record: dict[str, Any],
    episode: dict[str, Any],
    *,
    action_source: str,
    reward: float,
    done_episode: bool,
    done_intervention: bool,
) -> dict[str, Any]:
    """Build one replay transition from adjacent intervention records."""

    done = bool(done_episode or done_intervention)
    return {
        "obs": vector(record, "obs_vector", 21),
        "action": vector(record, "applied_delta_local_xyz", 3),
        "reward": float(reward),
        "next_obs": vector(next_record, "obs_vector", 21),
        "done": done,
        "done_episode": bool(done_episode),
        "done_intervention": bool(done_intervention),
        "episode_id": int(episode["episode_id"]),
        "seed": int(episode["seed"]),
        "env_step": int(record.get("env_step") or 0),
        "intervention_id": int(record.get("intervention_id") or 0),
        "intervention_step_i": int(record.get("intervention_step_i") or 0),
        "gate_score": float(record.get("gate_score") or 0.0),
        "action_source": action_source,
        "residual_scale": float(record.get("residual_scale") or 0.0),
        "noise_std": float(record.get("noise_std") or 0.0),
        "pred_delta_local_xyz": vector(record, "pred_delta_local_xyz", 3),
        "applied_delta_local_xyz": vector(record, "applied_delta_local_xyz", 3),
        "applied_delta_world_xyz": vector(record, "applied_delta_world_xyz", 3),
        "base_ee16": vector(record, "base_ee16", 16),
        "exec_ee16": vector(record, "exec_ee16", 16),
        "success": bool(episode.get("success", False)),
        "failure_reason": episode.get("failure_reason") or "",
        "saturation": int(record.get("saturation") or 0),
        "has_nan_or_inf": int(record.get("has_nan_or_inf") or 0),
    }


def vector(record: dict[str, Any], key: str, size: int) -> np.ndarray:
    """Read a fixed-size float vector from a record."""

    value = np.asarray(record.get(key, np.zeros(size, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if value.shape != (size,):
        raise ValueError(f"{key} must have shape [{size}], got {value.shape}.")
    return value


def transitions_to_arrays(transitions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Convert transition dicts to npz-friendly arrays."""

    fields = {
        "obs": (np.float32, (21,)),
        "action": (np.float32, (3,)),
        "reward": (np.float32, ()),
        "next_obs": (np.float32, (21,)),
        "done": (np.bool_, ()),
        "done_episode": (np.bool_, ()),
        "done_intervention": (np.bool_, ()),
        "episode_id": (np.int64, ()),
        "seed": (np.int64, ()),
        "env_step": (np.int64, ()),
        "intervention_id": (np.int64, ()),
        "intervention_step_i": (np.int64, ()),
        "gate_score": (np.float32, ()),
        "residual_scale": (np.float32, ()),
        "noise_std": (np.float32, ()),
        "pred_delta_local_xyz": (np.float32, (3,)),
        "applied_delta_local_xyz": (np.float32, (3,)),
        "applied_delta_world_xyz": (np.float32, (3,)),
        "base_ee16": (np.float32, (16,)),
        "exec_ee16": (np.float32, (16,)),
        "success": (np.bool_, ()),
        "saturation": (np.int64, ()),
        "has_nan_or_inf": (np.int64, ()),
    }
    arrays: dict[str, np.ndarray] = {}
    for key, (dtype, shape) in fields.items():
        if not transitions:
            arrays[key] = np.empty((0, *shape), dtype=dtype)
        else:
            arrays[key] = np.asarray([row[key] for row in transitions], dtype=dtype)
    arrays["action_source"] = np.asarray([row["action_source"] for row in transitions], dtype="U32")
    arrays["failure_reason"] = np.asarray([row["failure_reason"] for row in transitions], dtype="U64")
    return arrays


def replay_summary(
    replay: dict[str, np.ndarray],
    episode_rows: list[dict[str, Any]],
    rollout_summary: dict[str, Any],
) -> dict[str, Any]:
    """Summarize replay arrays for smoke diagnostics."""

    action_norm = np.linalg.norm(replay["action"], axis=1) if replay["action"].size else np.asarray([])
    reward = replay["reward"]
    return {
        "num_episodes": len(episode_rows),
        "num_transitions": int(replay["obs"].shape[0]),
        "obs_shape": list(replay["obs"].shape),
        "action_shape": list(replay["action"].shape),
        "next_obs_shape": list(replay["next_obs"].shape),
        "reward_mean": float(reward.mean()) if reward.size else 0.0,
        "reward_std": float(reward.std()) if reward.size else 0.0,
        "reward_min": float(reward.min()) if reward.size else 0.0,
        "reward_max": float(reward.max()) if reward.size else 0.0,
        "action_norm_mean": float(action_norm.mean()) if action_norm.size else 0.0,
        "action_norm_max": float(action_norm.max()) if action_norm.size else 0.0,
        "action_norm_p95": float(np.percentile(action_norm, 95)) if action_norm.size else 0.0,
        "done_ratio": float(replay["done"].mean()) if replay["done"].size else 0.0,
        "success_rate": float(rollout_summary.get("success_rate", 0.0)),
        "nan_inf_count": int(replay["has_nan_or_inf"].sum()) if replay["has_nan_or_inf"].size else 0,
        "saturation_ratio": float(replay["saturation"].mean()) if replay["saturation"].size else 0.0,
    }


def save_replay_artifacts(
    out_dir: str | Path,
    replay: dict[str, np.ndarray],
    episode_rows: list[dict[str, Any]],
    reward_summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, str]:
    """Write replay npz, optional parquet, episode summary, config, and reward summary."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "replay.npz"
    np.savez_compressed(npz_path, **replay)
    parquet_path = out / "replay.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [
            {
                key: replay[key][idx].tolist() if hasattr(replay[key][idx], "tolist") else replay[key][idx]
                for key in replay
            }
            for idx in range(int(replay["obs"].shape[0]))
        ]
        pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), parquet_path)
    except Exception:  # noqa: BLE001
        parquet_path = Path("")
    episodes_path = out / "episodes_summary.csv"
    write_episode_summary_csv(episodes_path, episode_rows)
    config_path = out / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    reward_path = out / "reward_summary.json"
    reward_path.write_text(json.dumps(reward_summary, indent=2), encoding="utf-8")
    paths = {
        "npz_path": str(npz_path),
        "episodes_summary_path": str(episodes_path),
        "config_path": str(config_path),
        "reward_summary_path": str(reward_path),
    }
    if str(parquet_path):
        paths["parquet_path"] = str(parquet_path)
    return paths


def write_episode_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-episode replay summary rows."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
