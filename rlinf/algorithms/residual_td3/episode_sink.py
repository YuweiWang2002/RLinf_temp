"""Episode event schemas and lightweight sinks for residual TD3 rollouts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class StepEvent:
    """One replan/step-level decision event emitted by a rollout loop."""

    episode_id: int
    seed: int
    env_step: int
    action_mode: str
    execution_stage: str
    trigger_source: str
    gate_score: float
    selected_action_chunk_indices: list[int] = field(default_factory=list)
    has_intervention: bool = False
    selected_action_chunk_index: int | None = None
    obs_features: list[float] | None = None
    action: list[float] | None = None
    reward: float | None = None
    done: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV friendly dict."""

        return asdict(self)

    def to_hybrid_row(self) -> dict[str, Any]:
        """Return a row compatible with current hybrid log conventions."""

        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "env_step": self.env_step,
            "env_step_start": self.metadata.get("env_step_start", self.env_step),
            "env_step_end": self.metadata.get("env_step_end", self.env_step),
            "replan_id": self.metadata.get("replan_id", 0),
            "gate_logit": self.metadata.get("gate_logit", 0.0),
            "gate_prob": self.gate_score,
            "gate_binary": int(self.execution_stage == "handover"),
            "handover_gate_binary": int(self.execution_stage == "handover"),
            "pregrasp_trigger_binary": int(self.execution_stage == "pregrasp"),
            "pregrasp_trigger_score": self.metadata.get("pregrasp_trigger_score", 0.0),
            "execution_mode": self.action_mode,
            "intervention_stage": self.execution_stage if self.has_intervention else "none",
            "trigger_source": self.trigger_source,
            "num_interventions": int(self.has_intervention),
            "num_intervention_steps": len(self.selected_action_chunk_indices),
            "selected_indices": self.selected_action_chunk_indices if self.has_intervention else None,
            "obs_state_06": self.metadata.get("obs_state_06", 0.0),
            "obs_state_13": self.metadata.get("obs_state_13", 0.0),
            "d_LR": self.metadata.get("d_LR"),
            "success": int(bool(self.metadata.get("success", False))),
            "done": int(bool(self.done)),
        }


@dataclass(frozen=True)
class InterventionEvent:
    """One residual intervention step event."""

    episode_id: int
    seed: int
    env_step: int
    intervention_id: int
    intervention_step_i: int
    stage: str
    trigger_source: str
    action_source: str
    pred_delta_local_xyz: list[float]
    applied_delta_local_xyz: list[float]
    applied_delta_world_xyz: list[float]
    base_ee16: list[float]
    exec_ee16: list[float]
    saturation: int = 0
    has_nan_or_inf: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV friendly dict."""

        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        """Return a record compatible with current replay conversion fields."""

        pred = np.asarray(self.pred_delta_local_xyz, dtype=np.float32)
        applied = np.asarray(self.applied_delta_local_xyz, dtype=np.float32)
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "env_step": self.env_step,
            "intervention_id": self.intervention_id,
            "intervention_step_i": self.intervention_step_i,
            "intervention_stage": self.stage,
            "trigger_source": self.trigger_source,
            "action_source": self.action_source,
            "gate_score": float(self.metadata.get("gate_score", 0.0)),
            "obs_vector": self.metadata.get("obs_vector", [0.0] * 21),
            "pred_delta_local_xyz": self.pred_delta_local_xyz,
            "applied_delta_local_xyz": self.applied_delta_local_xyz,
            "applied_delta_world_xyz": self.applied_delta_world_xyz,
            "pred_delta_norm": float(np.linalg.norm(pred)),
            "applied_delta_norm": float(np.linalg.norm(applied)),
            "base_ee16": self.base_ee16,
            "exec_ee16": self.exec_ee16,
            "residual_scale": float(self.metadata.get("residual_scale", 1.0)),
            "noise_std": float(self.metadata.get("noise_std", 0.0)),
            "saturation": int(self.saturation),
            "has_nan_or_inf": int(self.has_nan_or_inf),
        }


@dataclass(frozen=True)
class EpisodeSummary:
    """Episode-level rollout summary emitted at episode end."""

    episode_id: int
    seed: int
    success: bool
    failure_reason: str | None
    episode_len: int
    num_interventions: int
    total_ee_steps: int
    num_pregrasp_interventions: int
    num_handover_interventions: int
    first_pregrasp_trigger_step: int | None
    first_handover_trigger_step: int | None
    mock_runtime: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV friendly dict."""

        payload = asdict(self)
        payload["episode_length"] = payload.pop("episode_len")
        payload["total_ee_intervention_steps"] = payload.pop("total_ee_steps")
        return payload


class EpisodeSink(Protocol):
    """Interface for rollout artifact sinks."""

    def on_episode_start(self, episode_id: int, seed: int, metadata: dict[str, Any] | None = None) -> None:
        """Start collecting one episode."""

    def on_step(self, event: StepEvent) -> None:
        """Record a step/replan event."""

    def on_intervention(self, event: InterventionEvent) -> None:
        """Record an intervention event."""

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        """Finish one episode."""

    def close(self) -> dict[str, Any]:
        """Flush artifacts and return a summary."""


class EvalSink:
    """Artifact sink for rollout-eval style summaries."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        execution_mode: str,
        ee16_execution_strategy: str = "mock_pointwise",
        mock_runtime: bool = False,
        real_env: bool = True,
        real_model: bool = True,
        summary_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.execution_mode = execution_mode
        self.ee16_execution_strategy = ee16_execution_strategy
        self.mock_runtime = bool(mock_runtime)
        self.real_env = bool(real_env)
        self.real_model = bool(real_model)
        self.summary_metadata = dict(summary_metadata or {})
        self._steps_by_episode: dict[int, list[StepEvent]] = {}
        self._interventions_by_episode: dict[int, list[InterventionEvent]] = {}
        self._summaries: list[EpisodeSummary] = []

    def on_episode_start(self, episode_id: int, seed: int, metadata: dict[str, Any] | None = None) -> None:
        del seed, metadata
        self._steps_by_episode.setdefault(int(episode_id), [])
        self._interventions_by_episode.setdefault(int(episode_id), [])

    def on_step(self, event: StepEvent) -> None:
        self._steps_by_episode.setdefault(int(event.episode_id), []).append(event)

    def on_intervention(self, event: InterventionEvent) -> None:
        self._interventions_by_episode.setdefault(int(event.episode_id), []).append(event)

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        self._summaries.append(summary)

    def close(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        episodes = []
        for summary in self._summaries:
            episode_id = int(summary.episode_id)
            rows = [event.to_hybrid_row() for event in self._steps_by_episode.get(episode_id, [])]
            records = [
                event.to_record() for event in self._interventions_by_episode.get(episode_id, [])
            ]
            hybrid_path = self.output_dir / f"hybrid_log_episode_{episode_id:04d}.parquet"
            records_path = self.output_dir / f"intervention_records_episode_{episode_id:04d}.parquet"
            write_rows_with_parquet_fallback(hybrid_path, rows)
            write_rows_with_parquet_fallback(records_path, records)
            episode = summary.to_dict()
            episode.update(summary.metadata)
            episode.update(
                {
                    "hybrid_log_path": str(hybrid_path),
                    "intervention_records_path": str(records_path),
                    "real_env": self.real_env,
                    "real_model": self.real_model,
                }
            )
            episodes.append(episode)
        success_rate = (
            float(np.mean([episode["success"] for episode in episodes])) if episodes else 0.0
        )
        summary_payload = {
            "mock_runtime": self.mock_runtime,
            "real_env": self.real_env,
            "real_model": self.real_model,
            "execution_mode": self.execution_mode,
            "ee16_execution_strategy": self.ee16_execution_strategy,
            "success_rate": success_rate,
            "mean_return": 0.0,
            "num_episodes": len(episodes),
            "save_dir": str(self.output_dir),
            "episodes": episodes,
        }
        summary_payload.update(self.summary_metadata)
        write_json(self.output_dir / "summary.json", summary_payload)
        return summary_payload


class ReplaySink:
    """Mock-friendly replay sink using the minimal TD3 replay schema."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        action_source: str,
        mock_runtime: bool = False,
        real_env: bool = True,
        real_model: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.action_source = action_source
        self.mock_runtime = bool(mock_runtime)
        self.real_env = bool(real_env)
        self.real_model = bool(real_model)
        self._interventions_by_episode: dict[int, list[InterventionEvent]] = {}
        self._summaries: list[EpisodeSummary] = []

    def on_episode_start(self, episode_id: int, seed: int, metadata: dict[str, Any] | None = None) -> None:
        del seed, metadata
        self._interventions_by_episode.setdefault(int(episode_id), [])

    def on_step(self, event: StepEvent) -> None:
        del event

    def on_intervention(self, event: InterventionEvent) -> None:
        self._interventions_by_episode.setdefault(int(event.episode_id), []).append(event)

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        self._summaries.append(summary)

    def close(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        transitions = []
        episode_rows = []
        for summary in self._summaries:
            records = sorted(
                self._interventions_by_episode.get(int(summary.episode_id), []),
                key=lambda event: (event.intervention_id, event.intervention_step_i),
            )
            episode_rows.append(
                {
                    "episode_id": summary.episode_id,
                    "seed": summary.seed,
                    "success": int(summary.success),
                    "failure_reason": summary.failure_reason or "",
                    "num_transitions": len(records),
                    "episode_reward": 1.0 if summary.success else 0.0,
                }
            )
            for index, record in enumerate(records):
                next_record = records[index + 1] if index + 1 < len(records) else record
                transitions.append(
                    {
                        "obs": np.asarray(record.metadata.get("obs_vector", [0.0] * 21), dtype=np.float32),
                        "action": np.asarray(record.applied_delta_local_xyz, dtype=np.float32),
                        "reward": np.float32(1.0 if summary.success else 0.0),
                        "next_obs": np.asarray(
                            next_record.metadata.get("obs_vector", [0.0] * 21),
                            dtype=np.float32,
                        ),
                        "done": np.bool_(index + 1 == len(records)),
                    }
                )
        replay = transitions_to_minimal_replay(transitions)
        np.savez_compressed(self.output_dir / "replay.npz", **replay)
        write_rows_csv(self.output_dir / "episodes_summary.csv", episode_rows)
        reward_summary = replay_reward_summary(
            replay,
            episode_rows,
            mock_runtime=self.mock_runtime,
            real_env=self.real_env,
            real_model=self.real_model,
        )
        write_json(self.output_dir / "reward_summary.json", reward_summary)
        config = {
            "mock_runtime": self.mock_runtime,
            "real_env": self.real_env,
            "real_model": self.real_model,
            "action_source": self.action_source,
        }
        write_json(self.output_dir / "config.json", config)
        return {
            "reward_summary": reward_summary,
            "paths": {
                "npz_path": str(self.output_dir / "replay.npz"),
                "episodes_summary_path": str(self.output_dir / "episodes_summary.csv"),
                "config_path": str(self.output_dir / "config.json"),
                "reward_summary_path": str(self.output_dir / "reward_summary.json"),
            },
        }


def transitions_to_minimal_replay(transitions: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Convert minimal transition dicts into TD3-compatible arrays."""

    if not transitions:
        return {
            "obs": np.empty((0, 21), dtype=np.float32),
            "action": np.empty((0, 3), dtype=np.float32),
            "reward": np.empty((0,), dtype=np.float32),
            "next_obs": np.empty((0, 21), dtype=np.float32),
            "done": np.empty((0,), dtype=np.bool_),
        }
    return {
        "obs": np.asarray([row["obs"] for row in transitions], dtype=np.float32),
        "action": np.asarray([row["action"] for row in transitions], dtype=np.float32),
        "reward": np.asarray([row["reward"] for row in transitions], dtype=np.float32),
        "next_obs": np.asarray([row["next_obs"] for row in transitions], dtype=np.float32),
        "done": np.asarray([row["done"] for row in transitions], dtype=np.bool_),
    }


def replay_reward_summary(
    replay: dict[str, np.ndarray],
    episode_rows: list[dict[str, Any]],
    *,
    mock_runtime: bool,
    real_env: bool,
    real_model: bool,
) -> dict[str, Any]:
    """Build a compact reward/replay summary."""

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
        "done_ratio": float(replay["done"].mean()) if replay["done"].size else 0.0,
        "success_rate": float(np.mean([row["success"] for row in episode_rows])) if episode_rows else 0.0,
        "mock_runtime": bool(mock_runtime),
        "real_env": bool(real_env),
        "real_model": bool(real_model),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_rows_with_parquet_fallback(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write parquet if pyarrow is present, and always write a CSV fallback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    write_rows_csv(csv_path, rows)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)
    except Exception:  # noqa: BLE001
        if not path.exists():
            path.write_text("", encoding="utf-8")


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write row dicts as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
