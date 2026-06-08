"""No-GPU mock artifacts for residual TD3 pipeline validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.residual_td3.episode_logger import (
    residual_prediction_stats,
    write_json,
)
from rlinf.algorithms.residual_td3.episode_sink import (
    EpisodeSummary,
    EvalSink,
    InterventionEvent,
    ReplaySink,
    StepEvent,
)
from rlinf.algorithms.residual_td3.residual_actor import (
    ResidualActorConfig,
    ZeroInitResidualActorMLP,
)
from rlinf.algorithms.residual_td3.stage_decision import decide_execution_plan


def write_mock_rollout_eval(
    output_dir: Path,
    *,
    execution_mode: str,
    num_episodes: int,
    max_steps: int,
    seed: int,
    gate_threshold: float = 0.6,
    residual_horizon_k: int = 50,
) -> dict[str, Any]:
    """Write mock rollout summary, hybrid logs, and intervention records."""

    sink = EvalSink(
        output_dir,
        execution_mode=execution_mode,
        mock_runtime=True,
        real_env=False,
        real_model=False,
        summary_metadata={"max_steps": int(max_steps)},
    )
    for episode_id in range(int(num_episodes)):
        episode_seed = int(seed) + episode_id
        rows, records = mock_episode_rows_and_records(
            episode_id=episode_id,
            seed=episode_seed,
            execution_mode=execution_mode,
            max_steps=max_steps,
            gate_threshold=gate_threshold,
            residual_horizon_k=residual_horizon_k,
        )
        sink.on_episode_start(episode_id, episode_seed, {"mock_runtime": True})
        for event in step_events_from_rows(rows):
            sink.on_step(event)
        for event in intervention_events_from_records(records, action_source="zero"):
            sink.on_intervention(event)
        stats = residual_prediction_stats(records)
        sink.on_episode_end(
            EpisodeSummary(
                episode_id=episode_id,
                seed=episode_seed,
                success=False,
                failure_reason="mock_timeout",
                episode_len=int(max_steps),
                num_interventions=int(sum(row["num_interventions"] for row in rows)),
                total_ee_steps=int(sum(row["num_intervention_steps"] for row in rows)),
                num_pregrasp_interventions=int(
                    sum(1 for row in rows if row["intervention_stage"] == "pregrasp")
                ),
                num_handover_interventions=int(
                    sum(1 for row in rows if row["intervention_stage"] == "handover")
                ),
                first_pregrasp_trigger_step=first_stage_step(rows, "pregrasp"),
                first_handover_trigger_step=first_stage_step(rows, "handover"),
                mock_runtime=True,
                metadata={
                    "return": 0.0,
                    "replan_steps": len(rows),
                    **stats,
                },
            )
        )
    return sink.close()


def write_mock_collect_replay(
    output_dir: Path,
    *,
    action_source: str,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Write mock rollout logs and replay artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_summary = write_mock_rollout_eval(
        output_dir / "rollout",
        execution_mode="handover_only_k50",
        num_episodes=num_episodes,
        max_steps=max_steps,
        seed=seed,
    )
    sink = ReplaySink(
        output_dir,
        action_source=action_source,
        mock_runtime=True,
        real_env=False,
        real_model=False,
    )
    for episode in rollout_summary["episodes"]:
        episode_id = int(episode["episode_id"])
        episode_seed = int(episode["seed"])
        rows, records = mock_episode_rows_and_records(
            episode_id=episode_id,
            seed=episode_seed,
            execution_mode="handover_only_k50",
            max_steps=max_steps,
            gate_threshold=0.6,
            residual_horizon_k=50,
        )
        sink.on_episode_start(episode_id, episode_seed, {"mock_runtime": True})
        for event in intervention_events_from_records(records, action_source=action_source):
            sink.on_intervention(event)
        sink.on_episode_end(
            EpisodeSummary(
                episode_id=episode_id,
                seed=episode_seed,
                success=False,
                failure_reason="mock_timeout",
                episode_len=int(max_steps),
                num_interventions=int(sum(row["num_interventions"] for row in rows)),
                total_ee_steps=len(records),
                num_pregrasp_interventions=0,
                num_handover_interventions=int(
                    sum(1 for row in rows if row["intervention_stage"] == "handover")
                ),
                first_pregrasp_trigger_step=None,
                first_handover_trigger_step=first_stage_step(rows, "handover"),
                mock_runtime=True,
            )
        )
    summary = sink.close()
    config_path = output_dir / "config.json"
    config = {
        "mock_runtime": True,
        "real_env": False,
        "real_model": False,
        "action_source": action_source,
        "num_episodes": int(num_episodes),
        "max_steps": int(max_steps),
        "seed": int(seed),
        "rollout_save_dir": str(output_dir / "rollout"),
        "rollout_summary_path": str(output_dir / "rollout" / "summary.json"),
    }
    write_json(config_path, config)
    write_json(output_dir / "mock_collect_summary.json", summary)
    return summary


def ensure_mock_bc_actor_checkpoint(path: Path) -> Path:
    """Create a zero-init BC actor checkpoint if it does not already exist."""

    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = ResidualActorConfig(
        obs_dim=21,
        hidden_dim=256,
        chunk_len=1,
        residual_dim=3,
        delta_max=0.01,
        zero_init_output=True,
    )
    model = ZeroInitResidualActorMLP(cfg)
    torch.save(
        {
            "actor_config": cfg.__dict__,
            "model_state_dict": model.state_dict(),
            "mock_runtime": True,
        },
        path,
    )
    return path


def mock_episode_rows_and_records(
    *,
    episode_id: int,
    seed: int,
    execution_mode: str,
    max_steps: int,
    gate_threshold: float,
    residual_horizon_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build deterministic mock hybrid rows and intervention records."""

    chunk_len = max(1, min(50, int(max_steps)))
    num_replans = max(1, int(np.ceil(max_steps / chunk_len)))
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for replan_id in range(num_replans):
        env_step_start = replan_id * chunk_len
        env_step_end = min(int(max_steps), env_step_start + chunk_len)
        gate_prob = 1.0 if replan_id == num_replans - 1 else 0.0
        plan = decide_execution_plan(
            execution_mode=execution_mode,
            gate_prob=gate_prob,
            gate_threshold=gate_threshold,
            pregrasp_triggered=False,
            residual_horizon_k=residual_horizon_k,
            action_chunk_len=chunk_len,
        )
        rows.append(
            {
                "episode_id": episode_id,
                "replan_id": replan_id,
                "seed": seed,
                "env_step": env_step_end,
                "env_step_start": env_step_start,
                "env_step_end": env_step_end,
                "gate_logit": 0.0,
                "gate_prob": gate_prob,
                "gate_binary": int(plan.gate_binary),
                "handover_gate_binary": int(plan.gate_binary),
                "pregrasp_trigger_binary": 0,
                "pregrasp_trigger_score": 0.0,
                "execution_mode": plan.execution_mode,
                "intervention_stage": plan.intervention_stage_name,
                "trigger_source": plan.runtime_trigger_source,
                "num_interventions": int(plan.ee16_binary),
                "num_intervention_steps": len(plan.selected_action_chunk_indices),
                "selected_indices": plan.selected_action_chunk_indices if plan.ee16_binary else None,
                "obs_state_06": 0.5,
                "obs_state_13": 0.5,
                "d_LR": 0.1,
                "success": 0,
                "done": int(env_step_end >= int(max_steps)),
            }
        )
        if plan.ee16_binary:
            records.extend(
                mock_intervention_records(
                    episode_id=episode_id,
                    seed=seed,
                    env_step_start=env_step_start,
                    intervention_id=replan_id,
                    num_steps=len(plan.selected_action_chunk_indices),
                    gate_score=gate_prob,
                    stage=plan.intervention_stage_name,
                    trigger_source=plan.runtime_trigger_source,
                )
            )
    return rows, records


def mock_intervention_records(
    *,
    episode_id: int,
    seed: int,
    env_step_start: int,
    intervention_id: int,
    num_steps: int,
    gate_score: float,
    stage: str,
    trigger_source: str,
) -> list[dict[str, Any]]:
    """Build deterministic mock intervention records matching replay schema."""

    records = []
    for step_i in range(num_steps):
        obs = np.linspace(0.0, 1.0, 21, dtype=np.float32) + float(step_i) * 0.001
        delta = np.zeros(3, dtype=np.float32)
        base_ee16 = np.zeros(16, dtype=np.float32)
        exec_ee16 = np.zeros(16, dtype=np.float32)
        records.append(
            {
                "episode_id": episode_id,
                "seed": seed,
                "env_step": env_step_start + step_i,
                "intervention_id": intervention_id,
                "intervention_step_i": step_i,
                "intervention_stage": stage,
                "trigger_source": trigger_source,
                "gate_score": float(gate_score),
                "obs_vector": obs.tolist(),
                "pred_delta_local_xyz": delta.tolist(),
                "applied_delta_local_xyz": delta.tolist(),
                "applied_delta_world_xyz": delta.tolist(),
                "pred_delta_norm": 0.0,
                "applied_delta_norm": 0.0,
                "base_ee16": base_ee16.tolist(),
                "exec_ee16": exec_ee16.tolist(),
                "residual_scale": 1.0,
                "noise_std": 0.0,
                "saturation": 0,
                "has_nan_or_inf": 0,
            }
        )
    return records


def step_events_from_rows(rows: list[dict[str, Any]]) -> list[StepEvent]:
    """Convert mock hybrid rows into sink step events."""

    events = []
    for row in rows:
        selected = row.get("selected_indices") or []
        events.append(
            StepEvent(
                episode_id=int(row["episode_id"]),
                seed=int(row["seed"]),
                env_step=int(row["env_step"]),
                action_mode=str(row["execution_mode"]),
                execution_stage=str(row["intervention_stage"]),
                trigger_source=str(row["trigger_source"]),
                gate_score=float(row["gate_prob"]),
                selected_action_chunk_indices=list(selected),
                has_intervention=bool(row["num_interventions"]),
                obs_features=None,
                action=None,
                reward=0.0,
                done=bool(row["done"]),
                metadata={
                    "env_step_start": int(row["env_step_start"]),
                    "env_step_end": int(row["env_step_end"]),
                    "replan_id": int(row["replan_id"]),
                    "gate_logit": float(row["gate_logit"]),
                    "pregrasp_trigger_score": float(row["pregrasp_trigger_score"]),
                    "obs_state_06": float(row["obs_state_06"]),
                    "obs_state_13": float(row["obs_state_13"]),
                    "d_LR": row["d_LR"],
                    "success": bool(row["success"]),
                },
            )
        )
    return events


def intervention_events_from_records(
    records: list[dict[str, Any]],
    *,
    action_source: str,
) -> list[InterventionEvent]:
    """Convert mock intervention records into sink intervention events."""

    events = []
    for record in records:
        events.append(
            InterventionEvent(
                episode_id=int(record["episode_id"]),
                seed=int(record["seed"]),
                env_step=int(record["env_step"]),
                intervention_id=int(record["intervention_id"]),
                intervention_step_i=int(record["intervention_step_i"]),
                stage=str(record["intervention_stage"]),
                trigger_source=str(record["trigger_source"]),
                action_source=action_source,
                pred_delta_local_xyz=list(record["pred_delta_local_xyz"]),
                applied_delta_local_xyz=list(record["applied_delta_local_xyz"]),
                applied_delta_world_xyz=list(record["applied_delta_world_xyz"]),
                base_ee16=list(record["base_ee16"]),
                exec_ee16=list(record["exec_ee16"]),
                saturation=int(record["saturation"]),
                has_nan_or_inf=int(record["has_nan_or_inf"]),
                metadata={
                    "gate_score": float(record["gate_score"]),
                    "obs_vector": list(record["obs_vector"]),
                    "residual_scale": float(record["residual_scale"]),
                    "noise_std": float(record["noise_std"]),
                },
            )
        )
    return events


def first_stage_step(rows: list[dict[str, Any]], stage: str) -> int | None:
    """Return first env step for a mock stage intervention."""

    for row in rows:
        if row["intervention_stage"] == stage and row["num_interventions"]:
            return int(row["env_step_start"])
    return None


def read_replay_summary(npz_path: Path) -> dict[str, Any]:
    """Return a compact shape summary for mock validation tests."""

    with np.load(npz_path) as data:
        return {
            "obs_shape": list(data["obs"].shape),
            "action_shape": list(data["action"].shape),
            "reward_shape": list(data["reward"].shape),
            "next_obs_shape": list(data["next_obs"].shape),
            "done_shape": list(data["done"].shape),
        }
