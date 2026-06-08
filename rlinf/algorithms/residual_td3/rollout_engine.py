"""Minimal residual rollout engine skeleton.

This module intentionally keeps the existing rollout script as the source of
behavior for now.  It provides the stable object boundary that future work can
fill in as code moves out of the script.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.algorithms.residual_td3.episode_logger import write_json


@dataclass
class ResidualRolloutEngine:
    """Small orchestration boundary for residual rollout evaluation."""

    env: Any
    model: Any
    gate_runtime: Any | None
    intervention_runner: Any | None
    args: Any
    run_episode_fn: Callable[[Any, Any, Any | None, Any | None, Any, int, int], dict[str, Any]]

    def run_episode(self, episode_id: int, seed: int) -> dict[str, Any]:
        """Run one episode through the configured episode function."""

        return self.run_episode_fn(
            self.env,
            self.model,
            self.gate_runtime,
            self.intervention_runner,
            self.args,
            episode_id,
            seed,
        )

    def run(self, episode_seeds: Sequence[int]) -> dict[str, Any]:
        """Run episodes and return the aggregate summary used by current scripts."""

        save_dir = Path(self.args.save_dir)
        summaries = []
        for episode_id, seed in enumerate(episode_seeds):
            summary = self.run_episode(episode_id, int(seed))
            summaries.append(summary)
            write_json(save_dir / "summary.json", {"episodes": summaries})
        successes = [row["success"] for row in summaries]
        returns = [row["return"] for row in summaries]
        return {
            "execution_mode": self.args.execution_mode,
            "ee16_execution_strategy": self.args.ee16_execution_strategy,
            "gate_type": self.args.gate_type,
            "gate_threshold": self.args.gate_threshold,
            "gate_chunk_len": self.args.gate_chunk_len,
            "chunk_aware_gate_checkpoint": self.args.chunk_aware_gate_checkpoint,
            "enable_residual_intervention": bool(self.args.enable_residual_intervention),
            "enable_pregrasp_intervention": _pregrasp_intervention_enabled(self.args),
            "gripper_close_direction": self.args.gripper_close_direction,
            "gripper_close_threshold": self.args.gripper_close_threshold,
            "gripper_closing_delta_threshold": self.args.gripper_closing_delta_threshold,
            "pregrasp_cooldown_steps": self.args.pregrasp_cooldown_steps,
            "residual_actor": self.args.residual_actor,
            "residual_actor_checkpoint": self.args.residual_actor_checkpoint,
            "residual_dry_run": _resolve_residual_dry_run(self.args),
            "residual_scale": self.args.residual_scale,
            "enable_learned_residual_control": _learned_residual_control_enabled(self.args),
            "residual_horizon_k": self.args.residual_horizon_k,
            "residual_target_horizon_offset": self.args.residual_target_horizon_offset,
            "residual_max_delta_local_xyz": self.args.residual_max_delta_local_xyz,
            "chunk_len": self.args.chunk_len,
            "model_num_action_chunks": self.args.model_num_action_chunks,
            "max_steps": self.args.max_steps,
            "requested_num_episodes": self.args.num_episodes,
            "rollout_status": "completed",
            "simulator_crash": False,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "episodes": summaries,
            "save_dir": str(save_dir),
        }


def _resolve_residual_dry_run(args: Any) -> bool:
    if bool(args.residual_dry_run):
        return True
    if bool(args.enable_learned_residual_control):
        return False
    return not bool(args.enable_residual_intervention)


def _learned_residual_control_enabled(args: Any) -> bool:
    if bool(args.residual_dry_run):
        return False
    if bool(args.enable_learned_residual_control):
        return True
    return bool(args.enable_residual_intervention)


def _pregrasp_intervention_enabled(args: Any) -> bool:
    return bool(getattr(args, "enable_pregrasp_intervention", False)) or getattr(
        args,
        "execution_mode",
        "",
    ) in {
        "pregrasp_only_k50",
        "pregrasp_plus_handover_k50",
    }
