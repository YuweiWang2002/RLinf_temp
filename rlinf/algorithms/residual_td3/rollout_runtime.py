"""Runtime construction helpers shared by residual rollout entrypoints."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rlinf.algorithms.residual_td3.gate_head import ChunkAwareGateRuntime
from rlinf.algorithms.residual_td3.residual_actor_runtime import (
    build_intervention_runner_from_args,
)
from rlinf.algorithms.residual_td3.residual_ee_intervention import (
    ResidualEEInterventionRunner,
)
from rlinf.algorithms.residual_td3.stage_decision import (
    handover_intervention_enabled_for_mode,
    pregrasp_intervention_enabled_for_mode,
)
from scripts.rollout_pi05_fk_ee16_zero_residual import (
    build_actor_model_cfg,
    close_env,
    get_single_task,
    load_env_cfg,
    load_model,
    make_env,
    select_episode_seeds,
)


@dataclass
class ResidualRolloutRuntime:
    """Initialized rollout runtime shared by eval and replay collection."""

    args: Namespace
    env_args: SimpleNamespace
    model_args: SimpleNamespace
    env_cfg: Any
    episode_seeds: list[int]
    gate_runtime: ChunkAwareGateRuntime | None
    env: Any
    model: Any
    intervention_runner: ResidualEEInterventionRunner | None

    def close(self) -> None:
        """Close the underlying environment."""

        close_env(self.env)


class RolloutRuntimeFactory:
    """Build the env, model, gate, and residual intervention runtime."""

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def create(
        self,
        *,
        episode_seeds: list[int] | None = None,
        build_intervention: bool | None = None,
        debug_ee16_timing: bool | None = None,
    ) -> ResidualRolloutRuntime:
        """Initialize runtime using the same helpers as the legacy scripts."""

        env_args = build_env_args(self.args)
        model_args = build_model_args(self.args)
        env_cfg = load_env_cfg(env_args)
        if self.args.execution_mode == "gate_controlled_hybrid" and hasattr(
            env_cfg.task_config,
            "planner_backend",
        ):
            env_cfg.task_config.planner_backend = self.args.planner_backend
        resolved_episode_seeds = episode_seeds or select_episode_seeds(env_cfg, env_args)
        gate_runtime = load_gate_runtime(self.args)
        env = make_env(env_cfg)
        env.debug_ee16_timing = (
            bool(self.args.debug_ee16_timing)
            if debug_ee16_timing is None
            else bool(debug_ee16_timing)
        )
        env.ee16_execution_strategy = self.args.ee16_execution_strategy
        model = load_model(build_actor_model_cfg(model_args), self.args.device)

        should_build_intervention = (
            ee_intervention_enabled(self.args)
            if build_intervention is None
            else bool(build_intervention)
        )
        intervention_runner = None
        if should_build_intervention:
            task = get_single_task_after_reset(env)
            intervention_runner = build_intervention_runner(self.args, task)

        return ResidualRolloutRuntime(
            args=self.args,
            env_args=env_args,
            model_args=model_args,
            env_cfg=env_cfg,
            episode_seeds=resolved_episode_seeds,
            gate_runtime=gate_runtime,
            env=env,
            model=model,
            intervention_runner=intervention_runner,
        )


def build_env_args(args: Namespace) -> SimpleNamespace:
    """Build env args expected by the RoboTwin rollout helpers."""

    return SimpleNamespace(
        config=args.env_config,
        env_split=args.env_split,
        task_name=args.task_name,
        seed=args.seed,
        seed_offset=args.seed_offset,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        chunk_len=args.chunk_len,
        use_eval_success_seeds=args.use_eval_success_seeds,
        save_video=args.save_video,
        save_dir=args.save_dir,
        data_dir=str(Path(args.save_dir) / "data"),
        execution_mode="qpos14",
        robotwin_assets_path=args.robotwin_assets_path,
    )


def build_model_args(args: Namespace) -> SimpleNamespace:
    """Build model args expected by pi05 model loading helpers."""

    return SimpleNamespace(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats_path,
        model_num_action_chunks=args.model_num_action_chunks,
        config_name=args.config,
        num_images_in_input=args.num_images_in_input,
        noise_level=args.noise_level,
    )


def build_collect_replay_rollout_args(args: Namespace) -> SimpleNamespace:
    """Build legacy rollout args from collect-replay CLI args."""

    rollout_save_dir = args.rollout_save_dir or str(Path(args.save_dir) / "rollout")
    residual_actor = "random_noise" if args.action_source == "random_noise" else args.action_source
    return SimpleNamespace(
        config=args.config,
        env_config=args.env_config,
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats_path,
        chunk_aware_gate_checkpoint=args.chunk_aware_gate_checkpoint,
        gate_type="chunk_aware",
        gate_threshold=args.gate_threshold,
        gate_chunk_len=args.gate_chunk_len,
        execution_mode="gate_controlled_hybrid",
        task_config=None,
        task_config_missing_runtime_fields=[],
        enable_pregrasp_intervention=False,
        gripper_close_direction=None,
        left_gripper_open_value=None,
        left_gripper_closed_value=None,
        gripper_close_threshold=None,
        gripper_closing_delta_threshold=0.15,
        pregrasp_cooldown_steps=50,
        enable_residual_intervention=True,
        residual_actor=residual_actor,
        residual_actor_checkpoint=args.residual_actor_checkpoint,
        residual_dry_run=False,
        residual_scale=args.residual_scale,
        residual_noise_std=args.residual_noise_std,
        log_bc_residual_predictions=True,
        enable_learned_residual_control=True,
        residual_constant_delta_local_xyz=(0.0, 0.0, 0.0),
        residual_horizon_k=args.residual_horizon_k,
        residual_target_horizon_offset=args.residual_target_horizon_offset,
        residual_max_delta_local_xyz=args.residual_max_delta_local_xyz,
        left_stabilization_mode="none",
        left_deadband_xyz=1e-4,
        left_lowpass_alpha=0.5,
        ee16_execution_strategy=args.ee16_execution_strategy,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        chunk_len=args.chunk_len,
        model_num_action_chunks=args.model_num_action_chunks,
        seed=args.seed,
        seed_offset=args.seed_offset,
        use_eval_success_seeds=args.use_eval_success_seeds,
        task_name=args.task_name,
        env_split=args.env_split,
        save_dir=rollout_save_dir,
        save_video=args.save_video,
        video_base_dir=None,
        num_save_videos=None,
        video_temp_subsample=None,
        video_frame_mode=args.video_frame_mode,
        save_debug=args.save_debug,
        save_gate_plots=False,
        video_fps=args.video_fps,
        video_source=args.video_source,
        device=args.device,
        num_images_in_input=args.num_images_in_input,
        noise_level=args.noise_level,
        planner_backend=args.planner_backend,
        robotwin_runtime_bootstrap=args.robotwin_runtime_bootstrap,
        robotwin_path=args.robotwin_path,
        curobo_src_path=args.curobo_src_path,
        robotwin_assets_path=args.robotwin_assets_path,
        robotwin_setup_max_retries=args.robotwin_setup_max_retries,
        debug_ee16_timing=False,
    )


def load_gate_runtime(args: Namespace) -> ChunkAwareGateRuntime | None:
    """Load the chunk-aware gate runtime when handover intervention may need it."""

    if not handover_intervention_enabled(args) and args.chunk_aware_gate_checkpoint is None:
        return None
    if not args.chunk_aware_gate_checkpoint:
        raise ValueError("--chunk-aware-gate-checkpoint is required for handover gate intervention.")
    return ChunkAwareGateRuntime.load_from_checkpoint(
        args.chunk_aware_gate_checkpoint,
        device=args.device,
        threshold=args.gate_threshold,
    )


def build_intervention_runner(args: Namespace, task: Any) -> ResidualEEInterventionRunner:
    """Build the FK bridge, residual actor, and intervention runner."""

    return build_intervention_runner_from_args(
        task,
        args,
        resolve_residual_control_flags(args),
    )


def get_single_task_after_reset(env: Any) -> Any:
    """Reset env once and return the wrapped RoboTwin task."""

    env.reset()
    return get_single_task(env)


def validate_residual_control_args(args: Namespace) -> None:
    """Validate residual/pregrasp flags with legacy rollout semantics."""

    if bool(args.residual_dry_run) and bool(args.enable_learned_residual_control):
        raise ValueError(
            "--residual-dry-run and --enable-learned-residual-control cannot be enabled together."
        )
    if pregrasp_intervention_enabled(args) and getattr(args, "gripper_close_direction", None) is None:
        raise ValueError("--gripper-close-direction is required for pregrasp intervention.")
    if (
        getattr(args, "left_gripper_closed_value", None) is not None
        and getattr(args, "gripper_close_threshold", None) is None
    ):
        args.gripper_close_threshold = float(args.left_gripper_closed_value)


def resolve_residual_control_flags(args: Namespace) -> dict[str, bool]:
    """Resolve dry-run vs learned residual control flags."""

    validate_residual_control_args(args)
    if bool(args.residual_dry_run):
        return {"dry_run": True, "learned_residual_control_enabled": False}
    if bool(args.enable_learned_residual_control):
        return {"dry_run": False, "learned_residual_control_enabled": True}
    return {"dry_run": False, "learned_residual_control_enabled": False}


def resolve_residual_dry_run(args: Namespace) -> bool:
    """Return the effective residual dry-run flag."""

    return bool(resolve_residual_control_flags(args)["dry_run"])


def learned_residual_control_enabled(args: Namespace) -> bool:
    """Return the effective learned residual control flag."""

    return bool(resolve_residual_control_flags(args)["learned_residual_control_enabled"])


def handover_intervention_enabled(args: Namespace) -> bool:
    """Return whether the current args enable handover EE intervention."""

    return handover_intervention_enabled_for_mode(
        getattr(args, "execution_mode", "gate_controlled_hybrid")
    )


def pregrasp_intervention_enabled(args: Namespace) -> bool:
    """Return whether the current args enable pregrasp EE intervention."""

    return pregrasp_intervention_enabled_for_mode(
        getattr(args, "execution_mode", "gate_controlled_hybrid"),
        enable_pregrasp_intervention=getattr(args, "enable_pregrasp_intervention", False),
    )


def ee_intervention_enabled(args: Namespace) -> bool:
    """Return whether any EE intervention path is enabled."""

    return handover_intervention_enabled(args) or pregrasp_intervention_enabled(args)
