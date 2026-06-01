"""Collect residual replay from gate-triggered online interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from rlinf.algorithms.residual_td3.residual_replay import (
    ReplayRewardConfig,
    build_replay_from_rollout,
    load_baseline_success_by_seed,
    load_intervention_records,
    save_replay_artifacts,
)
from scripts.rollout_pi05_fk_ee16_zero_residual import (
    DEFAULT_CONFIG as DEFAULT_ENV_CONFIG,
)
from scripts.rollout_pi05_gate_controlled_hybrid_zero_residual import (
    bootstrap_robotwin_runtime,
    build_actor_model_cfg,
    build_env_args,
    build_intervention_runner,
    build_model_args,
    close_env,
    configure_line_buffering,
    get_single_task_after_reset,
    load_env_cfg,
    load_gate_runtime,
    load_model,
    make_env,
    run_episodes,
    select_episode_seeds,
)
from scripts.rollout_pi05_with_gate_logging import DEFAULT_NORM_STATS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--env-config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats-path", default=DEFAULT_NORM_STATS)
    parser.add_argument("--chunk-aware-gate-checkpoint", required=True)
    parser.add_argument("--action-source", choices=("zero", "bc", "random_noise"), required=True)
    parser.add_argument("--residual-actor-checkpoint", default=None)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-noise-std", type=float, default=0.001)
    parser.add_argument("--residual-horizon-k", type=int, default=10)
    parser.add_argument("--residual-max-delta-local-xyz", type=float, default=0.05)
    parser.add_argument("--residual-target-horizon-offset", type=int, default=0)
    parser.add_argument("--gate-threshold", type=float, default=0.6)
    parser.add_argument("--gate-chunk-len", type=int, default=50)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--model-num-action-chunks", type=int, default=50)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=100100000)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--use-eval-success-seeds", action="store_true")
    parser.add_argument("--env-split", choices=("train", "eval"), default="eval")
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--ee16-execution-strategy", choices=("pointwise", "last_target"), default="pointwise")
    parser.add_argument("--planner-backend", choices=("curobo", "mplib"), default="curobo")
    parser.add_argument("--robotwin-runtime-bootstrap", choices=("auto", "none"), default="auto")
    parser.add_argument("--robotwin-path", default=None)
    parser.add_argument("--curobo-src-path", default=None)
    parser.add_argument("--robotwin-assets-path", default=None)
    parser.add_argument("--robotwin-setup-max-retries", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-images-in-input", type=int, default=3)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-source", choices=("auto", "third_view", "observer", "main"), default="third_view")
    parser.add_argument("--video-frame-mode", choices=("replan", "executed_step"), default="replan")
    parser.add_argument("--save-dir", required=True, help="Residual replay output directory.")
    parser.add_argument("--rollout-save-dir", default=None, help="Optional rollout log directory.")
    parser.add_argument("--reward-mode", choices=("sparse_success", "baseline_comparison"), default="sparse_success")
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--reward-discount-to-steps", choices=("all", "last"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_line_buffering()
    bootstrap_robotwin_runtime(args)
    if args.reward_mode == "baseline_comparison" and not args.baseline_summary:
        raise ValueError("--baseline-summary is required when --reward-mode=baseline_comparison.")
    if args.action_source == "bc" and not args.residual_actor_checkpoint:
        raise ValueError("--residual-actor-checkpoint is required for --action-source=bc.")

    rollout_args = build_rollout_args(args)
    env = None
    try:
        out_dir = Path(args.save_dir)
        rollout_dir = Path(rollout_args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rollout_dir.mkdir(parents=True, exist_ok=True)

        env_args = build_env_args(rollout_args)
        model_args = build_model_args(rollout_args)
        env_cfg = load_env_cfg(env_args)
        if hasattr(env_cfg.task_config, "planner_backend"):
            env_cfg.task_config.planner_backend = rollout_args.planner_backend
        gate_runtime = load_gate_runtime(rollout_args)
        env = make_env(env_cfg)
        env.debug_ee16_timing = False
        env.ee16_execution_strategy = rollout_args.ee16_execution_strategy
        model = load_model(build_actor_model_cfg(model_args), rollout_args.device)
        task = get_single_task_after_reset(env)
        intervention_runner = build_intervention_runner(task, rollout_args)
        episode_seeds = select_episode_seeds(env_cfg, env_args)
        rollout_summary = run_episodes(
            env,
            model,
            gate_runtime,
            intervention_runner,
            rollout_args,
            episode_seeds,
        )
        rollout_summary_path = rollout_dir / "summary.json"
        records_by_episode = {
            int(ep["episode_id"]): load_intervention_records(ep["intervention_records_path"])
            for ep in rollout_summary.get("episodes", [])
        }
        reward_config = ReplayRewardConfig(
            mode=args.reward_mode,
            discount_to_steps=args.reward_discount_to_steps,
            baseline_summary_path=args.baseline_summary,
        )
        baseline_success = (
            load_baseline_success_by_seed(args.baseline_summary)
            if args.reward_mode == "baseline_comparison"
            else None
        )
        replay, episode_rows, reward_summary = build_replay_from_rollout(
            rollout_summary,
            records_by_episode,
            action_source=args.action_source,
            reward_config=reward_config,
            baseline_success_by_seed=baseline_success,
        )
        config = {
            **vars(args),
            "rollout_save_dir": str(rollout_dir),
            "rollout_summary_path": str(rollout_summary_path),
        }
        paths = save_replay_artifacts(out_dir, replay, episode_rows, reward_summary, config)
        print(json.dumps({"reward_summary": reward_summary, "paths": paths}, indent=2))
        return 0
    finally:
        if env is not None:
            close_env(env)


def build_rollout_args(args: argparse.Namespace) -> SimpleNamespace:
    """Build the rollout namespace expected by the hybrid rollout helpers."""

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


if __name__ == "__main__":
    raise SystemExit(main())
