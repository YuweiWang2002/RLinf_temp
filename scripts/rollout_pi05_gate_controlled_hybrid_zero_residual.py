"""Roll out pi05 with v2 gate-controlled qpos14/ee16 zero-residual switching."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.residual_td3.episode_logger import (
    print_summary,
    residual_prediction_stats,
    save_hybrid_log,
    save_hybrid_plot,
    save_intervention_records,
    write_json,
)
from rlinf.algorithms.residual_td3.gate_head import ChunkAwareGateRuntime
from rlinf.algorithms.residual_td3.pregrasp_trigger import (
    PregraspTriggerConfig,
    PregraspTriggerResult,
    detect_pregrasp_trigger,
)
from rlinf.algorithms.residual_td3.residual_actor_runtime import (
    build_pregrasp_intervention_runner as build_pregrasp_runner_from_handover,
)
from rlinf.algorithms.residual_td3.residual_ee_intervention import (
    ResidualEEInterventionRunner,
)
from rlinf.algorithms.residual_td3.rollout_engine import ResidualRolloutEngine
from rlinf.algorithms.residual_td3.rollout_runtime import (
    RolloutRuntimeFactory,
    handover_intervention_enabled,
    learned_residual_control_enabled,
    pregrasp_intervention_enabled,
    resolve_residual_dry_run,
    validate_residual_control_args,
)
from rlinf.algorithms.residual_td3.rollout_runtime import (
    resolve_residual_control_flags as _resolve_residual_control_flags,
)
from rlinf.algorithms.residual_td3.stage_decision import (
    InterventionStage,
    decide_execution_plan,
)
from rlinf.algorithms.residual_td3.task_config import (
    load_residual_task_config,
    task_config_to_rollout_defaults,
)
from scripts.rollout_pi05_fk_ee16_zero_residual import (
    DEFAULT_CONFIG as DEFAULT_ENV_CONFIG,
)
from scripts.rollout_pi05_fk_ee16_zero_residual import (
    append_frame,
    bootstrap_robotwin_runtime,
    configure_line_buffering,
    get_single_task,
    read_pose16,
    read_task_qpos14,
    save_video,
    to_numpy,
)
from scripts.rollout_pi05_with_gate_logging import (
    DEFAULT_NORM_STATS,
    extract_state14,
    predict_qpos14_chunk_with_feature,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--task-config", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--config", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--env-config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--checkpoint", required=pre_args.task_config is None)
    parser.add_argument("--norm-stats-path", default=DEFAULT_NORM_STATS)
    parser.add_argument(
        "--chunk-aware-gate-checkpoint",
        "--chunk_aware_gate_checkpoint",
        dest="chunk_aware_gate_checkpoint",
        default=None,
    )
    parser.add_argument(
        "--gate-type",
        "--gate_type",
        dest="gate_type",
        choices=("chunk_aware",),
        default="chunk_aware",
    )
    parser.add_argument("--gate-threshold", "--gate_threshold", dest="gate_threshold", type=float, default=0.6)
    parser.add_argument("--gate-chunk-len", "--gate_chunk_len", dest="gate_chunk_len", type=int, default=50)
    parser.add_argument(
        "--execution-mode",
        choices=(
            "qpos14_baseline",
            "gate_controlled_hybrid",
            "handover_only_k50",
            "pregrasp_only_k50",
            "pregrasp_plus_handover_k50",
        ),
        default="gate_controlled_hybrid",
    )
    parser.add_argument("--enable-pregrasp-intervention", action="store_true")
    parser.add_argument(
        "--gripper-close-direction",
        choices=("lower", "higher"),
        default=None,
        help="Direction of numeric left-gripper closing for pregrasp trigger.",
    )
    parser.add_argument("--left-gripper-open-value", type=float, default=None)
    parser.add_argument("--left-gripper-closed-value", type=float, default=None)
    parser.add_argument("--gripper-close-threshold", type=float, default=None)
    parser.add_argument("--gripper-closing-delta-threshold", type=float, default=0.15)
    parser.add_argument("--pregrasp-cooldown-steps", type=int, default=50)
    parser.add_argument(
        "--enable-residual-intervention",
        "--enable_residual_intervention",
        dest="enable_residual_intervention",
        action="store_true",
    )
    parser.add_argument(
        "--residual-actor",
        "--residual_actor",
        "--residual-actor-type",
        dest="residual_actor",
        choices=("zero", "constant", "zero_init", "bc", "td3", "random_noise"),
        default="zero",
    )
    parser.add_argument("--residual-actor-checkpoint", default=None)
    parser.add_argument("--residual-dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-noise-std", type=float, default=0.001)
    parser.add_argument("--log-bc-residual-predictions", action="store_true")
    parser.add_argument("--enable-learned-residual-control", action="store_true")
    parser.add_argument(
        "--residual-constant-delta-local-xyz",
        "--residual_constant_delta_local_xyz",
        dest="residual_constant_delta_local_xyz",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--residual-horizon-k", "--residual_horizon_k", dest="residual_horizon_k", type=int, default=50)
    parser.add_argument(
        "--max-handover-interventions-per-episode",
        type=int,
        default=None,
        help="Optional cap on handover EE intervention replans per episode; default keeps existing unlimited behavior.",
    )
    parser.add_argument(
        "--residual-target-horizon-offset",
        "--residual_target_horizon_offset",
        dest="residual_target_horizon_offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--residual-max-delta-local-xyz",
        "--residual_max_delta_local_xyz",
        dest="residual_max_delta_local_xyz",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--left-stabilization-mode",
        "--left_stabilization_mode",
        dest="left_stabilization_mode",
        choices=("none", "deadband", "lowpass", "freeze"),
        default="none",
    )
    parser.add_argument(
        "--left-deadband-xyz",
        "--left_deadband_xyz",
        dest="left_deadband_xyz",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--left-lowpass-alpha",
        "--left_lowpass_alpha",
        dest="left_lowpass_alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ee16-execution-strategy",
        "--ee16_execution_strategy",
        "--residual-execution-mode",
        "--residual_execution_mode",
        dest="ee16_execution_strategy",
        choices=("pointwise", "last_target"),
        default="pointwise",
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--model-num-action-chunks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100100000)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--use-eval-success-seeds", action="store_true")
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--env-split", choices=("train", "eval"), default="eval")
    parser.add_argument("--save-dir", required=pre_args.task_config is None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--video-frame-mode",
        choices=("replan", "executed_step"),
        default="replan",
        help=(
            "Frame sampling for the script-built mp4. 'replan' records one frame per "
            "policy replan for qpos/ee consistency; 'executed_step' records every "
            "obs returned by env.chunk_step, which makes pointwise videos much longer."
        ),
    )
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--save-gate-plots", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-source", choices=("auto", "third_view", "observer", "main"), default="third_view")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-images-in-input", type=int, default=3)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--planner-backend", choices=("curobo", "mplib"), default="curobo")
    parser.add_argument(
        "--robotwin-runtime-bootstrap",
        choices=("auto", "none"),
        default="auto",
        help="Auto-configure RoboTwin/curobo import paths for remote eval environments such as JS4.",
    )
    parser.add_argument("--robotwin-path", default=None)
    parser.add_argument("--curobo-src-path", default=None)
    parser.add_argument("--robotwin-assets-path", default=None)
    parser.add_argument("--robotwin-setup-max-retries", type=int, default=8)
    parser.add_argument("--debug-ee16-timing", action="store_true")
    if pre_args.task_config:
        task_cfg = load_residual_task_config(pre_args.task_config)
        if task_cfg.missing_runtime_fields:
            raise ValueError(
                "--task-config is missing runtime fields: "
                + ", ".join(task_cfg.missing_runtime_fields)
            )
        parser.set_defaults(**task_config_to_rollout_defaults(task_cfg))
    args = parser.parse_args(argv)
    if not hasattr(args, "task_config_missing_runtime_fields"):
        args.task_config_missing_runtime_fields = []
    return args


def main() -> int:
    args = parse_args()
    configure_line_buffering()
    bootstrap_robotwin_runtime(args)
    validate_residual_control_args(args)
    print_context(args)
    runtime = None
    try:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        runtime = RolloutRuntimeFactory(args).create()
        metrics = run_episodes(
            runtime.env,
            runtime.model,
            runtime.gate_runtime,
            runtime.intervention_runner,
            args,
            runtime.episode_seeds,
        )
        write_json(save_dir / "summary.json", metrics)
        print_summary(metrics)
        runtime.close()
        runtime = None
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as exc:  # noqa: BLE001
        print(f"HYBRID ROLLOUT: FAIL {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().strip().splitlines()[-24:]:
            print(f"  {line}")
        return 1
    finally:
        if runtime is not None:
            runtime.close()


def build_pregrasp_intervention_runner(
    handover_runner: ResidualEEInterventionRunner,
) -> ResidualEEInterventionRunner:
    return build_pregrasp_runner_from_handover(handover_runner)


def run_episodes(
    env: Any,
    model: Any,
    gate_runtime: ChunkAwareGateRuntime | None,
    intervention_runner: ResidualEEInterventionRunner | None,
    args: argparse.Namespace,
    episode_seeds: list[int],
) -> dict[str, Any]:
    return ResidualRolloutEngine(
        env=env,
        model=model,
        gate_runtime=gate_runtime,
        intervention_runner=intervention_runner,
        args=args,
        run_episode_fn=run_episode,
    ).run(episode_seeds)


def run_episode(
    env: Any,
    model: Any,
    gate_runtime: ChunkAwareGateRuntime | None,
    intervention_runner: ResidualEEInterventionRunner | None,
    args: argparse.Namespace,
    episode_id: int,
    seed: int,
) -> dict[str, Any]:
    obs, _ = env.reset(env_seeds=[seed])
    task = get_single_task(env)
    frames: list[np.ndarray] = []
    append_frame(frames, env, obs, args.video_source)
    rows: list[dict[str, Any]] = []
    intervention_records: list[dict[str, object]] = []
    pregrasp_config = build_pregrasp_trigger_config(args)
    pregrasp_runner = build_pregrasp_intervention_runner(intervention_runner) if intervention_runner else None
    last_pregrasp_trigger_step: int | None = None
    episode_return = 0.0
    env_steps = 0
    replan_id = 0
    done = False
    start = time.perf_counter()

    while env_steps < args.max_steps and not done:
        qpos_full, info = predict_qpos14_chunk_with_feature(
            model,
            obs,
            action_head_hidden_chunk_len=args.gate_chunk_len,
        )
        gate_action_chunk = qpos_full[:, : args.gate_chunk_len, :].contiguous()
        if gate_action_chunk.shape[1] != args.gate_chunk_len:
            raise ValueError(f"gate action chunk must have length {args.gate_chunk_len}.")
        qpos_exec_chunk = qpos_full[:, : args.chunk_len, :].contiguous()
        z_t = info["action_head_hidden"]
        gate_logit, gate_prob = compute_gate(gate_runtime, z_t, gate_action_chunk)
        pregrasp_result = detect_pregrasp_trigger(
            qpos_exec_chunk.detach().cpu().numpy(),
            env_step=env_steps,
            config=pregrasp_config,
            last_trigger_step=last_pregrasp_trigger_step,
        )
        execution_plan = decide_execution_plan(
            execution_mode=args.execution_mode,
            gate_prob=gate_prob,
            gate_threshold=args.gate_threshold,
            pregrasp_triggered=pregrasp_result.triggered,
            enable_pregrasp_intervention=getattr(args, "enable_pregrasp_intervention", False),
            residual_horizon_k=args.residual_horizon_k,
            target_horizon_offset=args.residual_target_horizon_offset,
            action_chunk_len=qpos_exec_chunk.shape[1],
            handover_interventions_so_far=count_stage_interventions(rows, "handover"),
            max_handover_interventions_per_episode=args.max_handover_interventions_per_episode,
        )
        intervention_stage = execution_plan.intervention_stage_name
        gate_binary = execution_plan.gate_binary
        handover_gate_binary = bool(
            handover_intervention_enabled(args) and gate_prob >= args.gate_threshold
        )
        ee16_binary = execution_plan.ee16_binary
        execution_mode = execution_plan.execution_mode
        env_step_start = env_steps

        env.robotwin_action_mode = execution_plan.env_action_mode
        env.ee16_execution_strategy = args.ee16_execution_strategy
        fk_time_s = 0.0
        residual_norm = 0.0
        right_xyz_movement_norm = 0.0
        intervention_meta: dict[str, object] = {}
        if ee16_binary:
            active_runner = (
                intervention_runner
                if execution_plan.stage == InterventionStage.HANDOVER
                else pregrasp_runner
            )
            if active_runner is None:
                raise RuntimeError("EE intervention mode requires a residual intervention runner.")
            if execution_plan.stage == InterventionStage.PREGRASP:
                last_pregrasp_trigger_step = env_step_start
            fk_start = time.perf_counter()
            intervention = active_runner.run(
                qpos_exec_chunk.to(args.device),
                gate_score=gate_prob,
                gate_threshold=args.gate_threshold,
                episode_id=episode_id,
                env_step=env_step_start,
                intervention_id=replan_id,
                intervention_stage=intervention_stage,
                trigger_source=execution_plan.runtime_trigger_source,
                trigger_metadata=pregrasp_metadata(pregrasp_result)
                if execution_plan.stage == InterventionStage.PREGRASP
                else {},
            )
            fk_time_s = time.perf_counter() - fk_start
            env_action_chunk = intervention.exec_ee16_chunk.detach().cpu().numpy()
            intervention_records.extend(intervention.records)
            intervention_meta = intervention.metadata
            residual_norm = float(
                intervention.exec_ee16_chunk[..., 8:11]
                .sub(intervention.base_ee16_chunk[..., 8:11])
                .norm()
                .detach()
                .cpu()
            )
            right_xyz_movement_norm = chunk_right_xyz_movement_norm(intervention.base_ee16_chunk)
        else:
            env_action_chunk = qpos_exec_chunk.detach().cpu().numpy()

        env_start = time.perf_counter()
        obs_list, rewards, terms, truncs, infos_list, env_debug = execute_env_action_chunk(
            env,
            env_action_chunk,
            frames,
            args,
        )
        env_time_s = time.perf_counter() - env_start
        obs = obs_list[-1]
        rewards_np = to_numpy(rewards).astype(np.float64)
        terms_np = to_numpy(terms).astype(bool)
        truncs_np = to_numpy(truncs).astype(bool)
        info_last = infos_list[-1] if infos_list else {}
        reward_sum = float(rewards_np.sum())
        episode_return += reward_sum
        env_steps += int(rewards_np.shape[1])
        done = bool(np.logical_or(terms_np, truncs_np).any())
        success = read_success(task, info_last, done)
        pose = read_pose_debug(task)
        row = build_hybrid_row(
            episode_id=episode_id,
            replan_id=replan_id,
            env_step_start=env_step_start,
            env_step_end=env_steps,
            seed=seed,
            gate_logit=gate_logit,
            gate_prob=gate_prob,
            gate_binary=gate_binary,
            handover_gate_binary=handover_gate_binary,
            pregrasp_result=pregrasp_result,
            intervention_stage=intervention_stage,
            execution_mode=execution_mode,
            args=args,
            qpos_chunk=qpos_exec_chunk,
            state14=extract_state14(obs),
            reward=reward_sum,
            done=done,
            success=success,
            pose=pose,
            fk_time_s=fk_time_s,
            env_time_s=env_time_s,
            env_debug=env_debug,
            residual_norm=residual_norm,
            right_xyz_movement_norm=right_xyz_movement_norm,
            intervention_meta=intervention_meta,
        )
        rows.append(row)
        print(
            f"episode={episode_id} replan={replan_id} env_steps={env_steps} "
            f"mode={execution_mode} stage={intervention_stage} gate_prob={gate_prob:.6g} "
            f"pregrasp_score={pregrasp_result.score:.6g} return={episode_return:.6g} success={success}"
        )
        replan_id += 1

    success = bool(getattr(task, "eval_success", False))
    log_path = save_hybrid_log(Path(args.save_dir), episode_id, rows)
    record_path = save_intervention_records(Path(args.save_dir), episode_id, intervention_records)
    plot_path = (
        save_hybrid_plot(Path(args.save_dir), episode_id, rows, args.gate_threshold)
        if args.save_gate_plots
        else None
    )
    video_path = None
    if args.save_video:
        video_path = Path(args.save_dir) / "videos" / f"episode_{episode_id:04d}_seed_{seed}.mp4"
        save_video(frames, video_path, args.video_fps)
    if args.save_debug:
        write_json(
            Path(args.save_dir) / f"episode_{episode_id:04d}_debug.json",
            {"rows": rows, "intervention_records": intervention_records},
        )

    probs = np.asarray([row["gate_prob"] for row in rows], dtype=np.float32)
    residual_stats = residual_prediction_stats(intervention_records)
    summary = {
        "episode_id": episode_id,
        "seed": seed,
        "success": success,
        "return": episode_return,
        "episode_length": env_steps,
        "replan_steps": replan_id,
        "wall_time_s": time.perf_counter() - start,
        "gate_prob_max": float(probs.max()) if probs.size else 0.0,
        "gate_prob_mean": float(probs.mean()) if probs.size else 0.0,
        "gate_first_activation_env_step": first_activation(rows),
        "first_gate_step": first_activation(rows, field="env_step_start"),
        "first_intervention_step": first_intervention(rows, field="env_step_start"),
        "num_interventions": sum(1 for row in rows if row["execution_mode"] == "ee16_zero_residual"),
        "total_ee_intervention_steps": int(sum(row["num_intervention_steps"] for row in rows)),
        "first_pregrasp_trigger_step": first_stage_intervention(rows, "pregrasp", field="env_step_start"),
        "first_handover_trigger_step": first_stage_intervention(rows, "handover", field="env_step_start"),
        "num_pregrasp_interventions": count_stage_interventions(rows, "pregrasp"),
        "num_handover_interventions": count_stage_interventions(rows, "handover"),
        "total_pregrasp_ee_steps": total_stage_ee_steps(rows, "pregrasp"),
        "total_handover_ee_steps": total_stage_ee_steps(rows, "handover"),
        "pred_norm_mean": residual_stats["pred_norm_mean"],
        "pred_norm_std": residual_stats["pred_norm_std"],
        "pred_norm_max": residual_stats["pred_norm_max"],
        "pred_norm_p50": residual_stats["pred_norm_p50"],
        "pred_norm_p90": residual_stats["pred_norm_p90"],
        "pred_norm_p95": residual_stats["pred_norm_p95"],
        "pred_norm_p99": residual_stats["pred_norm_p99"],
        "applied_norm_mean": residual_stats["applied_norm_mean"],
        "applied_norm_std": residual_stats["applied_norm_std"],
        "applied_norm_max": residual_stats["applied_norm_max"],
        "applied_norm_p50": residual_stats["applied_norm_p50"],
        "applied_norm_p90": residual_stats["applied_norm_p90"],
        "applied_norm_p95": residual_stats["applied_norm_p95"],
        "applied_norm_p99": residual_stats["applied_norm_p99"],
        "saturation_ratio": residual_stats["saturation_ratio"],
        "nan_inf_count": residual_stats["nan_inf_count"],
        "dry_run": resolve_residual_dry_run(args),
        "learned_residual_control_enabled": learned_residual_control_enabled(args),
        "residual_scale": args.residual_scale,
        "residual_actor_checkpoint": args.residual_actor_checkpoint,
        "selected_action_chunk_indices": [
            row["selected_indices"] for row in rows if row["execution_mode"] == "ee16_zero_residual"
        ],
        "failure_reason": failure_reason(success, done, env_steps, args.max_steps),
        "simulator_crash": False,
        "execution_mode_timeline": [row["execution_mode"] for row in rows],
        "intervention_stage_timeline": [row["intervention_stage"] for row in rows],
        "hybrid_log_path": str(log_path),
        "intervention_records_path": str(record_path),
        "gate_plot_path": str(plot_path) if plot_path is not None else None,
        "video_path": str(video_path) if video_path is not None else None,
    }
    print(f"EPISODE SUMMARY {json.dumps(summary, sort_keys=True)}")
    return summary


def compute_gate(
    gate_runtime: ChunkAwareGateRuntime | None,
    z_t: torch.Tensor,
    action_chunk: torch.Tensor,
) -> tuple[float, float]:
    if gate_runtime is None:
        return 0.0, 0.0
    logit = gate_runtime.logits_from_inputs(z_t, action_chunk)
    prob = torch.sigmoid(logit)
    return float(logit.detach().cpu().reshape(-1)[0]), float(prob.detach().cpu().reshape(-1)[0])


def execute_env_action_chunk(
    env: Any,
    env_action_chunk: np.ndarray,
    frames: list[np.ndarray],
    args: argparse.Namespace,
):
    env_debug: dict[str, Any] = {}
    obs_list, rewards, terms, truncs, infos_list = env.chunk_step(env_action_chunk)
    if infos_list and "ee16_chunk_debug" in infos_list[-1]:
        env_debug = dict(infos_list[-1]["ee16_chunk_debug"])
    if args.save_video:
        if args.video_frame_mode == "executed_step":
            for step_obs in obs_list:
                append_frame(frames, env, step_obs, args.video_source)
        elif obs_list:
            append_frame(frames, env, obs_list[-1], args.video_source)
    return obs_list, rewards, terms, truncs, infos_list, env_debug


def build_hybrid_row(**kwargs: Any) -> dict[str, Any]:
    args = kwargs["args"]
    qpos_chunk = kwargs["qpos_chunk"]
    left_gripper = qpos_chunk[0, :, 6].detach().cpu()
    right_gripper = qpos_chunk[0, :, 13].detach().cpu()
    pregrasp_result: PregraspTriggerResult = kwargs["pregrasp_result"]
    row = {
        "episode_id": kwargs["episode_id"],
        "replan_id": kwargs["replan_id"],
        "env_step": kwargs["env_step_end"],
        "env_step_start": kwargs["env_step_start"],
        "env_step_end": kwargs["env_step_end"],
        "seed": kwargs["seed"],
        "gate_logit": kwargs["gate_logit"],
        "gate_prob": kwargs["gate_prob"],
        "gate_binary": int(kwargs["gate_binary"]),
        "handover_gate_binary": int(kwargs["handover_gate_binary"]),
        "pregrasp_trigger_binary": int(pregrasp_result.triggered),
        "pregrasp_trigger_score": pregrasp_result.score,
        "first_closing_chunk_index": pregrasp_result.first_closing_chunk_index,
        "left_gripper_start": pregrasp_result.left_gripper_start,
        "left_gripper_end": pregrasp_result.left_gripper_end,
        "left_gripper_min": pregrasp_result.left_gripper_min,
        "left_gripper_max": pregrasp_result.left_gripper_max,
        "pregrasp_trigger_reason": pregrasp_result.trigger_reason,
        "pregrasp_trigger_in_cooldown": int(pregrasp_result.in_cooldown),
        "execution_mode": kwargs["execution_mode"],
        "intervention_stage": kwargs["intervention_stage"],
        "trigger_source": kwargs["intervention_meta"].get("trigger_source", ""),
        "ee16_execution_strategy": args.ee16_execution_strategy,
        "residual_actor": args.residual_actor,
        "residual_dry_run": resolve_residual_dry_run(args),
        "learned_residual_control_enabled": learned_residual_control_enabled(args),
        "residual_scale": args.residual_scale,
        "residual_horizon_k": args.residual_horizon_k,
        "residual_target_horizon_offset": args.residual_target_horizon_offset,
        "residual_max_delta_local_xyz": args.residual_max_delta_local_xyz,
        "qpos_action_norm": float(qpos_chunk.norm().detach().cpu()),
        "action_chunk_left_gripper_min": float(left_gripper.min()),
        "action_chunk_left_gripper_max": float(left_gripper.max()),
        "action_chunk_left_gripper_mean": float(left_gripper.mean()),
        "action_chunk_right_gripper_min": float(right_gripper.min()),
        "action_chunk_right_gripper_max": float(right_gripper.max()),
        "action_chunk_right_gripper_mean": float(right_gripper.mean()),
        "fk_time_s": kwargs["fk_time_s"],
        "ee16_execution_time_s": kwargs["env_time_s"] if kwargs["execution_mode"] == "ee16_zero_residual" else 0.0,
        "take_action_calls": int(kwargs["env_debug"].get("take_action_calls", 0)),
        "right_xyz_movement_norm": kwargs["right_xyz_movement_norm"],
        "residual_norm": kwargs["residual_norm"],
        "reward": kwargs["reward"],
        "done": int(kwargs["done"]),
        "success": int(kwargs["success"]),
        "selected_indices": kwargs["intervention_meta"].get("selected_indices"),
        "num_intervention_steps": int(kwargs["intervention_meta"].get("num_steps_executed", 0)),
        "max_delta_norm": float(kwargs["intervention_meta"].get("max_delta_norm", 0.0)),
        "mean_delta_norm": float(kwargs["intervention_meta"].get("mean_delta_norm", 0.0)),
        "pred_delta_norm_mean": float(kwargs["intervention_meta"].get("pred_delta_norm_mean", 0.0)),
        "pred_delta_norm_max": float(kwargs["intervention_meta"].get("pred_delta_norm_max", 0.0)),
        "applied_delta_norm_mean": float(kwargs["intervention_meta"].get("applied_delta_norm_mean", 0.0)),
        "applied_delta_norm_max": float(kwargs["intervention_meta"].get("applied_delta_norm_max", 0.0)),
        "saturation_count": int(kwargs["intervention_meta"].get("saturation_count", 0)),
        "nan_inf_count": int(kwargs["intervention_meta"].get("nan_inf_count", 0)),
        "left_stabilization_count": kwargs["intervention_meta"].get("left_stabilization_count", 0),
        "simulator_crash": 0,
        **kwargs["pose"],
    }
    state14 = np.asarray(kwargs["state14"], dtype=np.float32).reshape(-1)
    row.update({f"obs_state_{idx:02d}": float(value) for idx, value in enumerate(state14)})
    return row


def read_pose_debug(task: Any) -> dict[str, float | None]:
    try:
        pose16 = read_pose16(task, read_task_qpos14(task))
        left = pose16[0:3]
        right = pose16[8:11]
        return {
            "left_ee_x": float(left[0]),
            "left_ee_y": float(left[1]),
            "left_ee_z": float(left[2]),
            "right_ee_x": float(right[0]),
            "right_ee_y": float(right[1]),
            "right_ee_z": float(right[2]),
            "d_LR": float(np.linalg.norm(left - right)),
        }
    except Exception:  # noqa: BLE001
        return {
            "left_ee_x": None,
            "left_ee_y": None,
            "left_ee_z": None,
            "right_ee_x": None,
            "right_ee_y": None,
            "right_ee_z": None,
            "d_LR": None,
        }


def read_success(task: Any, info: dict[str, Any], done: bool) -> bool:
    if "success" in info:
        return bool(np.any(to_numpy(info["success"])))
    return bool(done and getattr(task, "eval_success", False))


def chunk_right_xyz_movement_norm(endpose16_chunk: torch.Tensor) -> float:
    right_xyz = endpose16_chunk[..., 8:11]
    if right_xyz.shape[1] <= 1:
        return 0.0
    return float((right_xyz[:, 1:] - right_xyz[:, :-1]).norm(dim=-1).sum().detach().cpu())


def first_activation(rows: list[dict[str, Any]], field: str = "env_step") -> int | None:
    for row in rows:
        if row["gate_binary"]:
            return int(row[field])
    return None


def first_intervention(rows: list[dict[str, Any]], field: str = "env_step") -> int | None:
    for row in rows:
        if row["execution_mode"] == "ee16_zero_residual":
            return int(row[field])
    return None


def first_stage_intervention(rows: list[dict[str, Any]], stage: str, field: str = "env_step") -> int | None:
    for row in rows:
        if row.get("intervention_stage") == stage and row["execution_mode"] == "ee16_zero_residual":
            return int(row[field])
    return None


def count_stage_interventions(rows: list[dict[str, Any]], stage: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("intervention_stage") == stage and row["execution_mode"] == "ee16_zero_residual"
    )


def total_stage_ee_steps(rows: list[dict[str, Any]], stage: str) -> int:
    return int(
        sum(
            row["num_intervention_steps"]
            for row in rows
            if row.get("intervention_stage") == stage and row["execution_mode"] == "ee16_zero_residual"
        )
    )


def pregrasp_metadata(result: PregraspTriggerResult) -> dict[str, object]:
    return {
        "pregrasp_trigger_score": result.score,
        "first_closing_chunk_index": result.first_closing_chunk_index,
        "left_gripper_start": result.left_gripper_start,
        "left_gripper_end": result.left_gripper_end,
        "left_gripper_min": result.left_gripper_min,
        "left_gripper_max": result.left_gripper_max,
        "trigger_reason": result.trigger_reason,
    }


def failure_reason(success: bool, done: bool, episode_length: int, max_steps: int) -> str | None:
    if success:
        return None
    if episode_length >= max_steps:
        return "timeout"
    if done:
        return "terminated_without_success"
    return "stopped_without_success"


def print_context(args: argparse.Namespace) -> None:
    print("Runtime Context")
    print("---------------")
    for key, value in vars(args).items():
        print(f"{key}={value}")
    print(f"resolved_residual_dry_run={resolve_residual_dry_run(args)}")
    print(f"resolved_learned_residual_control_enabled={learned_residual_control_enabled(args)}")
    for name in ("REPO_PATH", "ROBOTWIN_PATH", "ROBOTWIN_ASSETS_PATH", "ROBOT_PLATFORM", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


def resolve_residual_control_flags(args: argparse.Namespace) -> dict[str, bool]:
    return _resolve_residual_control_flags(args)


def build_pregrasp_trigger_config(args: argparse.Namespace) -> PregraspTriggerConfig:
    close_threshold = getattr(args, "gripper_close_threshold", None)
    if close_threshold is None:
        close_threshold = getattr(args, "left_gripper_closed_value", None)
    return PregraspTriggerConfig(
        enabled=pregrasp_intervention_enabled(args),
        close_direction=getattr(args, "gripper_close_direction", None),
        close_threshold=close_threshold,
        closing_delta_threshold=getattr(args, "gripper_closing_delta_threshold", 0.15),
        cooldown_steps=getattr(args, "pregrasp_cooldown_steps", 50),
    )


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
