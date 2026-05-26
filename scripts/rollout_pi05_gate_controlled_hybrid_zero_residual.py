"""Roll out pi05 with v2 gate-controlled qpos14/ee16 zero-residual switching."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.residual_td3.fk_bridge import AlohaFKBridge
from rlinf.algorithms.residual_td3.gate_head import ChunkAwareGateRuntime
from rlinf.algorithms.residual_td3.residual_ee_intervention import (
    ConstantResidualActor,
    ResidualEEInterventionConfig,
    ResidualEEInterventionRunner,
    ZeroInitResidualActor,
    ZeroResidualActor,
)
from scripts.rollout_pi05_fk_ee16_zero_residual import (
    DEFAULT_CONFIG as DEFAULT_ENV_CONFIG,
)
from scripts.rollout_pi05_fk_ee16_zero_residual import (
    append_frame,
    build_actor_model_cfg,
    close_env,
    get_single_task,
    load_env_cfg,
    load_model,
    make_env,
    read_pose16,
    read_task_qpos14,
    save_video,
    select_episode_seeds,
    to_numpy,
)
from scripts.rollout_pi05_with_gate_logging import (
    DEFAULT_NORM_STATS,
    extract_state14,
    predict_qpos14_chunk_with_feature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--env-config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--checkpoint", required=True)
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
        choices=("qpos14_baseline", "gate_controlled_hybrid"),
        default="gate_controlled_hybrid",
    )
    parser.add_argument(
        "--enable-residual-intervention",
        "--enable_residual_intervention",
        dest="enable_residual_intervention",
        action="store_true",
    )
    parser.add_argument(
        "--residual-actor",
        "--residual_actor",
        dest="residual_actor",
        choices=("zero", "constant", "zero_init"),
        default="zero",
    )
    parser.add_argument(
        "--residual-constant-delta-local-xyz",
        "--residual_constant_delta_local_xyz",
        dest="residual_constant_delta_local_xyz",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--residual-horizon-k", "--residual_horizon_k", dest="residual_horizon_k", type=int, default=5)
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
        default=0.02,
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
    parser.add_argument("--save-dir", required=True)
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
    parser.add_argument("--debug-ee16-timing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_context(args)
    env = None
    try:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        env_args = build_env_args(args)
        model_args = build_model_args(args)
        env_cfg = load_env_cfg(env_args)
        if args.execution_mode == "gate_controlled_hybrid" and hasattr(env_cfg.task_config, "planner_backend"):
            env_cfg.task_config.planner_backend = args.planner_backend
        episode_seeds = select_episode_seeds(env_cfg, env_args)
        gate_runtime = load_gate_runtime(args)
        env = make_env(env_cfg)
        env.debug_ee16_timing = bool(args.debug_ee16_timing)
        env.ee16_execution_strategy = args.ee16_execution_strategy
        model = load_model(build_actor_model_cfg(model_args), args.device)
        intervention_runner = None
        if args.execution_mode == "gate_controlled_hybrid":
            task = get_single_task_after_reset(env)
            intervention_runner = build_intervention_runner(task, args)
        metrics = run_episodes(env, model, gate_runtime, intervention_runner, args, episode_seeds)
        write_json(save_dir / "summary.json", metrics)
        print_summary(metrics)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"HYBRID ROLLOUT: FAIL {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().strip().splitlines()[-24:]:
            print(f"  {line}")
        return 1
    finally:
        if env is not None:
            close_env(env)


def build_env_args(args: argparse.Namespace) -> SimpleNamespace:
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
    )


def build_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats_path,
        model_num_action_chunks=args.model_num_action_chunks,
        config_name=args.config,
        num_images_in_input=args.num_images_in_input,
        noise_level=args.noise_level,
    )


def load_gate_runtime(args: argparse.Namespace) -> ChunkAwareGateRuntime | None:
    if args.execution_mode == "qpos14_baseline" and args.chunk_aware_gate_checkpoint is None:
        return None
    if not args.chunk_aware_gate_checkpoint:
        raise ValueError("--chunk-aware-gate-checkpoint is required for gate-controlled hybrid rollout.")
    return ChunkAwareGateRuntime.load_from_checkpoint(
        args.chunk_aware_gate_checkpoint,
        device=args.device,
        threshold=args.gate_threshold,
    )


def build_intervention_runner(task: Any, args: argparse.Namespace) -> ResidualEEInterventionRunner:
    if args.residual_actor == "zero":
        actor = ZeroResidualActor()
    elif args.residual_actor == "zero_init":
        actor = ZeroInitResidualActor(
            chunk_len=args.residual_horizon_k,
            delta_max=args.residual_max_delta_local_xyz,
            device=args.device,
        )
    else:
        actor = ConstantResidualActor(tuple(float(v) for v in args.residual_constant_delta_local_xyz))
    cfg = ResidualEEInterventionConfig(
        horizon_k=args.residual_horizon_k,
        target_horizon_offset=args.residual_target_horizon_offset,
        max_delta_local_xyz=args.residual_max_delta_local_xyz,
        left_stabilization_mode=args.left_stabilization_mode,
        left_deadband_xyz=args.left_deadband_xyz,
        left_lowpass_alpha=args.left_lowpass_alpha,
    )
    return ResidualEEInterventionRunner(
        AlohaFKBridge.from_robotwin_task(task, device=args.device),
        actor,
        cfg,
    )


def get_single_task_after_reset(env: Any) -> Any:
    env.reset()
    return get_single_task(env)


def run_episodes(
    env: Any,
    model: Any,
    gate_runtime: ChunkAwareGateRuntime | None,
    intervention_runner: ResidualEEInterventionRunner | None,
    args: argparse.Namespace,
    episode_seeds: list[int],
) -> dict[str, Any]:
    save_dir = Path(args.save_dir)
    summaries = []
    for episode_id, seed in enumerate(episode_seeds):
        summary = run_episode(env, model, gate_runtime, intervention_runner, args, episode_id, seed)
        summaries.append(summary)
        write_json(save_dir / "summary.json", {"episodes": summaries})
    successes = [row["success"] for row in summaries]
    returns = [row["return"] for row in summaries]
    return {
        "execution_mode": args.execution_mode,
        "ee16_execution_strategy": args.ee16_execution_strategy,
        "gate_type": args.gate_type,
        "gate_threshold": args.gate_threshold,
        "gate_chunk_len": args.gate_chunk_len,
        "chunk_aware_gate_checkpoint": args.chunk_aware_gate_checkpoint,
        "enable_residual_intervention": bool(args.enable_residual_intervention),
        "residual_actor": args.residual_actor,
        "residual_horizon_k": args.residual_horizon_k,
        "residual_target_horizon_offset": args.residual_target_horizon_offset,
        "residual_max_delta_local_xyz": args.residual_max_delta_local_xyz,
        "chunk_len": args.chunk_len,
        "model_num_action_chunks": args.model_num_action_chunks,
        "max_steps": args.max_steps,
        "requested_num_episodes": args.num_episodes,
        "rollout_status": "completed",
        "simulator_crash": False,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "episodes": summaries,
        "save_dir": str(save_dir),
    }


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
        gate_binary = bool(args.execution_mode == "gate_controlled_hybrid" and gate_prob >= args.gate_threshold)
        execution_mode = "ee16_zero_residual" if gate_binary else "qpos14"
        env_step_start = env_steps

        env.robotwin_action_mode = "ee16" if gate_binary else "qpos14"
        env.ee16_execution_strategy = args.ee16_execution_strategy
        fk_time_s = 0.0
        residual_norm = 0.0
        right_xyz_movement_norm = 0.0
        intervention_meta: dict[str, object] = {}
        if gate_binary:
            if intervention_runner is None:
                raise RuntimeError("gate-controlled hybrid mode requires a residual intervention runner.")
            fk_start = time.perf_counter()
            intervention = intervention_runner.run(
                qpos_exec_chunk.to(args.device),
                gate_score=gate_prob,
                gate_threshold=args.gate_threshold,
                episode_id=episode_id,
                env_step=env_step_start,
                intervention_id=replan_id,
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
            f"mode={execution_mode} gate_prob={gate_prob:.6g} return={episode_return:.6g} success={success}"
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
        "selected_action_chunk_indices": [
            row["selected_indices"] for row in rows if row["execution_mode"] == "ee16_zero_residual"
        ],
        "failure_reason": failure_reason(success, done, env_steps, args.max_steps),
        "simulator_crash": False,
        "execution_mode_timeline": [row["execution_mode"] for row in rows],
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
    right_gripper = qpos_chunk[0, :, 13].detach().cpu()
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
        "execution_mode": kwargs["execution_mode"],
        "ee16_execution_strategy": args.ee16_execution_strategy,
        "residual_actor": args.residual_actor,
        "residual_horizon_k": args.residual_horizon_k,
        "residual_target_horizon_offset": args.residual_target_horizon_offset,
        "residual_max_delta_local_xyz": args.residual_max_delta_local_xyz,
        "qpos_action_norm": float(qpos_chunk.norm().detach().cpu()),
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


def failure_reason(success: bool, done: bool, episode_length: int, max_steps: int) -> str | None:
    if success:
        return None
    if episode_length >= max_steps:
        return "timeout"
    if done:
        return "terminated_without_success"
    return "stopped_without_success"


def save_hybrid_log(save_dir: Path, episode_id: int, rows: list[dict[str, Any]]) -> Path:
    path = save_dir / f"hybrid_log_episode_{episode_id:04d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)
    csv_path = path.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path


def save_intervention_records(save_dir: Path, episode_id: int, records: list[dict[str, object]]) -> Path:
    path = save_dir / f"intervention_records_episode_{episode_id:04d}.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(records) if records else pa.table({}), path)
    return path


def save_hybrid_plot(save_dir: Path, episode_id: int, rows: list[dict[str, Any]], threshold: float) -> Path | None:
    if not rows:
        return None
    import matplotlib.pyplot as plt

    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"hybrid_plot_episode_{episode_id:04d}.png"
    x = np.asarray([row["env_step"] for row in rows], dtype=np.int64)
    gate_prob = np.asarray([row["gate_prob"] for row in rows], dtype=np.float32)
    gate = np.asarray([row["gate_binary"] for row in rows], dtype=np.float32)
    mode = np.asarray([1.0 if row["execution_mode"] == "ee16_zero_residual" else 0.0 for row in rows])
    left_gripper = np.asarray([row["obs_state_06"] for row in rows], dtype=np.float32)
    right_gripper = np.asarray([row["obs_state_13"] for row in rows], dtype=np.float32)
    dist = np.asarray([np.nan if row["d_LR"] is None else row["d_LR"] for row in rows], dtype=np.float32)
    success = np.asarray([row["success"] for row in rows], dtype=np.float32)
    done = np.asarray([row["done"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, gate_prob, label="gate_prob")
    axes[0].axhline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:g}")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="upper right")
    axes[1].step(x, gate, where="post", label="gate_binary")
    axes[1].step(x, mode, where="post", label="ee16_zero_residual")
    axes[1].legend(loc="upper right")
    axes[2].plot(x, left_gripper, label="left_gripper")
    axes[2].plot(x, right_gripper, label="right_gripper")
    axes[2].legend(loc="upper right")
    axes[3].plot(x, dist, label="d_LR")
    axes[3].scatter(x[done > 0], done[done > 0], marker="x", label="done")
    axes[3].scatter(x[success > 0], success[success > 0], marker="o", label="success")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("env_step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_context(args: argparse.Namespace) -> None:
    print("Runtime Context")
    print("---------------")
    for key, value in vars(args).items():
        print(f"{key}={value}")
    for name in ("REPO_PATH", "ROBOTWIN_PATH", "ROBOT_PLATFORM", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


def print_summary(metrics: dict[str, Any]) -> None:
    print("HYBRID SUMMARY")
    print("--------------")
    for key in ("execution_mode", "ee16_execution_strategy", "success_rate", "mean_return", "save_dir"):
        print(f"{key}={metrics[key]}")


if __name__ == "__main__":
    raise SystemExit(main())
