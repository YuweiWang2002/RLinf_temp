"""Roll out pi05 qpos14 policy with GateHead logging only.

This script executes the original pi05 qpos14 action chunk unchanged. The
GateHead path is a side channel: pi05 exports ``action_head_hidden`` only when
requested here, and the gate prediction is logged without modifying actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.residual_td3.gate_head import (
    ChunkAwareGateRuntime,
    GateHeadRuntime,
)
from scripts.rollout_pi05_fk_ee16_zero_residual import (
    DEFAULT_CONFIG as DEFAULT_ENV_CONFIG,
)
from scripts.rollout_pi05_fk_ee16_zero_residual import (
    append_frame,
    build_actor_model_cfg,
    close_env,
    load_env_cfg,
    load_model,
    make_env,
    read_pose16,
    read_task_qpos14,
    save_video,
    select_episode_seeds,
    to_numpy,
)

DEFAULT_NORM_STATS = "/tmp/tmp_w6y6w0rl/data/pytorch_checkpoint/handover_expert/norm_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--env-config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats-path", default=DEFAULT_NORM_STATS)
    parser.add_argument("--gate-head-checkpoint", default=None)
    parser.add_argument("--chunk-aware-gate-checkpoint", default=None)
    parser.add_argument("--gate-mode", choices=("z_only", "chunk_aware"), default="z_only")
    parser.add_argument("--gate-threshold", type=float, default=0.8)
    parser.add_argument("--gate-feature-action-chunk", type=int, default=None)
    parser.add_argument("--gate-chunk-len", type=int, default=50)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument("--model-num-action-chunks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100100000)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--use-eval-success-seeds", action="store_true")
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--env-split", choices=("train", "eval"), default="eval")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--diagnostics-root", default="logs/gate_online_diagnostics")
    parser.add_argument("--diagnostics-run-name", default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-gate-plots", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-source", choices=("auto", "third_view", "observer", "main"), default="third_view")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-images-in-input", type=int, default=3)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--rule-distance-threshold", type=float, default=0.294)
    parser.add_argument("--rule-motion-threshold", type=float, default=1e-3)
    parser.add_argument("--rule-gripper-closed-threshold", type=float, default=0.5)
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
        episode_seeds = select_episode_seeds(env_cfg, env_args)
        env = make_env(env_cfg)
        model = load_model(build_actor_model_cfg(model_args), args.device)
        gate_runtime = load_gate_runtime(args)
        metrics = run_episodes(env, model, gate_runtime, args, episode_seeds)
        write_summary(save_dir / "summary.json", metrics)
        print_summary(metrics)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"GATE ROLLOUT: FAIL {type(exc).__name__}: {exc}")
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


def load_gate_runtime(args: argparse.Namespace) -> GateHeadRuntime | ChunkAwareGateRuntime:
    if args.gate_mode == "z_only":
        if not args.gate_head_checkpoint:
            raise ValueError("--gate-head-checkpoint is required for --gate-mode z_only.")
        return GateHeadRuntime.load_from_checkpoint(
            args.gate_head_checkpoint,
            device=args.device,
            threshold=args.gate_threshold,
        )
    if not args.chunk_aware_gate_checkpoint:
        raise ValueError("--chunk-aware-gate-checkpoint is required for --gate-mode chunk_aware.")
    return ChunkAwareGateRuntime.load_from_checkpoint(
        args.chunk_aware_gate_checkpoint,
        device=args.device,
        threshold=args.gate_threshold,
    )


def run_episodes(
    env: Any,
    model: Any,
    gate_runtime: GateHeadRuntime | ChunkAwareGateRuntime,
    args: argparse.Namespace,
    episode_seeds: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_features: list[dict[str, Any]] = []
    save_dir = Path(args.save_dir)
    diagnostics_dir = build_diagnostics_dir(args)
    for episode_id, seed in enumerate(episode_seeds):
        summary, rows, feature_rows = run_episode(env, model, gate_runtime, args, episode_id, seed)
        summaries.append(summary)
        diagnostic_rows.extend(rows)
        diagnostic_features.extend(feature_rows)
        save_online_diagnostics(diagnostics_dir, diagnostic_rows, diagnostic_features)
        write_summary(save_dir / "summary.json", {"episodes": summaries})
    return {"episodes": summaries, "save_dir": str(save_dir), "diagnostics_dir": str(diagnostics_dir)}


def run_episode(
    env: Any,
    model: Any,
    gate_runtime: GateHeadRuntime | ChunkAwareGateRuntime,
    args: argparse.Namespace,
    episode_id: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    obs, _ = env.reset(env_seeds=[seed])
    task = get_task(env)
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    append_frame(frames, env, obs, args.video_source)
    episode_return = 0.0
    env_steps = 0
    replan_step = 0
    success = False
    done = False
    start = time.perf_counter()

    while env_steps < args.max_steps and not done:
        feature_frame_index = env_steps
        feature_video_frame_index = len(frames) - 1
        hidden_chunk_len = (
            args.gate_chunk_len
            if args.gate_mode == "chunk_aware"
            else args.gate_feature_action_chunk
        )
        qpos_chunk, info = predict_qpos14_chunk_with_feature(
            model,
            obs,
            action_head_hidden_chunk_len=hidden_chunk_len,
        )
        gate_action_chunk = qpos_chunk[:, : args.gate_chunk_len, :].contiguous()
        if args.gate_mode == "chunk_aware" and gate_action_chunk.shape[1] != args.gate_chunk_len:
            raise ValueError(
                "chunk-aware gate requires predicted action chunk with shape "
                f"[1, {args.gate_chunk_len}, 14], got {tuple(gate_action_chunk.shape)}."
            )
        qpos_chunk = qpos_chunk[:, : args.chunk_len, :].contiguous()
        z_t = info["action_head_hidden"]
        gate_logit = compute_gate_logit(gate_runtime, args, z_t, gate_action_chunk)
        gate_prob = torch.sigmoid(gate_logit)
        gate_binary = (gate_prob >= gate_runtime.cfg.threshold).to(dtype=torch.float32)
        z_norm = float(z_t.detach().to(dtype=torch.float32).norm(dim=-1).cpu().reshape(-1)[0])
        gate_logit_value = float(gate_logit.detach().cpu().reshape(-1)[0])
        gate_prob_value = float(gate_prob.detach().cpu().reshape(-1)[0])
        gate_binary_value = float(gate_binary.detach().cpu().reshape(-1)[0])
        state14 = extract_state14(obs)

        pose_debug = read_pose_debug(task)
        action_np = qpos_chunk.detach().cpu().numpy()
        obs_list, rewards, terms, truncs, infos_list = env.chunk_step(action_np)
        if args.save_video:
            for step_obs in obs_list:
                append_frame(frames, env, step_obs, args.video_source)
        obs = obs_list[-1]
        rewards_np = to_numpy(rewards).astype(np.float64)
        terms_np = to_numpy(terms).astype(bool)
        truncs_np = to_numpy(truncs).astype(bool)
        info_last = infos_list[-1] if infos_list else {}
        reward_sum = float(rewards_np.sum())
        episode_return += reward_sum
        env_steps += int(rewards_np.shape[1])
        done = bool(np.logical_or(terms_np, truncs_np).any())
        success = success or read_success(task, info_last, done)
        rows.append(
            build_log_row(
                episode_id=episode_id,
                step_id=replan_step,
                frame_index=feature_frame_index,
                video_frame_index=feature_video_frame_index,
                z_norm=z_norm,
                gate_logit=gate_logit_value,
                gate_prob=gate_prob_value,
                gate_binary=gate_binary_value,
                gate_feature_action_chunk=int(info.get("action_head_hidden_chunk_len", model.config.action_chunk)),
                gate_mode=args.gate_mode,
                gate_action_chunk=gate_action_chunk,
                state14=state14,
                qpos_action_chunk=qpos_chunk,
                reward=reward_sum,
                done=done,
                success=success,
                pose_debug=pose_debug,
            )
        )
        feature_rows.append(
            {
                "episode_id": episode_id,
                "step_id": replan_step,
                "frame_index": feature_frame_index,
                "video_frame_index": feature_video_frame_index,
                "z": z_t.detach().to(dtype=torch.float32).cpu().reshape(-1).numpy(),
                "state": state14,
                "gate_logit": gate_logit_value,
                "gate_prob": gate_prob_value,
                "z_norm": z_norm,
                "action_chunk": gate_action_chunk.detach().to(dtype=torch.float32).cpu().reshape(
                    args.gate_chunk_len,
                    14,
                ).numpy(),
            }
        )
        print(
            f"episode={episode_id} replan={replan_step} env_steps={env_steps} "
            f"return={episode_return:.6g} success={int(success)} "
            f"gate_prob={gate_prob_value:.6g} gate={int(gate_binary_value)}"
        )
        replan_step += 1

    annotate_rule_gate(rows, args)
    log_path = save_gate_log(Path(args.save_dir), episode_id, rows)
    plot_path = None
    if args.save_gate_plots and episode_id < 5:
        plot_path = save_gate_plot(Path(args.save_dir), episode_id, rows, args.gate_threshold)
    video_path = None
    if args.save_video:
        video_path = Path(args.save_dir) / "videos" / f"episode_{episode_id:04d}_seed_{seed}.mp4"
        save_video(frames, video_path, args.video_fps)

    probs = np.asarray([row["gate_prob"] for row in rows], dtype=np.float32)
    gates = np.asarray([row["gate_binary"] for row in rows], dtype=np.float32)
    summary = {
        "episode_id": episode_id,
        "seed": seed,
        "success": bool(success),
        "episode_length": env_steps,
        "return": episode_return,
        "replan_steps": replan_step,
        "wall_time_s": time.perf_counter() - start,
        "gate_prob_min": float(probs.min()) if probs.size else 0.0,
        "gate_prob_max": float(probs.max()) if probs.size else 0.0,
        "gate_prob_mean": float(probs.mean()) if probs.size else 0.0,
        "gate_activation_ratio_t08": float(gates.mean()) if gates.size else 0.0,
        "gate_log_path": str(log_path),
        "gate_plot_path": str(plot_path) if plot_path is not None else None,
        "video_path": str(video_path) if video_path is not None else None,
    }
    print(f"EPISODE SUMMARY {json.dumps(summary, sort_keys=True)}")
    return summary, rows, feature_rows


def predict_qpos14_chunk_with_feature(
    model: Any,
    obs: dict[str, Any],
    *,
    action_head_hidden_chunk_len: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    env_obs = dict(obs)
    env_obs.setdefault("extra_view_images", None)
    with torch.no_grad():
        actions, info = model.predict_action_batch(
            env_obs,
            mode="eval",
            compute_values=False,
            return_action_head_hidden=True,
            action_head_hidden_chunk_len=action_head_hidden_chunk_len,
        )
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    actions = actions.to(dtype=torch.float32)
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 14:
        raise ValueError(f"pi05 qpos14 action chunk must have shape [1, C, 14], got {actions.shape}.")
    z_t = info.get("action_head_hidden")
    if not torch.is_tensor(z_t):
        raise ValueError("predict_action_batch did not return action_head_hidden.")
    if z_t.ndim != 2:
        raise ValueError(f"action_head_hidden must have shape [B, Z], got {tuple(z_t.shape)}.")
    return actions.contiguous(), info


def compute_gate_logit(
    gate_runtime: GateHeadRuntime | ChunkAwareGateRuntime,
    args: argparse.Namespace,
    z_t: torch.Tensor,
    action_chunk: torch.Tensor,
) -> torch.Tensor:
    if args.gate_mode == "chunk_aware":
        if not isinstance(gate_runtime, ChunkAwareGateRuntime):
            raise TypeError("chunk_aware gate mode requires ChunkAwareGateRuntime.")
        return gate_runtime.logits_from_inputs(z_t, action_chunk)
    if not isinstance(gate_runtime, GateHeadRuntime):
        raise TypeError("z_only gate mode requires GateHeadRuntime.")
    return gate_runtime.logits_from_feature(z_t)


def build_log_row(
    *,
    episode_id: int,
    step_id: int,
    frame_index: int,
    video_frame_index: int,
    z_norm: float,
    gate_logit: float,
    gate_prob: float,
    gate_binary: float,
    gate_feature_action_chunk: int,
    gate_mode: str,
    gate_action_chunk: torch.Tensor,
    state14: np.ndarray,
    qpos_action_chunk: torch.Tensor,
    reward: float,
    done: bool,
    success: bool,
    pose_debug: dict[str, float | None],
) -> dict[str, Any]:
    first_action = qpos_action_chunk[0, 0].detach().cpu()
    last_action = qpos_action_chunk[0, -1].detach().cpu()
    action_delta = last_action - first_action
    action_mean = qpos_action_chunk[0].detach().cpu().mean(dim=0)
    action_std = qpos_action_chunk[0].detach().cpu().std(dim=0, unbiased=False)
    gate_action_chunk_cpu = gate_action_chunk[0].detach().cpu()
    row = {
        f"obs_state_{index:02d}": float(value)
        for index, value in enumerate(np.asarray(state14, dtype=np.float32).reshape(-1))
    }
    row.update(
        {
            f"qpos_action_first_{index:02d}": float(value)
            for index, value in enumerate(first_action)
        }
    )
    row.update(
        {
            f"qpos_action_last_{index:02d}": float(value)
            for index, value in enumerate(last_action)
        }
    )
    result = {
        "episode_id": episode_id,
        "step_id": step_id,
        "frame_index": frame_index,
        "video_frame_index": video_frame_index,
        "gate_mode": gate_mode,
        "gate_feature_action_chunk": gate_feature_action_chunk,
        "z_norm": z_norm,
        "gate_logit": gate_logit,
        "gate_prob": gate_prob,
        "gate_binary": gate_binary,
        "gate_binary_t02": float(gate_prob >= 0.2),
        "gate_binary_t04": float(gate_prob >= 0.4),
        "gate_binary_t06": float(gate_prob >= 0.6),
        "gate_binary_t08": float(gate_prob >= 0.8),
        "gate_binary_t05": float(gate_prob >= 0.5),
        "gate_binary_t07": float(gate_prob >= 0.7),
        "gate_binary_t09": float(gate_prob >= 0.9),
        "qpos_action_norm": float(qpos_action_chunk.norm().detach().cpu()),
        "qpos_action_mean_norm": float(action_mean.norm()),
        "qpos_action_std_norm": float(action_std.norm()),
        "qpos_action_delta_norm": float(action_delta.norm()),
        "action_chunk_norm": float(gate_action_chunk_cpu.norm()),
        "qpos_left_arm_delta_norm": float(action_delta[:6].norm()),
        "qpos_right_arm_delta_norm": float(action_delta[7:13].norm()),
        "left_gripper": float(state14[6]),
        "right_gripper": float(state14[13]),
        "action_first_left_gripper": float(first_action[6]),
        "action_first_right_gripper": float(first_action[13]),
        "action_last_left_gripper": float(last_action[6]),
        "action_last_right_gripper": float(last_action[13]),
        "reward": reward,
        "done": int(done),
        "success": int(success),
        **pose_debug,
        **row,
    }
    if gate_mode == "chunk_aware":
        result["gate_prob_chunk_aware"] = gate_prob
        result["gate_binary_chunk_aware"] = gate_binary
    return result


def read_pose_debug(task: Any) -> dict[str, float | None]:
    try:
        qpos14 = read_task_qpos14(task)
        pose16 = read_pose16(task, qpos14)
        left_pos = pose16[0:3]
        right_pos = pose16[8:11]
        return {
            "left_ee_x": float(left_pos[0]),
            "left_ee_y": float(left_pos[1]),
            "left_ee_z": float(left_pos[2]),
            "right_ee_x": float(right_pos[0]),
            "right_ee_y": float(right_pos[1]),
            "right_ee_z": float(right_pos[2]),
            "left_right_distance": float(np.linalg.norm(left_pos - right_pos)),
        }
    except Exception:  # noqa: BLE001
        return {
            "left_ee_x": None,
            "left_ee_y": None,
            "left_ee_z": None,
            "right_ee_x": None,
            "right_ee_y": None,
            "right_ee_z": None,
            "left_right_distance": None,
        }


def get_task(env: Any) -> Any:
    return env.venv.envs[0].task


def read_success(task: Any, info: dict[str, Any], done: bool) -> bool:
    if "success" in info:
        return bool(np.any(to_numpy(info["success"])))
    return bool(done and getattr(task, "eval_success", False))


def extract_state14(obs: dict[str, Any]) -> np.ndarray:
    state = obs.get("states")
    if state is None:
        state = obs.get("observation.state")
    if state is None:
        raise KeyError("online diagnostics require obs['states'] or obs['observation.state'].")
    state_np = to_numpy(state).astype(np.float32).reshape(-1)
    if state_np.shape[0] != 14:
        raise ValueError(f"observation.state must flatten to 14 values, got shape {state_np.shape}.")
    return state_np


def build_diagnostics_dir(args: argparse.Namespace) -> Path:
    run_name = args.diagnostics_run_name or Path(args.save_dir).name
    diagnostics_dir = Path(args.diagnostics_root) / run_name
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    return diagnostics_dir


def annotate_rule_gate(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    left_pos = np.asarray(
        [[row["left_ee_x"], row["left_ee_y"], row["left_ee_z"]] for row in rows],
        dtype=np.float64,
    )
    right_pos = np.asarray(
        [[row["right_ee_x"], row["right_ee_y"], row["right_ee_z"]] for row in rows],
        dtype=np.float64,
    )
    left_gripper = np.asarray([row["left_gripper"] for row in rows], dtype=np.float64)
    right_gripper = np.asarray([row["right_gripper"] for row in rows], dtype=np.float64)
    valid_pose = np.isfinite(left_pos).all(axis=1) & np.isfinite(right_pos).all(axis=1)
    v_left = np.zeros(len(rows), dtype=np.float64)
    v_right = np.zeros(len(rows), dtype=np.float64)
    for index in range(1, len(rows)):
        if valid_pose[index] and valid_pose[index - 1]:
            v_left[index] = float(np.linalg.norm(left_pos[index] - left_pos[index - 1]))
            v_right[index] = float(np.linalg.norm(right_pos[index] - right_pos[index - 1]))
    distance = np.asarray(
        [
            np.nan if row["left_right_distance"] is None else row["left_right_distance"]
            for row in rows
        ],
        dtype=np.float64,
    )
    both_moving = (v_left > args.rule_motion_threshold) & (v_right > args.rule_motion_threshold)
    both_closed = (
        (left_gripper > args.rule_gripper_closed_threshold)
        & (right_gripper > args.rule_gripper_closed_threshold)
    )
    close_and_moving = both_moving & np.isfinite(distance) & (distance < args.rule_distance_threshold)
    rule_gate = both_closed | close_and_moving
    for index, row in enumerate(rows):
        row["rule_gate"] = int(rule_gate[index])
        row["rule_both_grippers_closed"] = int(both_closed[index])
        row["rule_both_arms_moving"] = int(both_moving[index])
        row["rule_close_and_moving"] = int(close_and_moving[index])
        row["rule_v_left"] = float(v_left[index])
        row["rule_v_right"] = float(v_right[index])
        row["rule_distance_threshold"] = float(args.rule_distance_threshold)


def save_online_diagnostics(
    diagnostics_dir: Path,
    rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if feature_rows:
        np.savez_compressed(
            diagnostics_dir / "online_features.npz",
            episode_id=np.asarray([row["episode_id"] for row in feature_rows], dtype=np.int64),
            step_id=np.asarray([row["step_id"] for row in feature_rows], dtype=np.int64),
            frame_index=np.asarray([row["frame_index"] for row in feature_rows], dtype=np.int64),
            video_frame_index=np.asarray(
                [row["video_frame_index"] for row in feature_rows],
                dtype=np.int64,
            ),
            z=np.stack([row["z"] for row in feature_rows]).astype(np.float32),
            state=np.stack([row["state"] for row in feature_rows]).astype(np.float32),
            z_norm=np.asarray([row["z_norm"] for row in feature_rows], dtype=np.float32),
            gate_logit=np.asarray([row["gate_logit"] for row in feature_rows], dtype=np.float32),
            gate_prob=np.asarray([row["gate_prob"] for row in feature_rows], dtype=np.float32),
            action_chunk=np.stack([row["action_chunk"] for row in feature_rows]).astype(np.float32),
        )
    else:
        np.savez_compressed(
            diagnostics_dir / "online_features.npz",
            episode_id=np.empty((0,), dtype=np.int64),
            step_id=np.empty((0,), dtype=np.int64),
            frame_index=np.empty((0,), dtype=np.int64),
            video_frame_index=np.empty((0,), dtype=np.int64),
            z=np.empty((0, 0), dtype=np.float32),
            state=np.empty((0, 14), dtype=np.float32),
            z_norm=np.empty((0,), dtype=np.float32),
            gate_logit=np.empty((0,), dtype=np.float32),
            gate_prob=np.empty((0,), dtype=np.float32),
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows) if rows else pa.table({})
    pq.write_table(table, diagnostics_dir / "online_gate_log.parquet")
    write_summary(diagnostics_dir / "threshold_sweep.json", {"thresholds": threshold_sweep(rows)})
    print(f"ONLINE DIAGNOSTICS: {diagnostics_dir}")


def threshold_sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    probs = np.asarray([row["gate_prob"] for row in rows], dtype=np.float64)
    rule = np.asarray([row.get("rule_gate", 0) for row in rows], dtype=bool)
    results = []
    for threshold in np.linspace(0.1, 0.9, 9):
        pred = probs >= threshold
        lengths = positive_segment_lengths(pred)
        active_indices = np.flatnonzero(pred)
        overlap_count = int(np.logical_and(pred, rule).sum())
        union_count = int(np.logical_or(pred, rule).sum())
        results.append(
            {
                "threshold": float(threshold),
                "activation_ratio": float(pred.mean()),
                "number_of_segments": int(len(lengths)),
                "mean_segment_length": float(np.mean(lengths)) if lengths else 0.0,
                "overlap_with_rule_gate": float(overlap_count / max(int(rule.sum()), 1)),
                "iou_with_rule_gate": float(overlap_count / max(union_count, 1)),
                "first_activation_step": int(active_indices[0]) if active_indices.size else None,
                "last_activation_step": int(active_indices[-1]) if active_indices.size else None,
            }
        )
    return results


def positive_segment_lengths(mask: np.ndarray) -> list[int]:
    lengths = []
    start = None
    for index, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            lengths.append(index - start)
            start = None
    if start is not None:
        lengths.append(len(mask) - start)
    return lengths


def save_gate_log(save_dir: Path, episode_id: int, rows: list[dict[str, Any]]) -> Path:
    path = save_dir / f"gate_log_episode_{episode_id:04d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"GATE LOG: {path}")
    return path


def save_gate_plot(save_dir: Path, episode_id: int, rows: list[dict[str, Any]], threshold: float) -> Path | None:
    if not rows:
        return None
    import matplotlib.pyplot as plt

    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"gate_plot_episode_{episode_id:04d}.png"
    x = np.asarray([row["frame_index"] for row in rows], dtype=np.int64)
    prob = np.asarray([row["gate_prob"] for row in rows], dtype=np.float32)
    gate = np.asarray([row["gate_binary"] for row in rows], dtype=np.float32)
    rule_gate = np.asarray([row.get("rule_gate", 0) for row in rows], dtype=np.float32)
    left_gripper = np.asarray([row["left_gripper"] for row in rows], dtype=np.float32)
    right_gripper = np.asarray([row["right_gripper"] for row in rows], dtype=np.float32)
    dist = np.asarray(
        [np.nan if row["left_right_distance"] is None else row["left_right_distance"] for row in rows],
        dtype=np.float32,
    )
    done = np.asarray([row["done"] for row in rows], dtype=np.float32)
    success = np.asarray([row["success"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, prob, label="gate_prob")
    axes[0].axhline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:g}")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="upper right")
    axes[1].step(x, gate, where="post", label="gate_binary")
    axes[1].step(x, rule_gate, where="post", label="rule_gate", alpha=0.75)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[2].plot(x, left_gripper, label="left_gripper")
    axes[2].plot(x, right_gripper, label="right_gripper")
    axes[2].legend(loc="upper right")
    axes[3].plot(x, dist, label="left_right_distance")
    axes[3].scatter(x[done > 0], done[done > 0], marker="x", label="done")
    axes[3].scatter(x[success > 0], success[success > 0], marker="o", label="success")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("frame_index")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"GATE PLOT: {path}")
    return path


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"SUMMARY: {path}")


def print_context(args: argparse.Namespace) -> None:
    print("Runtime Context")
    print("---------------")
    for key, value in vars(args).items():
        print(f"{key}={value}")
    for name in ("REPO_PATH", "ROBOTWIN_PATH", "ROBOT_PLATFORM", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


def print_summary(metrics: dict[str, Any]) -> None:
    print("GATE ROLLOUT SUMMARY")
    print("--------------------")
    for episode in metrics["episodes"]:
        print(
            f"episode={episode['episode_id']} success={int(episode['success'])} "
            f"length={episode['episode_length']} "
            f"gate_prob_min={episode['gate_prob_min']:.6g} "
            f"gate_prob_max={episode['gate_prob_max']:.6g} "
            f"gate_prob_mean={episode['gate_prob_mean']:.6g} "
            f"gate_activation_ratio_t08={episode['gate_activation_ratio_t08']:.6g}"
        )


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
