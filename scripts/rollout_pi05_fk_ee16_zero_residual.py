"""Roll out pi05 qpos14 policy through FK -> ee16 zero-residual actions."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_CONFIG = "examples/embodiment/config/robotwin_handover_block_openpi_pi05_eval.yaml"
DEFAULT_CHECKPOINT = "/nfs/data3/rlinf_data/pytorch_checkpoint/"
DEFAULT_NORM_STATS = (
    "/nfs/data3/piNFT/checkpoints/pi05_aloha_robotwin_handover/"
    "pi05_base/19999/assets/handover_expert/norm_stats.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-stats-path", default=DEFAULT_NORM_STATS)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--env-split", choices=("train", "eval"), default="eval")
    parser.add_argument(
        "--execution-mode",
        choices=("fk_ee16_zero_residual", "qpos14"),
        default="fk_ee16_zero_residual",
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument("--model-num-action-chunks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--use-eval-success-seeds", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--save-dir", default="logs/pi05_fk_ee16_zero_residual")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-source",
        choices=("auto", "third_view", "observer", "main"),
        default="third_view",
    )
    parser.add_argument("--robotwin-action-mode", default="ee16")
    parser.add_argument(
        "--ee16-execution-strategy",
        choices=("pointwise", "last_target"),
        default="pointwise",
    )
    parser.add_argument("--config-name", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--num-images-in-input", type=int, default=3)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--planner-backend", choices=("curobo", "mplib"), default="curobo")
    parser.add_argument("--debug-ee16-timing", action="store_true")
    parser.add_argument("--debug-action-smoothness", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    print_context(args)

    env = None
    try:
        env_cfg = load_env_cfg(args)
        episode_seeds = select_episode_seeds(env_cfg, args)
        model_cfg = build_actor_model_cfg(args)
        model = load_model(model_cfg, args.device)
        env = make_env(env_cfg)
        env.debug_ee16_timing = bool(args.debug_ee16_timing)
        env.ee16_execution_strategy = args.ee16_execution_strategy
        pipeline = None
        if args.execution_mode == "fk_ee16_zero_residual":
            task = get_single_task_after_reset(env)
            pipeline = build_pipeline(task, args)
            env.close(clear_cache=False)
            env = make_env(env_cfg)
            env.debug_ee16_timing = bool(args.debug_ee16_timing)
            env.ee16_execution_strategy = args.ee16_execution_strategy

        metrics = run_episodes(env, model, pipeline, args, episode_seeds)
        print_metrics(metrics)
        write_summary(metrics, args)
        return 0 if metrics["dispatch_errors"] == 0 else (2 if args.strict else 0)
    except Exception as exc:  # noqa: BLE001
        print(f"ROLLOUT: FAIL {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().strip().splitlines()[-24:]:
            print(f"  {line}")
        return 1
    finally:
        if env is not None:
            close_env(env)


def run_episodes(
    env: Any,
    model: Any,
    pipeline: Any,
    args: argparse.Namespace,
    episode_seeds: list[int],
) -> dict[str, Any]:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    episode_summaries: list[dict[str, Any]] = []
    dispatch_errors = 0

    for episode_id, episode_seed in enumerate(episode_seeds):
        obs, _ = env.reset(env_seeds=[episode_seed])
        task = get_single_task(env)
        frames: list[np.ndarray] = []
        append_frame(frames, env, obs, args.video_source)
        episode_return = 0.0
        env_steps = 0
        replan_steps = 0
        done = False
        debug_rows: list[dict[str, Any]] = []
        initial_pose16 = read_pose16(task, read_task_qpos14(task))
        final_pose16 = initial_pose16.copy()
        episode_start = time.perf_counter()
        episode_take_action_calls = 0
        episode_env_chunk_times = []
        episode_planner_failures = 0

        while env_steps < args.max_steps and not done:
            step_start = time.perf_counter()
            policy_start = time.perf_counter()
            qpos14_action_chunk, infer_info = predict_qpos14_chunk(model, obs, args)
            policy_time = time.perf_counter() - policy_start
            qpos14_action_chunk = qpos14_action_chunk[:, : args.chunk_len, :].contiguous()
            if qpos14_action_chunk.shape[1] == 0:
                raise ValueError("qpos14 action chunk is empty after chunk-len slicing.")

            fk_time = 0.0
            if args.execution_mode == "fk_ee16_zero_residual":
                assert pipeline is not None
                fk_start = time.perf_counter()
                pipeline_out = pipeline.build_zero_residual_action(qpos14_action_chunk.to(args.device))
                fk_time = time.perf_counter() - fk_start
                base_endpose16_chunk = pipeline_out.base_endpose16_chunk
                executed_endpose16_chunk = pipeline_out.executed_endpose16_chunk
                targets_before_compression = int(executed_endpose16_chunk.shape[1])
                env_action_chunk = collapse_consecutive_duplicate_targets(
                    executed_endpose16_chunk.detach().cpu().numpy()
                )
                targets_after_compression = int(env_action_chunk.shape[1])
                zero_residual_norm = float(pipeline_out.zero_residual_chunk.norm().detach().cpu())
                right_xyz_movement_norm = chunk_right_xyz_movement_norm(base_endpose16_chunk)
            else:
                base_endpose16_chunk = None
                executed_endpose16_chunk = None
                env_action_chunk = qpos14_action_chunk.detach().cpu().numpy()
                targets_before_compression = int(env_action_chunk.shape[1])
                targets_after_compression = int(env_action_chunk.shape[1])
                zero_residual_norm = 0.0
                right_xyz_movement_norm = 0.0

            before_pose16 = read_pose16(task, read_task_qpos14(task))
            try:
                env_start = time.perf_counter()
                obs_list, rewards, terms, truncs, infos_list, env_debug = execute_env_action_chunk(
                    env,
                    env_action_chunk,
                    frames,
                    args,
                )
                env_chunk_step_time = time.perf_counter() - env_start
            except Exception as exc:  # noqa: BLE001
                dispatch_errors += 1
                print(f"DISPATCH ERROR episode={episode_id} replan={replan_steps}: {type(exc).__name__}: {exc}")
                raise
            total_step_time = time.perf_counter() - step_start

            obs = obs_list[-1]
            rewards_np = to_numpy(rewards).astype(np.float64)
            terms_np = to_numpy(terms).astype(bool)
            truncs_np = to_numpy(truncs).astype(bool)
            infos = infos_list[-1] if infos_list else {}
            final_pose16 = read_pose16(task, read_task_qpos14(task))
            qpos_tracking_error = qpos14_tracking_error(qpos14_action_chunk, task)

            episode_return += float(rewards_np.sum())
            env_steps += int(rewards_np.shape[1])
            replan_steps += 1
            done = bool(np.logical_or(terms_np, truncs_np).any())
            success = bool(to_numpy(infos.get("success", [False])).reshape(-1)[0]) if infos else False
            smoothness = {}
            if args.debug_action_smoothness:
                smoothness = action_smoothness_summary(
                    qpos14_action_chunk=qpos14_action_chunk,
                    endpose16_chunk=executed_endpose16_chunk,
                )
            timing = {
                "policy_time_s": policy_time,
                "fk_time_s": fk_time,
                "env_chunk_step_time_s": env_chunk_step_time,
                "total_step_time_s": total_step_time,
                "qpos14_chunk_shape": tuple(qpos14_action_chunk.shape),
                "endpose16_chunk_shape": (
                    tuple(executed_endpose16_chunk.shape)
                    if executed_endpose16_chunk is not None
                    else None
                ),
                "targets_before_compression": targets_before_compression,
                "targets_after_compression": targets_after_compression,
                **env_debug,
            }
            timing.setdefault("execution_strategy", current_execution_strategy(args))
            timing.setdefault("chunk_len", int(env_action_chunk.shape[1]))
            timing.setdefault("selected_target_index", int(env_action_chunk.shape[1] - 1))
            timing.setdefault("take_action_calls", 0)
            timing.setdefault("planner_failure", False)
            timing["right_xyz_movement_norm"] = right_xyz_movement_norm
            timing["final_target_right_xyz"] = (
                env_action_chunk[0, -1, 8:11].astype(float).tolist()
                if env_action_chunk.shape[-1] == 16
                else None
            )
            episode_take_action_calls += int(timing.get("take_action_calls", 0))
            episode_env_chunk_times.append(float(env_chunk_step_time))
            episode_planner_failures += int(bool(timing.get("planner_failure", False)))
            debug_rows.append(
                build_debug_row(
                    episode_id=episode_id,
                    seed=episode_seed,
                    replan_step=replan_steps - 1,
                    qpos14_action_chunk=qpos14_action_chunk,
                    base_endpose16_chunk=base_endpose16_chunk,
                    executed_endpose16_chunk=executed_endpose16_chunk,
                    before_pose16=before_pose16,
                    after_pose16=final_pose16,
                    rewards=rewards_np,
                    terms=terms_np,
                    truncs=truncs_np,
                    success=success,
                    qpos14_action_norm=float(qpos14_action_chunk.norm().detach().cpu()),
                    right_xyz_movement_norm=right_xyz_movement_norm,
                    zero_residual_norm=zero_residual_norm,
                    qpos_tracking_error=qpos_tracking_error,
                    timing=timing,
                    smoothness=smoothness,
                    infer_info_keys=sorted(infer_info.keys()),
                )
            )
            print_step(
                episode_id,
                replan_steps - 1,
                env_steps,
                episode_return,
                success,
                qpos14_action_chunk,
                right_xyz_movement_norm,
                zero_residual_norm,
                qpos_tracking_error,
            )
            if args.debug_ee16_timing:
                print_timing_debug(episode_id, replan_steps - 1, timing)
            if args.debug_action_smoothness:
                print_smoothness_debug(episode_id, replan_steps - 1, smoothness)

        success = bool(getattr(task, "eval_success", False))
        episode_wall_time = time.perf_counter() - episode_start
        summary = {
            "episode": episode_id,
            "seed": episode_seed,
            "return": episode_return,
            "success": success,
            "episode_len": env_steps,
            "replan_steps": replan_steps,
            "episode_wall_time_s": episode_wall_time,
            "video_frames": len(frames),
            "execution_strategy": current_execution_strategy(args),
            "total_take_action_calls": episode_take_action_calls,
            "mean_take_action_calls_per_chunk": (
                episode_take_action_calls / replan_steps if replan_steps else 0.0
            ),
            "mean_env_chunk_time_s": (
                float(np.mean(episode_env_chunk_times)) if episode_env_chunk_times else 0.0
            ),
            "planner_failure_count": episode_planner_failures,
            "initial_pose16": initial_pose16.tolist(),
            "final_pose16": final_pose16.tolist(),
        }
        episode_summaries.append(summary)
        if args.save_debug:
            save_debug_episode(save_dir, episode_id, debug_rows, summary)
        if args.save_video:
            video_path = save_dir / "videos" / f"episode_{episode_id:03d}_seed_{episode_seed}.mp4"
            save_video(frames, video_path, args.video_fps)

    returns = [row["return"] for row in episode_summaries]
    lengths = [row["episode_len"] for row in episode_summaries]
    successes = [row["success"] for row in episode_summaries]
    return {
        "execution_mode": args.execution_mode,
        "execution_strategy": current_execution_strategy(args),
        "num_episodes": args.num_episodes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_episode_len": float(np.mean(lengths)) if lengths else 0.0,
        "dispatch_errors": dispatch_errors,
        "episodes": episode_summaries,
        "save_dir": str(save_dir),
    }


def build_pipeline(task: Any, args: argparse.Namespace):
    from rlinf.algorithms.residual_td3.action_adapter import (
        ResidualActionAdapter,
        ResidualActionSpec,
    )
    from rlinf.algorithms.residual_td3.endpose_action_pipeline import (
        EndposeActionPipeline,
        EndposeActionPipelineConfig,
    )
    from rlinf.algorithms.residual_td3.fk_bridge import AlohaFKBridge

    fk_bridge = AlohaFKBridge.from_robotwin_task(task, device=args.device)
    residual_adapter = ResidualActionAdapter(
        ResidualActionSpec(
            base_action_space="robotwin_endpose16",
            residual_mode="right_xyz_world_frame",
            residual_frame="world",
            base_action_dim=16,
            residual_chunk_len=args.chunk_len,
            env_action_chunk_len=args.chunk_len,
            right_xyz_indices=[8, 9, 10],
        )
    )
    return EndposeActionPipeline(
        fk_bridge=fk_bridge,
        residual_adapter=residual_adapter,
        config=EndposeActionPipelineConfig(
            residual_chunk_len=args.chunk_len,
            env_action_chunk_len=args.chunk_len,
            robotwin_action_mode=args.robotwin_action_mode,
        ),
    )


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
        for step_obs in obs_list:
            append_frame(frames, env, step_obs, args.video_source)
    return obs_list, rewards, terms, truncs, infos_list, env_debug


def current_execution_strategy(args: argparse.Namespace) -> str:
    if args.execution_mode != "fk_ee16_zero_residual":
        return "qpos14"
    return args.ee16_execution_strategy


def as_tensor_1d(value: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=dtype).reshape(-1).cpu()
    return torch.as_tensor(np.asarray(value).reshape(-1), dtype=dtype)


def merge_ee16_chunk_debug(step_debugs: list[dict[str, Any]]) -> dict[str, Any]:
    if not step_debugs:
        return {}
    merged = {
        "chunk_len": len(step_debugs),
        "take_action_calls": 0,
        "take_action_times": [],
        "hold_noop_count": 0,
        "left_epsilon_count": 0,
        "right_epsilon_count": 0,
        "step_times": [],
    }
    for item in step_debugs:
        merged["take_action_calls"] += int(item.get("take_action_calls", 0))
        merged["take_action_times"].extend(item.get("take_action_times", []))
        merged["hold_noop_count"] += int(item.get("hold_noop_count", 0))
        merged["left_epsilon_count"] += int(item.get("left_epsilon_count", 0))
        merged["right_epsilon_count"] += int(item.get("right_epsilon_count", 0))
        merged["step_times"].extend(item.get("step_times", []))
    return merged


def action_smoothness_summary(
    qpos14_action_chunk: torch.Tensor,
    endpose16_chunk: torch.Tensor | None,
) -> dict[str, float]:
    qpos = qpos14_action_chunk.detach().cpu()
    qpos_delta = torch.linalg.norm(qpos[:, 1:] - qpos[:, :-1], dim=-1)
    summary = prefixed_delta_stats("qpos14_delta", qpos_delta)
    if endpose16_chunk is None:
        return summary
    endpose = endpose16_chunk.detach().cpu()
    summary.update(
        prefixed_delta_stats(
            "left_xyz_delta",
            torch.linalg.norm(endpose[:, 1:, 0:3] - endpose[:, :-1, 0:3], dim=-1),
        )
    )
    summary.update(
        prefixed_delta_stats(
            "right_xyz_delta",
            torch.linalg.norm(endpose[:, 1:, 8:11] - endpose[:, :-1, 8:11], dim=-1),
        )
    )
    summary.update(
        prefixed_delta_stats(
            "left_quat_angular_delta",
            quat_angular_delta(endpose[:, :-1, 3:7], endpose[:, 1:, 3:7]),
        )
    )
    summary.update(
        prefixed_delta_stats(
            "right_quat_angular_delta",
            quat_angular_delta(endpose[:, :-1, 11:15], endpose[:, 1:, 11:15]),
        )
    )
    return summary


def prefixed_delta_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p95": 0.0,
        }
    flat = values.reshape(-1).to(torch.float32)
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_max": float(flat.max()),
        f"{prefix}_p95": float(torch.quantile(flat, 0.95)),
    }


def quat_angular_delta(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    q0 = torch.nn.functional.normalize(q0.to(torch.float32), dim=-1)
    q1 = torch.nn.functional.normalize(q1.to(torch.float32), dim=-1)
    dots = torch.abs((q0 * q1).sum(dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dots)


def predict_qpos14_chunk(model: Any, obs: dict[str, Any], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    env_obs = dict(obs)
    if getattr(model.config, "config_name", "").find("agilex") >= 0:
        env_obs.setdefault("extra_view_images", env_obs.get("wrist_images"))
    else:
        env_obs.setdefault("extra_view_images", None)
    with torch.no_grad():
        actions, info = model.predict_action_batch(
            env_obs,
            mode="eval",
            compute_values=False,
        )
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    actions = actions.to(dtype=torch.float32)
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 14:
        raise ValueError(f"pi05 qpos14 action chunk must have shape [1, C, 14], got {actions.shape}.")
    return actions.contiguous(), info


def load_env_cfg(args: argparse.Namespace) -> Any:
    from omegaconf import OmegaConf

    repo = Path(os.environ.get("REPO_PATH", Path.cwd())).resolve()
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples" / "embodiment"))
    config_path = Path(args.config)
    config_dir = repo / "examples" / "embodiment" / "config"
    resolved_config_path = (Path.cwd() / config_path).resolve()
    if resolved_config_path.is_relative_to(config_dir):
        from hydra import compose, initialize_config_dir

        config_name = str(resolved_config_path.relative_to(config_dir).with_suffix(""))
        with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
            cfg = compose(config_name=config_name)
    else:
        cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    if not hasattr(cfg.env, args.env_split):
        raise ValueError(f"config does not contain env.{args.env_split}: {args.config}")
    env_cfg = deepcopy(getattr(cfg.env, args.env_split))
    OmegaConf.set_struct(env_cfg, False)
    if args.task_name is not None:
        env_cfg.task_config.task_name = args.task_name
    env_cfg.total_num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.auto_reset = False
    env_cfg.ignore_terminations = False
    env_cfg.max_episode_steps = args.max_steps
    env_cfg.task_config.step_lim = args.max_steps
    env_cfg.task_config.episode_num = 1
    env_cfg.task_config.use_seed = bool(args.use_eval_success_seeds)
    env_cfg.task_config.render_freq = 0
    env_cfg.task_config.collect_data = bool(args.save_video)
    env_cfg.task_config.eval_video_log = bool(args.save_video)
    if args.data_dir is not None or args.save_video:
        data_dir = Path(args.data_dir) if args.data_dir is not None else Path(args.save_dir) / "data"
        env_cfg.task_config.save_path = str(data_dir)
    env_cfg.task_config.data_type.qpos = True
    env_cfg.task_config.data_type.endpose = True
    env_cfg.task_config.data_type.third_view = bool(args.save_video)
    if args.execution_mode == "fk_ee16_zero_residual":
        env_cfg.robotwin_action_mode = args.robotwin_action_mode
        env_cfg.ee16_execution_strategy = args.ee16_execution_strategy
        env_cfg.task_config.action_type = "ee"
        env_cfg.task_config.planner_backend = args.planner_backend
    else:
        env_cfg.robotwin_action_mode = "qpos14"
        env_cfg.ee16_execution_strategy = "pointwise"
        env_cfg.task_config.action_type = "qpos"
    if args.chunk_len <= 0:
        raise ValueError("chunk-len must be positive.")
    return env_cfg


def select_episode_seeds(env_cfg: Any, args: argparse.Namespace) -> list[int]:
    if not args.use_eval_success_seeds:
        return [args.seed + episode_id for episode_id in range(args.num_episodes)]
    seeds_path = Path(str(env_cfg.seeds_path))
    with seeds_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    task_name = str(getattr(env_cfg.task_config, "task_name", "handover_block"))
    seeds = data[task_name]["success_seeds"]
    selected = [int(seed) for seed in seeds[args.seed_offset : args.seed_offset + args.num_episodes]]
    if len(selected) != args.num_episodes:
        raise ValueError(f"requested {args.num_episodes} eval seeds, got {len(selected)}")
    print(f"EVAL SEEDS: {selected}")
    return selected


def build_actor_model_cfg(args: argparse.Namespace) -> Any:
    from omegaconf import OmegaConf

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path}")
    model_path = checkpoint_path.parent if checkpoint_path.name.endswith(".safetensors") else checkpoint_path
    cfg = OmegaConf.load("examples/embodiment/config/model/pi0_5.yaml")
    cfg.model_path = str(model_path)
    cfg.norm_stats_path = args.norm_stats_path
    cfg.num_action_chunks = args.model_num_action_chunks
    cfg.action_dim = 14
    cfg.add_value_head = True
    cfg.openpi.config_name = args.config_name
    cfg.openpi.action_chunk = args.model_num_action_chunks
    cfg.openpi.action_env_dim = 14
    cfg.openpi.num_images_in_input = args.num_images_in_input
    cfg.openpi.noise_level = args.noise_level
    cfg.openpi.detach_critic_input = True
    root = OmegaConf.create({"actor": {"model": cfg}})
    OmegaConf.resolve(root)
    print("MODEL CFG")
    print("---------")
    print(f"checkpoint={checkpoint_path}")
    print(f"model_path={model_path}")
    print(f"norm_stats_path={args.norm_stats_path}")
    print(f"config_name={args.config_name}")
    print(f"model_num_action_chunks={args.model_num_action_chunks}")
    print("expected_env_action_shape=[B,C,14]")
    return root.actor.model


def load_model(actor_model_cfg: Any, device: str) -> Any:
    if not getattr(actor_model_cfg, "norm_stats_path", None):
        install_norm_stats_fallback(action_dim=int(actor_model_cfg.action_dim))
    from rlinf.models import get_model

    model = get_model(actor_model_cfg)
    model.eval()
    model.to(torch.device(device))
    print("MODEL LOAD: OK")
    return model


def install_norm_stats_fallback(action_dim: int) -> None:
    from openpi.shared.normalize import NormStats
    from openpi.training import checkpoints

    original_load_norm_stats = checkpoints.load_norm_stats

    def load_norm_stats(checkpoint_dir: str, asset_id: str) -> dict[str, NormStats]:
        try:
            return original_load_norm_stats(checkpoint_dir, asset_id)
        except FileNotFoundError as exc:
            zeros = np.zeros(action_dim, dtype=np.float32)
            ones = np.ones(action_dim, dtype=np.float32)
            stats = NormStats(mean=zeros, std=ones, q01=-ones, q99=ones)
            print(f"NORM STATS: FALLBACK identity stats for {asset_id}: {exc}")
            return {"state": stats, "actions": stats}

    checkpoints.load_norm_stats = load_norm_stats


def make_env(env_cfg: Any) -> Any:
    ensure_robotwin_vector_env_importable()
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    env = RoboTwinEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    print("ENV CREATE: OK")
    return env


def ensure_robotwin_vector_env_importable() -> None:
    try:
        importlib.import_module("robotwin.envs.vector_env")
        return
    except ModuleNotFoundError:
        pass

    robotwin_path = os.environ.get("ROBOTWIN_PATH")
    if not robotwin_path:
        raise
    cwd = os.getcwd()
    try:
        os.chdir(robotwin_path)
        compat_mod = importlib.import_module("envs.vector_env")
    finally:
        os.chdir(cwd)
    robotwin_envs = importlib.import_module("robotwin.envs")
    sys.modules["robotwin.envs.vector_env"] = compat_mod
    setattr(robotwin_envs, "vector_env", compat_mod)
    print("IMPORT COMPAT: mapped envs.vector_env -> robotwin.envs.vector_env")


def get_single_task_after_reset(env: Any) -> Any:
    env.reset()
    return get_single_task(env)


def get_single_task(env: Any) -> Any:
    try:
        return env.venv.envs[0].task
    except Exception as exc:  # noqa: BLE001
        raise NotImplementedError("Could not access env.venv.envs[0].task.") from exc


def collapse_consecutive_duplicate_targets(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 16:
        raise ValueError(f"expected ee16 actions shape [1, C, 16], got {actions.shape}.")
    kept = [actions[:, 0:1, :]]
    last = actions[:, 0, :]
    for step_id in range(1, actions.shape[1]):
        current = actions[:, step_id, :]
        if np.allclose(current, last, atol=1e-7, rtol=1e-7):
            continue
        kept.append(actions[:, step_id : step_id + 1, :])
        last = current
    collapsed = np.concatenate(kept, axis=1)
    if collapsed.shape[1] != actions.shape[1]:
        print(f"execution_chunk_dedup={actions.shape[1]}->{collapsed.shape[1]}")
    return collapsed


def read_task_qpos14(task: Any) -> np.ndarray:
    left = np.asarray(task.robot.get_left_arm_jointState(), dtype=np.float32).reshape(7)
    right = np.asarray(task.robot.get_right_arm_jointState(), dtype=np.float32).reshape(7)
    return np.concatenate((left, right))


def read_pose16(task: Any, qpos14: np.ndarray) -> np.ndarray:
    left = np.asarray(task.get_arm_pose("left"), dtype=np.float32).reshape(7)
    right = np.asarray(task.get_arm_pose("right"), dtype=np.float32).reshape(7)
    return np.concatenate((left, [qpos14[6]], right, [qpos14[13]])).astype(np.float32)


def chunk_right_xyz_movement_norm(endpose16_chunk: torch.Tensor) -> float:
    right_xyz = endpose16_chunk[..., 8:11]
    if right_xyz.shape[1] <= 1:
        return 0.0
    return float((right_xyz[:, 1:] - right_xyz[:, :-1]).norm(dim=-1).sum().detach().cpu())


def qpos14_tracking_error(qpos14_action_chunk: torch.Tensor, task: Any) -> float:
    target = qpos14_action_chunk[0, -1].detach().cpu().numpy()
    observed = read_task_qpos14(task)
    return float(np.linalg.norm(target - observed))


def build_debug_row(**kwargs: Any) -> dict[str, Any]:
    row = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            row[key] = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            row[key] = value
        else:
            row[key] = value
    return row


def save_debug_episode(
    save_dir: Path,
    episode_id: int,
    debug_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    ep_dir = save_dir / f"episode_{episode_id:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {}
    scalars = []
    for idx, row in enumerate(debug_rows):
        scalar_row = {}
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                npz_payload[f"step_{idx:04d}_{key}"] = value
            else:
                scalar_row[key] = value
        scalars.append(scalar_row)
    if npz_payload:
        np.savez_compressed(ep_dir / "rollout_debug.npz", **npz_payload)
    with (ep_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "steps": scalars}, f, indent=2)
    print(f"DEBUG SAVE: {ep_dir}")


def write_summary(metrics: dict[str, Any], args: argparse.Namespace) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "rollout_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"SUMMARY SAVE: {path}")


def print_context(args: argparse.Namespace) -> None:
    print("Runtime Context")
    print("---------------")
    for key, value in vars(args).items():
        print(f"{key}={value}")
    for name in ("REPO_PATH", "ROBOTWIN_PATH", "ROBOT_PLATFORM", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


def print_step(
    episode_id: int,
    replan_step: int,
    env_steps: int,
    episode_return: float,
    success: bool,
    qpos14_action_chunk: torch.Tensor,
    right_xyz_movement_norm: float,
    zero_residual_norm: float,
    qpos_tracking_error: float,
) -> None:
    print(
        f"episode={episode_id} replan={replan_step} env_steps={env_steps} "
        f"return={episode_return:.6g} success={success} "
        f"qpos14_action_norm={float(qpos14_action_chunk.norm().detach().cpu()):.6g} "
        f"fk_right_xyz_movement_norm={right_xyz_movement_norm:.6g} "
        f"zero_residual_norm={zero_residual_norm:.6g} "
        f"qpos_tracking_error={qpos_tracking_error:.6g}"
    )


def print_metrics(metrics: dict[str, Any]) -> None:
    print("ROLLOUT SUMMARY")
    print("---------------")
    for key in (
        "execution_mode",
        "execution_strategy",
        "num_episodes",
        "success_rate",
        "mean_return",
        "mean_episode_len",
        "dispatch_errors",
        "save_dir",
    ):
        print(f"{key}={metrics[key]}")


def print_timing_debug(episode_id: int, replan_step: int, timing: dict[str, Any]) -> None:
    take_times = [float(v) for v in timing.get("take_action_times", [])]
    step_times = [float(v) for v in timing.get("step_times", [])]
    take_mean = float(np.mean(take_times)) if take_times else 0.0
    take_max = float(np.max(take_times)) if take_times else 0.0
    step_mean = float(np.mean(step_times)) if step_times else 0.0
    print(
        "TIMING "
        f"episode={episode_id} replan={replan_step} "
        f"policy_s={timing['policy_time_s']:.6g} "
        f"fk_s={timing['fk_time_s']:.6g} "
        f"env_s={timing['env_chunk_step_time_s']:.6g} "
        f"total_s={timing['total_step_time_s']:.6g} "
        f"qpos_shape={timing['qpos14_chunk_shape']} "
        f"ee_shape={timing['endpose16_chunk_shape']} "
        f"strategy={timing.get('execution_strategy', 'unknown')} "
        f"selected_target={timing.get('selected_target_index', 'n/a')} "
        f"targets={timing['targets_before_compression']}->{timing['targets_after_compression']} "
        f"take_calls={timing.get('take_action_calls', 0)} "
        f"take_mean_s={take_mean:.6g} "
        f"take_max_s={take_max:.6g} "
        f"ee_step_mean_s={step_mean:.6g} "
        f"right_xyz_move={timing.get('right_xyz_movement_norm', 0.0):.6g} "
        f"planner_failure={timing.get('planner_failure', False)} "
        f"hold_noop={timing.get('hold_noop_count', 0)} "
        f"left_eps={timing.get('left_epsilon_count', 0)} "
        f"right_eps={timing.get('right_epsilon_count', 0)}"
    )


def print_smoothness_debug(episode_id: int, replan_step: int, smoothness: dict[str, float]) -> None:
    if not smoothness:
        return
    keys = (
        "qpos14_delta_mean",
        "qpos14_delta_max",
        "right_xyz_delta_mean",
        "right_xyz_delta_max",
        "left_xyz_delta_mean",
        "left_xyz_delta_max",
        "right_quat_angular_delta_max",
        "left_quat_angular_delta_max",
    )
    payload = " ".join(f"{key}={smoothness.get(key, 0.0):.6g}" for key in keys)
    print(f"SMOOTHNESS episode={episode_id} replan={replan_step} {payload}")


def append_frame(frames: list[np.ndarray], env: Any, obs: dict[str, Any], video_source: str) -> None:
    frame = get_frame(env, obs, video_source)
    if frame is not None:
        frames.append(frame)


def get_frame(env: Any, obs: dict[str, Any], video_source: str) -> np.ndarray | None:
    candidates = []
    if video_source in ("auto", "third_view", "observer"):
        raw_obs = None
        try:
            raw_obs = env.venv.envs[0].task.get_obs()
        except Exception:  # noqa: BLE001
            try:
                raw_obs = env.venv.get_obs()[0]
            except Exception:  # noqa: BLE001
                raw_obs = None
        if raw_obs is not None:
            candidates.extend([raw_obs.get("third_view_rgb"), raw_obs.get("full_image")])
    if video_source in ("auto", "main"):
        candidates.append(obs.get("main_images"))
    for value in candidates:
        if value is None:
            continue
        arr = to_numpy(value)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            return np.asarray(arr, dtype=np.uint8).copy()
    return None


def save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        print(f"VIDEO: SKIP no frames path={path}")
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"VIDEO: OK path={path} frames={len(frames)}")


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def close_env(env: Any) -> None:
    try:
        env.close(clear_cache=False)
        print("ENV CLOSE: OK")
    except TypeError:
        env.close()
        print("ENV CLOSE: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"ENV CLOSE: FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
