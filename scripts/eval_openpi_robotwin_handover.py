"""Evaluate OpenPI pi05 on RoboTwin handover_block with per-episode artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG_NAME = "robotwin_handover_block_openpi_pi05_eval"
DEFAULT_WEIGHTS_DIR = "/nfs/data3/rlinf_data/pytorch_checkpoint"
DEFAULT_NORM_STATS = (
    "/home/user/wyw/piNFT/assets/pi05_aloha_robotwin_handover/"
    "handover_expert/norm_stats.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--norm-stats-path", default=DEFAULT_NORM_STATS)
    parser.add_argument("--openpi-config-name", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="logs/openpi_handover_eval")
    parser.add_argument("--data-dir", default="data/openpi_handover_eval")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_context(args)

    try:
        ensure_robotwin_vector_env_importable()
        cfg = compose_cfg(args)
        seeds = load_eval_seeds(cfg.env.eval.seeds_path, args.seed_offset, args.episodes)
        model = load_model(cfg.actor.model, args.device)
        rows = run_eval(args, cfg.env.eval, model, seeds)
        write_summary(Path(args.output_dir) / "summary.csv", rows)
        print_summary(rows)
    except Exception as exc:  # noqa: BLE001
        print(f"EVAL: FAIL {type(exc).__name__}: {exc}")
        raise

    return 0


def print_context(args: argparse.Namespace) -> None:
    print("Runtime Context")
    print("---------------")
    print(f"python={sys.executable}")
    print(f"cwd={os.getcwd()}")
    print(f"weights_dir={args.weights_dir}")
    print(f"norm_stats_path={args.norm_stats_path}")
    for name in ("REPO_PATH", "ROBOTWIN_PATH", "ROBOT_PLATFORM", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


def compose_cfg(args: argparse.Namespace) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo = Path(os.environ.get("REPO_PATH", Path.cwd())).resolve()
    os.environ.setdefault("EMBODIED_PATH", str(repo / "examples" / "embodiment"))
    config_dir = repo / "examples" / "embodiment" / "config"
    overrides = [
        f"actor.model.model_path={args.weights_dir}",
        f"actor.model.norm_stats_path={args.norm_stats_path}",
        f"actor.model.openpi.config_name={args.openpi_config_name}",
        f"rollout.model.model_path={args.weights_dir}",
        f"rollout.model.norm_stats_path={args.norm_stats_path}",
        "env.eval.total_num_envs=1",
        "env.eval.auto_reset=False",
        "env.eval.ignore_terminations=False",
        f"env.eval.max_episode_steps={args.max_steps}",
        f"env.eval.max_steps_per_rollout_epoch={args.max_steps}",
        f"env.eval.task_config.step_lim={args.max_steps}",
        f"env.eval.task_config.save_path={args.data_dir}",
        "env.eval.task_config.collect_data=True",
        "env.eval.task_config.eval_video_log=True",
        "env.eval.task_config.data_type.third_view=True",
        "env.eval.task_config.data_type.qpos=True",
        "env.eval.task_config.data_type.endpose=True",
        "env.eval.task_config.camera.collect_wrist_camera=True",
    ]
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def ensure_robotwin_vector_env_importable() -> None:
    import importlib

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


def load_eval_seeds(seeds_path: str, offset: int, episodes: int) -> list[int]:
    with open(seeds_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seeds = data["handover_block"]["success_seeds"]
    selected = [int(seed) for seed in seeds[offset : offset + episodes]]
    if len(selected) != episodes:
        raise ValueError(f"requested {episodes} seeds, got {len(selected)}")
    print(f"EVAL SEEDS: {selected}")
    return selected


def load_model(actor_model_cfg: Any, device: str) -> Any:
    import torch

    from rlinf.models import get_model

    model = get_model(actor_model_cfg)
    model.eval()
    model.to(torch.device(device))
    print("MODEL LOAD: OK")
    return model


def run_eval(args: argparse.Namespace, env_cfg: Any, model: Any, seeds: list[int]) -> list[dict[str, Any]]:
    from copy import deepcopy

    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    rows = []
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    cfg = deepcopy(env_cfg)
    cfg.seed = int(seeds[0])
    cfg.task_config.use_seed = True
    cfg.task_config.episode_num = len(seeds)
    cfg.task_config.save_path = str(Path(args.data_dir).resolve())
    env = RoboTwinEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    try:
        for episode_idx, seed in enumerate(seeds):
            row = run_episode(args, env, model, episode_idx, seed, video_dir)
            rows.append(row)
            write_summary(output_dir / "summary.csv", rows)
    finally:
        try:
            env.close(clear_cache=False)
        except TypeError:
            env.close()
    return rows


def run_episode(
    args: argparse.Namespace,
    env: Any,
    model: Any,
    episode_idx: int,
    seed: int,
    video_dir: Path,
) -> dict[str, Any]:
    frames: list[np.ndarray] = []
    success = False
    reward_value = 0.0
    info_keys: list[str] = []
    steps = 0
    obs, _ = env.reset(env_seeds=[seed])
    append_frame(frames, env, obs)
    for steps in range(1, args.max_steps + 1):
        actions = predict_action(model, obs)
        obs, reward, termination, truncation, info = env.step(actions)
        append_frame(frames, env, obs)
        reward_value += float(np.asarray(to_numpy(reward)).reshape(-1)[0])
        info_keys = sorted(info.keys())
        success = success or read_success(env, termination, info)
        if bool(np.any(to_numpy(termination))) or bool(np.any(to_numpy(truncation))):
            break
    video_path = video_dir / f"episode_{episode_idx:03d}_seed_{seed}.mp4"
    save_video(frames, video_path, args.video_fps)

    row = {
        "episode": episode_idx,
        "seed": seed,
        "success": int(success),
        "steps": steps,
        "return": reward_value,
        "video": str(video_path),
        "info_keys": ";".join(info_keys),
    }
    print(
        "EPISODE "
        f"{episode_idx}: seed={seed} success={row['success']} "
        f"steps={steps} return={reward_value:.6g} video={video_path}"
    )
    return row


def predict_action(model: Any, obs: dict[str, Any]) -> np.ndarray:
    import torch

    env_obs = dict(obs)
    env_obs.setdefault("extra_view_images", None)
    with torch.no_grad():
        actions, _ = model.predict_action_batch(
            env_obs,
            mode="eval",
            compute_values=False,
        )
    return to_numpy(actions)


def read_success(env: Any, termination: Any, info: dict[str, Any]) -> bool:
    if bool(np.any(to_numpy(termination))):
        return True
    if "success" in info:
        return bool(np.any(to_numpy(info["success"])))
    try:
        return bool(getattr(env.venv.envs[0].task, "eval_success", False))
    except Exception:  # noqa: BLE001
        return False


def append_frame(frames: list[np.ndarray], env: Any, obs: dict[str, Any]) -> None:
    frame = get_frame(env, obs)
    if frame is not None:
        frames.append(frame)


def get_frame(env: Any, obs: dict[str, Any]) -> np.ndarray | None:
    raw_obs = None
    try:
        raw_obs = env.venv.envs[0].task.get_obs()
    except Exception:  # noqa: BLE001
        try:
            raw_obs = env.venv.get_obs()[0]
        except Exception:  # noqa: BLE001
            raw_obs = None
    candidates = []
    if raw_obs is not None:
        candidates.extend([raw_obs.get("third_view_rgb"), raw_obs.get("full_image")])
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


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    successes = sum(int(row["success"]) for row in rows)
    total = len(rows)
    rate = successes / total if total else 0.0
    print(f"SUCCESS: {successes}/{total} = {rate:.3f}")


def to_numpy(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == "__main__":
    raise SystemExit(main())
