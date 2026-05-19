"""Minimal OpenPI -> RoboTwin handover_block probe.

This script keeps all adaptation local to the probe: it creates the RoboTwin EE
environment, loads an OpenPI checkpoint, runs one inference from the reset
observation, converts the first predicted action to a 14D EE command, and steps
the environment once.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CONFIG = "examples/embodiment/config/robotwin_handover_block_ee_probe.yaml"
DEFAULT_WEIGHTS = "/nfs/data3/openpi-torch/pi05_base/model.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config-name", default="pi05_aloha_robotwin")
    parser.add_argument(
        "--action-mode",
        choices=("raw", "openpi-ee14-to-pose14", "agilex-qpos-fk-ee"),
        default="raw",
        help="How to adapt the first OpenPI 14D action before env.step.",
    )
    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-path", default="logs/openpi_robotwin_probe.mp4")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-source",
        choices=("auto", "observer", "main"),
        default="auto",
        help="Which RoboTwin camera stream to record when available.",
    )
    parser.add_argument("--close-env", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_context(args)

    try:
        import safetensors.torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"IMPORT safetensors.torch: FAIL {type(exc).__name__}: {exc}")
        return 1
    print("IMPORT safetensors.torch: OK")

    if not runtime_available():
        print("SKIP: RoboTwin/RLinf/OpenPI runtime is unavailable.")
        return 2 if args.strict else 0

    try:
        cfg = load_probe_config(args.config)
        actor_model_cfg = build_actor_model_cfg(args.weights, args.config_name)
        model = load_model(actor_model_cfg, args.device)
        print_semantic_alignment(args.config_name, args.action_mode)
        if args.action_mode == "agilex-qpos-fk-ee":
            run_agilex_qpos_fk_eval(args, cfg, model)
            print("PROBE: PASS")
            return 0

        if args.save_video and args.video_source != "main":
            cfg.task_config.data_type.third_view = True
        env = make_env(cfg)
        obs, _ = env.reset()
        frames: list[np.ndarray] = []
        append_runtime_frame(frames, env, obs, args.video_source)
        print_obs(obs)

        for step_id in range(args.eval_steps):
            action = predict_action(model, obs)
            openpi_action_14 = first_14d_action(action)
            print_action(f"step_{step_id}.openpi_action_14d", openpi_action_14)
            env_action_14 = adapt_action(openpi_action_14, args.action_mode)
            print_action(f"step_{step_id}.env_action_14d", env_action_14)

            obs, reward, termination, truncation, info = env.step(env_action_14.reshape(1, 14))
            append_runtime_frame(frames, env, obs, args.video_source)
            print(f"ENV STEP {step_id}: OK")
            print(f"reward={to_numpy(reward)}")
            print(f"termination={to_numpy(termination)}")
            print(f"truncation={to_numpy(truncation)}")
            print(f"info_keys={sorted(info.keys())}")
            if is_done(termination) or is_done(truncation):
                print(f"EVAL STOP: done at step {step_id}")
                break

        print_obs(obs, label="final_obs")
        if args.save_video:
            save_video(frames, args.video_path, args.video_fps)
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE: FAIL {type(exc).__name__}: {exc}")
        print_short_traceback()
        return 1
    finally:
        if "env" in locals() and args.close_env:
            try:
                env.close()
                print("ENV CLOSE: OK")
            except Exception as exc:  # noqa: BLE001
                print(f"ENV CLOSE: FAIL {type(exc).__name__}: {exc}")

    print("PROBE: PASS")
    return 0


def print_context(args: argparse.Namespace) -> None:
    print("Runtime Context")
    print("---------------")
    print(f"python={sys.executable}")
    print(f"cwd={os.getcwd()}")
    print(f"config={args.config}")
    print(f"weights={args.weights}")
    print(f"device={args.device}")
    for name in ("REPO_PATH", "ROBOTWIN_PATH", "ROBOT_PLATFORM", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


def runtime_available() -> bool:
    try:
        import openpi  # noqa: F401
        import robotwin  # noqa: F401
        import rlinf  # noqa: F401
        ensure_robotwin_vector_env_importable()
    except Exception as exc:  # noqa: BLE001
        print(f"runtime import failed: {type(exc).__name__}: {exc}")
        return False
    return True


def load_probe_config(path: str) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(path))
    OmegaConf.resolve(cfg)
    env_cfg = cfg.env.train
    if env_cfg.task_config.task_name != "handover_block":
        raise ValueError(f"expected handover_block, got {env_cfg.task_config.task_name}")
    if env_cfg.task_config.action_type != "ee":
        raise ValueError(f"expected action_type=ee, got {env_cfg.task_config.action_type}")
    return env_cfg


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


def clone_env_cfg(env_cfg: Any, action_type: str) -> Any:
    from copy import deepcopy

    cloned = deepcopy(env_cfg)
    cloned.task_config.action_type = action_type
    cloned.task_config.data_type.endpose = True
    cloned.task_config.data_type.third_view = True
    return cloned


def build_actor_model_cfg(weights: str, config_name: str) -> Any:
    from omegaconf import OmegaConf

    weights_path = Path(weights)
    model_path = weights_path.parent if weights_path.name.endswith(".safetensors") else weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"weights path does not exist: {weights_path}")
    if weights_path.name.endswith(".safetensors"):
        print(f"WEIGHTS FILE: OK {weights_path}")
    else:
        print(f"WEIGHTS DIR: OK {model_path}")

    default_path = Path("examples/embodiment/config/model/pi0_5.yaml")
    cfg = OmegaConf.load(default_path)
    cfg.model_path = str(model_path)
    cfg.action_dim = 14
    cfg.openpi.config_name = config_name
    root = OmegaConf.create({"actor": {"model": cfg}})
    OmegaConf.resolve(root)
    print("MODEL CFG: OK")
    print(OmegaConf.to_yaml(root.actor.model))
    return root.actor.model


def load_model(actor_model_cfg: Any, device: str) -> Any:
    import torch

    install_norm_stats_fallback(action_dim=int(actor_model_cfg.action_dim))

    from rlinf.models import get_model

    model = get_model(actor_model_cfg)
    model.eval()
    model.to(torch.device(device))
    print("MODEL LOAD: OK")
    return model


def install_norm_stats_fallback(action_dim: int) -> None:
    """Use identity norm stats when the probe checkpoint lacks them."""
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


def predict_action(model: Any, obs: dict[str, Any]) -> np.ndarray:
    import torch

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
    print(f"INFERENCE: OK info_keys={sorted(info.keys())}")
    return to_numpy(actions)


def first_14d_action(action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape[-1] != 14:
        raise ValueError(f"expected action last dim 14, got shape {arr.shape}")
    if arr.ndim == 3:
        return arr[0, 0].copy()
    if arr.ndim == 2:
        return arr[0].copy()
    if arr.ndim == 1:
        return arr.copy()
    raise ValueError(f"unsupported action shape: {arr.shape}")


def adapt_action(action_14: np.ndarray, action_mode: str) -> np.ndarray:
    if action_mode == "raw":
        return np.asarray(action_14, dtype=np.float32).copy()
    if action_mode == "openpi-ee14-to-pose14":
        from rlinf_robotwin.control.action_adapter import openpi_ee14_to_pose14

        return np.asarray(openpi_ee14_to_pose14(np.asarray(action_14)), dtype=np.float32)
    if action_mode == "agilex-qpos-fk-ee":
        raise RuntimeError("agilex-qpos-fk-ee uses the two-pass FK eval path.")
    raise ValueError(f"unsupported action_mode: {action_mode}")


def run_agilex_qpos_fk_eval(args: argparse.Namespace, env_cfg: Any, model: Any) -> None:
    print("FK PASS")
    print("-------")
    qpos_env = make_env(clone_env_cfg(env_cfg, "qpos"))
    pose_actions: list[np.ndarray] = []
    ee_actions: list[np.ndarray] = []
    qpos_pose_rows: list[dict[str, float]] = []
    qpos_frames: list[np.ndarray] = []
    try:
        obs, _ = qpos_env.reset()
        append_runtime_frame(qpos_frames, qpos_env, obs, args.video_source)
        print_obs(obs, label="qpos_obs")
        for step_id in range(args.eval_steps):
            qpos_action = first_14d_action(predict_action(model, obs))
            print_action(f"fk_step_{step_id}.qpos_action_14d", qpos_action)
            obs, reward, termination, truncation, info = qpos_env.step(
                qpos_action.reshape(1, 14)
            )
            append_runtime_frame(qpos_frames, qpos_env, obs, args.video_source)
            pose_action = read_task_pose14(qpos_env)
            pose_actions.append(pose_action)
            ee_actions.append(build_ee_action16(pose_action, qpos_action[6], qpos_action[13]))
            qpos_pose_rows.append(make_pose_row(step_id, pose_action, prefix="qpos"))
            print_action(f"fk_step_{step_id}.ee_pose14_from_runtime", pose_action)
            print(f"QPOS STEP {step_id}: OK reward={to_numpy(reward)}")
            print(f"termination={to_numpy(termination)} truncation={to_numpy(truncation)}")
            print(f"info_keys={sorted(info.keys())}")
            if is_done(termination) or is_done(truncation):
                print(f"FK PASS STOP: done at step {step_id}")
                break
        if args.save_video:
            qpos_video_path, _, _, _ = derive_eval_artifact_paths(args.video_path)
            save_video(qpos_frames, qpos_video_path, args.video_fps)
    finally:
        close_env(qpos_env, "QPOS ENV")

    print("EE REPLAY PASS")
    print("--------------")
    ee_env = make_env(clone_env_cfg(env_cfg, "ee"))
    frames: list[np.ndarray] = []
    ee_pose_rows: list[dict[str, float]] = []
    compare_rows: list[dict[str, float]] = []
    try:
        obs, _ = ee_env.reset()
        task = ee_env.venv.envs[0].task
        append_runtime_frame(frames, ee_env, obs, args.video_source)
        print_obs(obs, label="ee_obs")
        for step_id, (pose_action, ee_action) in enumerate(zip(pose_actions, ee_actions, strict=True)):
            task.take_action(ee_action, action_type="ee")
            obs = build_direct_task_obs(ee_env)
            append_runtime_frame(frames, ee_env, obs, args.video_source)
            ee_pose = read_task_pose14(ee_env)
            ee_pose_rows.append(make_pose_row(step_id, ee_pose, prefix="ee"))
            compare_rows.append(compare_pose_row(step_id, pose_action, ee_pose))
            print(f"EE STEP {step_id}: OK")
            print_pose_error(step_id, pose_action, ee_pose)
            if getattr(task, "eval_success", False) or task.take_action_cnt >= task.step_lim:
                print(f"EE REPLAY STOP: done at step {step_id}")
                break
        print_obs(obs, label="final_ee_obs")
        qpos_video_path, ee_video_path, qpos_csv_path, ee_csv_path = derive_eval_artifact_paths(
            args.video_path
        )
        compare_csv_path = str(Path(args.video_path).with_name(f"{Path(args.video_path).stem}_compare.csv"))
        write_csv(qpos_csv_path, qpos_pose_rows)
        write_csv(ee_csv_path, ee_pose_rows)
        write_csv(compare_csv_path, compare_rows)
        print_compare_summary(compare_rows)
        if args.save_video:
            save_video(frames, ee_video_path, args.video_fps)
    finally:
        if args.close_env:
            close_env(ee_env, "EE ENV")


def read_task_pose14(env: Any) -> np.ndarray:
    task = env.venv.envs[0].task
    left = np.asarray(task.get_arm_pose("left"), dtype=np.float32).reshape(7)
    right = np.asarray(task.get_arm_pose("right"), dtype=np.float32).reshape(7)
    pose14 = np.concatenate((left, right), axis=-1)
    if not np.all(np.isfinite(pose14)):
        raise ValueError(f"non-finite pose14 from get_arm_pose: {pose14}")
    left_norm = np.linalg.norm(pose14[3:7])
    right_norm = np.linalg.norm(pose14[10:14])
    print(f"runtime pose quat_norms left={left_norm:.6g} right={right_norm:.6g}")
    return pose14


def build_ee_action16(pose14: np.ndarray, left_gripper: float, right_gripper: float) -> np.ndarray:
    arr = np.asarray(pose14, dtype=np.float32).reshape(14)
    return np.concatenate(
        (
            arr[0:7],
            np.asarray([left_gripper], dtype=np.float32),
            arr[7:14],
            np.asarray([right_gripper], dtype=np.float32),
        )
    )


def build_direct_task_obs(env: Any) -> dict[str, Any]:
    raw_obs = env.venv.get_obs()[0]
    raw_obs["instruction"] = env.venv.envs[0].task.get_instruction()
    return env._extract_obs_image([raw_obs])


def derive_eval_artifact_paths(video_path: str) -> tuple[str, str, str, str]:
    base = Path(video_path)
    stem = base.stem
    qpos_video = str(base.with_name(f"{stem}_qpos.mp4"))
    ee_video = str(base.with_name(f"{stem}_ee.mp4"))
    qpos_csv = str(base.with_name(f"{stem}_qpos_poses.csv"))
    ee_csv = str(base.with_name(f"{stem}_ee_poses.csv"))
    return qpos_video, ee_video, qpos_csv, ee_csv


def get_runtime_frame(env: Any, obs: dict[str, Any], video_source: str) -> np.ndarray | None:
    raw_obs = None
    try:
        raw_obs = env.venv.envs[0].task.get_obs()
    except Exception as exc:  # noqa: BLE001
        print(f"FRAME CAPTURE task.get_obs FAIL {type(exc).__name__}: {exc}")
        try:
            raw_obs_list = env.venv.get_obs()
        except Exception as inner_exc:  # noqa: BLE001
            print(f"FRAME CAPTURE venv.get_obs FAIL {type(inner_exc).__name__}: {inner_exc}")
            raw_obs_list = None
        raw_obs = raw_obs_list[0] if raw_obs_list else None
    candidates: list[tuple[str, Any]] = []
    if raw_obs is not None:
        candidates.append(("observer", raw_obs.get("third_view_rgb")))
        candidates.append(("main", raw_obs.get("full_image")))
    candidates.append(("main", obs.get("main_images")))

    preferred = ["observer", "main"] if video_source == "auto" else [video_source]
    for wanted in preferred:
        for name, value in candidates:
            if name != wanted or value is None:
                continue
            arr = to_numpy_or_none(value)
            if arr is None:
                continue
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3:
                print(f"FRAME SOURCE: {wanted}")
                return np.asarray(arr, dtype=np.uint8).copy()
    return None


def append_runtime_frame(
    frames: list[np.ndarray], env: Any, obs: dict[str, Any], video_source: str
) -> None:
    frame = get_runtime_frame(env, obs, video_source)
    if frame is not None:
        frames.append(frame)


def close_env(env: Any, label: str) -> None:
    try:
        env.venv.close()
        print(f"{label} CLOSE: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"{label} CLOSE: FAIL {type(exc).__name__}: {exc}")


def print_semantic_alignment(config_name: str, action_mode: str) -> None:
    print("ACTION SEMANTICS")
    print("----------------")
    print("RoboTwin task.take_action(action_type=ee) expects RoboTwinEEAction16:")
    print("  [left xyz_quat7, left_gripper1, right xyz_quat7, right_gripper1]")
    if "agilex" in config_name:
        print("OpenPI config is Agilex-style and outputs qpos/gripper 14D:")
        print("  [left qpos6, left gripper1, right qpos6, right gripper1]")
        print(f"semantic_match={action_mode == 'agilex-qpos-fk-ee'}")
        if action_mode == "agilex-qpos-fk-ee":
            print("adapter=RoboTwin runtime qpos env -> task.get_arm_pose world EE pose14 -> ee_action16")
    else:
        print("OpenPI config is treated as OpenPIEEAction14:")
        print("  [left xyz_rotvec6, left gripper1, right xyz_rotvec6, right gripper1]")
        print(f"semantic_match={action_mode == 'openpi-ee14-to-pose14'}")
    print(f"dimension_match=True action_dim=14 action_mode={action_mode}")


def print_obs(obs: dict[str, Any], label: str = "obs") -> None:
    print(f"{label.upper()} KEYS: {sorted(obs.keys())}")
    for key in ("main_images", "wrist_images", "states", "task_descriptions"):
        if key not in obs:
            continue
        value = obs[key]
        arr = to_numpy_or_none(value)
        if arr is None:
            print(f"{label}.{key}: {type(value).__name__}={value}")
        else:
            print(f"{label}.{key}: shape={arr.shape} dtype={arr.dtype} minmax={minmax(arr)}")


def print_action(label: str, action: np.ndarray) -> None:
    arr = np.asarray(action)
    print(f"{label}: shape={arr.shape} dtype={arr.dtype} minmax={minmax(arr)}")
    print(f"{label}.left={format_vector(arr[:7])}")
    print(f"{label}.right={format_vector(arr[7:14])}")


def save_video(frames: list[np.ndarray], path: str, fps: int) -> None:
    if not frames:
        print("VIDEO: SKIP no frames")
        return
    import imageio.v2 as imageio

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=fps)
    print(f"VIDEO: OK path={output} frames={len(frames)} fps={fps}")


def make_pose_row(step_id: int, pose14: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(pose14, dtype=np.float64).reshape(14)
    keys = (
        "left_x",
        "left_y",
        "left_z",
        "left_qx",
        "left_qy",
        "left_qz",
        "left_qw",
        "right_x",
        "right_y",
        "right_z",
        "right_qx",
        "right_qy",
        "right_qz",
        "right_qw",
    )
    row: dict[str, float] = {"step": float(step_id)}
    for key, value in zip(keys, arr, strict=True):
        row[f"{prefix}_{key}"] = float(value)
    return row


def compare_pose_row(step_id: int, target_pose14: np.ndarray, actual_pose14: np.ndarray) -> dict[str, float]:
    target = np.asarray(target_pose14, dtype=np.float64).reshape(14)
    actual = np.asarray(actual_pose14, dtype=np.float64).reshape(14)
    left_pos_err = float(np.linalg.norm(actual[0:3] - target[0:3]))
    right_pos_err = float(np.linalg.norm(actual[7:10] - target[7:10]))
    left_quat_dot = float(np.clip(np.abs(np.dot(target[3:7], actual[3:7])), -1.0, 1.0))
    right_quat_dot = float(np.clip(np.abs(np.dot(target[10:14], actual[10:14])), -1.0, 1.0))
    left_rot_err = float(2.0 * np.arccos(left_quat_dot))
    right_rot_err = float(2.0 * np.arccos(right_quat_dot))
    return {
        "step": float(step_id),
        "left_pos_err_l2": left_pos_err,
        "right_pos_err_l2": right_pos_err,
        "left_rot_err_rad": left_rot_err,
        "right_rot_err_rad": right_rot_err,
    }


def print_pose_error(step_id: int, target_pose14: np.ndarray, actual_pose14: np.ndarray) -> None:
    row = compare_pose_row(step_id, target_pose14, actual_pose14)
    print(
        "POSE ERROR "
        f"step={step_id} "
        f"left_pos={row['left_pos_err_l2']:.6g} "
        f"right_pos={row['right_pos_err_l2']:.6g} "
        f"left_rot={row['left_rot_err_rad']:.6g} "
        f"right_rot={row['right_rot_err_rad']:.6g}"
    )


def print_compare_summary(compare_rows: list[dict[str, float]]) -> None:
    if not compare_rows:
        print("COMPARE: SKIP no rows")
        return
    for key in ("left_pos_err_l2", "right_pos_err_l2", "left_rot_err_rad", "right_rot_err_rad"):
        values = np.asarray([row[key] for row in compare_rows], dtype=np.float64)
        print(f"COMPARE {key}: mean={values.mean():.6g} max={values.max():.6g}")


def write_csv(path: str, rows: list[dict[str, float]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"CSV: SKIP path={output} rows=0")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: OK path={output} rows={len(rows)}")


def is_done(value: Any) -> bool:
    arr = np.asarray(to_numpy(value))
    return bool(np.any(arr))


def to_numpy(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_numpy_or_none(value: Any) -> np.ndarray | None:
    try:
        return to_numpy(value)
    except Exception:  # noqa: BLE001
        return None


def minmax(arr: np.ndarray) -> str:
    if arr.size == 0 or arr.dtype.kind not in "biufc":
        return "n/a"
    return f"{float(np.min(arr)):.6g}/{float(np.max(arr)):.6g}"


def format_vector(value: Any) -> str:
    flat = np.asarray(value).reshape(-1)
    shown = ", ".join(f"{float(x):.6g}" for x in flat[:16])
    return f"[{shown}]"


def print_short_traceback() -> None:
    for line in traceback.format_exc().strip().splitlines()[-12:]:
        print(f"  {line}")


if __name__ == "__main__":
    raise SystemExit(main())
