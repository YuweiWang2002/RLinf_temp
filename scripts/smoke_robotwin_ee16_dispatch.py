"""Real RoboTwin smoke test for RLinf ee16 dispatch.

RLinf canonical endpose16 uses xyzw quaternions. RoboTwin's task
``get_arm_pose`` and ``take_action(action_type="ee")`` use wxyz at the
task boundary, so this script converts explicitly when building canonical
RLinf actions from RoboTwin poses.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.util
import math
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CONFIG = "examples/embodiment/config/robotwin_handover_block_ee_probe.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--robotwin-action-mode", default="ee16")
    parser.add_argument(
        "--mode",
        choices=("identity", "chunk_identity", "right_xyz_delta"),
        default="identity",
    )
    parser.add_argument("--chunk-len", type=int, default=1)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_context(args)
    try:
        cfg = load_env_cfg(args)
        env = make_env(cfg, args)
        try:
            env.reset()
            task = get_single_task(env)
            before = read_dual_pose(task)
            action16 = build_canonical_action16(task, before)
            print_quat_boundary()
            print_action("action16_canonical_xyzw.before_mode", action16)

            if args.mode == "identity":
                obs, reward, term, trunc, info = env.step(action16.reshape(1, 16))
                del obs
            elif args.mode == "chunk_identity":
                chunk = np.repeat(
                    action16.reshape(1, 1, 16),
                    repeats=args.chunk_len,
                    axis=1,
                )
                obs_list, reward, term, trunc, info_list = env.chunk_step(chunk)
                del obs_list
                info = info_list[-1] if info_list else {}
            else:
                action16 = action16.copy()
                axis_idx = {"x": 8, "y": 9, "z": 10}[args.axis]
                action16[axis_idx] += args.delta
                print_action("action16_canonical_xyzw.delta_mode", action16)
                obs, reward, term, trunc, info = env.step(action16.reshape(1, 16))
                del obs

            after = read_dual_pose(task)
            summary = summarize(before, after)
            print_summary(args, summary, reward, term, trunc, info)
            if args.mode == "right_xyz_delta":
                summary[f"right_{args.axis}_observed_delta"] = print_delta_diagnostic(
                    args,
                    before,
                    after,
                )
            if is_suspicious(args, summary):
                print("SMOKE: SUSPICIOUS")
                print_pose("before", before)
                print_pose("after", after)
                print_action("action16_canonical_xyzw", action16)
                return 2 if args.strict else 0
            print("SMOKE: PASS")
            return 0
        finally:
            close_env(env)
    except Exception as exc:  # noqa: BLE001
        print(f"SMOKE: FAIL {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().strip().splitlines()[-16:]:
            print(f"  {line}")
        return 1


def print_context(args: argparse.Namespace) -> None:
    print("Runtime Context")
    print("---------------")
    print(f"python={sys.executable}")
    print(f"cwd={os.getcwd()}")
    print(f"config={args.config}")
    print(f"robotwin_action_mode={args.robotwin_action_mode}")
    print(f"mode={args.mode}")
    print(f"chunk_len={args.chunk_len}")
    print(f"axis={args.axis}")
    print(f"delta={args.delta}")
    print(f"num_envs={args.num_envs}")
    print(f"seed={args.seed}")
    print(f"device={args.device}")
    for name in ("REPO_PATH", "ROBOTWIN_PATH", "ROBOT_PLATFORM", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


def load_env_cfg(args: argparse.Namespace) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(args.config))
    OmegaConf.resolve(cfg)
    env_cfg = cfg.env.train
    env_cfg.total_num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.robotwin_action_mode = args.robotwin_action_mode
    env_cfg.auto_reset = False
    env_cfg.ignore_terminations = False
    env_cfg.task_config.action_type = "ee"
    env_cfg.task_config.episode_num = args.num_envs
    env_cfg.task_config.use_seed = False
    env_cfg.task_config.render_freq = 0
    env_cfg.task_config.collect_data = False
    env_cfg.task_config.eval_video_log = False
    return env_cfg


def make_env(env_cfg: Any, args: argparse.Namespace) -> Any:
    ensure_robotwin_vector_env_importable()
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    env = RoboTwinEnv(
        cfg=env_cfg,
        num_envs=args.num_envs,
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
    except ModuleNotFoundError:
        os.chdir(cwd)
        ensure_robotwin_vector_env_pyc_importable(robotwin_path)
        importlib.import_module("robotwin.envs.vector_env")
        return
    finally:
        os.chdir(cwd)

    robotwin_envs = importlib.import_module("robotwin.envs")
    sys.modules["robotwin.envs.vector_env"] = compat_mod
    setattr(robotwin_envs, "vector_env", compat_mod)
    print("IMPORT COMPAT: mapped envs.vector_env -> robotwin.envs.vector_env")
    return


def ensure_robotwin_vector_env_pyc_importable(robotwin_path: str) -> None:
    pycache_dir = Path(robotwin_path) / "robotwin" / "envs" / "__pycache__"
    candidates = sorted(pycache_dir.glob("vector_env.*.pyc"))
    if not candidates:
        return
    _ensure_namespace_package("robotwin", Path(robotwin_path) / "robotwin")
    _ensure_namespace_package("robotwin.envs", Path(robotwin_path) / "robotwin" / "envs")
    loader = importlib.machinery.SourcelessFileLoader(
        "robotwin.envs.vector_env",
        str(candidates[0]),
    )
    spec = importlib.util.spec_from_loader("robotwin.envs.vector_env", loader)
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules["robotwin.envs.vector_env"] = module
    setattr(sys.modules["robotwin.envs"], "vector_env", module)
    print(f"IMPORT COMPAT: loaded {candidates[0]} -> robotwin.envs.vector_env")


def _ensure_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def get_single_task(env: Any) -> Any:
    if env.num_envs != 1:
        raise ValueError("This smoke script currently expects --num-envs 1.")
    try:
        return env.venv.envs[0].task
    except Exception as exc:  # noqa: BLE001
        raise NotImplementedError("Could not access env.venv.envs[0].task.") from exc


def read_dual_pose(task: Any) -> dict[str, np.ndarray]:
    left_wxyz = np.asarray(task.get_arm_pose("left"), dtype=np.float64).reshape(7)
    right_wxyz = np.asarray(task.get_arm_pose("right"), dtype=np.float64).reshape(7)
    return {
        "left_xyzw": pose_wxyz_to_xyzw(left_wxyz),
        "right_xyzw": pose_wxyz_to_xyzw(right_wxyz),
        "left_wxyz": left_wxyz,
        "right_wxyz": right_wxyz,
    }


def build_canonical_action16(task: Any, poses: dict[str, np.ndarray]) -> np.ndarray:
    left_gripper = get_gripper(task, "left")
    right_gripper = get_gripper(task, "right")
    return np.concatenate(
        (
            poses["left_xyzw"],
            np.asarray([left_gripper], dtype=np.float64),
            poses["right_xyzw"],
            np.asarray([right_gripper], dtype=np.float64),
        ),
        axis=0,
    ).astype(np.float32)


def get_gripper(task: Any, side: str) -> float:
    robot = task.robot
    if side == "left":
        if hasattr(robot, "get_left_gripper_val"):
            return float(robot.get_left_gripper_val())
        return float(getattr(robot, "left_gripper_val", 0.0))
    if hasattr(robot, "get_right_gripper_val"):
        return float(robot.get_right_gripper_val())
    return float(getattr(robot, "right_gripper_val", 0.0))


def pose_wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    converted = np.array(pose, copy=True)
    converted[3:7] = pose[[4, 5, 6, 3]]
    return converted


def quat_angle_error_xyzw(a_quat: np.ndarray, b_quat: np.ndarray) -> float:
    a = a_quat / np.linalg.norm(a_quat)
    b = b_quat / np.linalg.norm(b_quat)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


def summarize(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "left_pos_err": float(np.linalg.norm(after["left_xyzw"][0:3] - before["left_xyzw"][0:3])),
        "right_pos_err": float(np.linalg.norm(after["right_xyzw"][0:3] - before["right_xyzw"][0:3])),
        "left_quat_err_rad": quat_angle_error_xyzw(before["left_xyzw"][3:7], after["left_xyzw"][3:7]),
        "right_quat_err_rad": quat_angle_error_xyzw(before["right_xyzw"][3:7], after["right_xyzw"][3:7]),
    }


def is_suspicious(args: argparse.Namespace, summary: dict[str, float]) -> bool:
    if args.mode in ("identity", "chunk_identity"):
        return (
            summary["left_pos_err"] > 0.01
            or summary["right_pos_err"] > 0.01
            or summary["left_quat_err_rad"] > 0.05
            or summary["right_quat_err_rad"] > 0.05
        )
    if summary["left_pos_err"] > 0.02:
        return True
    observed = summary[f"right_{args.axis}_observed_delta"]
    if args.delta == 0.0:
        return abs(observed) > 1e-4
    same_direction = math.copysign(1.0, observed) == math.copysign(1.0, args.delta)
    enough_motion = abs(observed) >= 0.25 * abs(args.delta)
    return not (same_direction and enough_motion)


def print_quat_boundary() -> None:
    print("QUAT ORDER")
    print("----------")
    print("RoboTwin task.get_arm_pose returns [x,y,z,qw,qx,qy,qz] (wxyz).")
    print("RLinf canonical endpose16 is [x,y,z,qx,qy,qz,qw] (xyzw).")
    print("This script converts get_arm_pose wxyz -> canonical xyzw.")
    print("RoboTwinEnv ee16 dispatch converts canonical xyzw -> RoboTwin wxyz.")


def print_summary(
    args: argparse.Namespace,
    summary: dict[str, float],
    reward: Any,
    term: Any,
    trunc: Any,
    info: dict[str, Any],
) -> None:
    print("SUMMARY")
    print("-------")
    print(f"mode={args.mode}")
    for key, value in summary.items():
        print(f"{key}={value:.9g}")
    print(f"reward={to_numpy(reward)}")
    print(f"termination={to_numpy(term)}")
    print(f"truncation={to_numpy(trunc)}")
    print(f"info_keys={sorted(info.keys())}")
    print("dispatch_path=RoboTwinEnv.ee16 -> task.take_action(action_type='ee')")


def print_delta_diagnostic(
    args: argparse.Namespace,
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
) -> float:
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    delta = after["right_xyzw"][axis_idx] - before["right_xyzw"][axis_idx]
    print(f"right_{args.axis}_observed_delta={delta:.9g}")
    print(f"right_{args.axis}_requested_delta={args.delta:.9g}")
    return float(delta)


def print_pose(label: str, poses: dict[str, np.ndarray]) -> None:
    print(f"{label}.left_xyzw={format_vec(poses['left_xyzw'])}")
    print(f"{label}.right_xyzw={format_vec(poses['right_xyzw'])}")
    print(f"{label}.left_wxyz_raw={format_vec(poses['left_wxyz'])}")
    print(f"{label}.right_wxyz_raw={format_vec(poses['right_wxyz'])}")


def print_action(label: str, action: np.ndarray) -> None:
    print(f"{label}: shape={action.shape} values={format_vec(action)}")


def format_vec(value: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x):.9g}" for x in np.asarray(value).reshape(-1)) + "]"


def to_numpy(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
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
    raise SystemExit(main())
