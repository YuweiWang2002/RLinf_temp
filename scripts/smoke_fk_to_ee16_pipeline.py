"""Smoke qpos14 -> FK -> residual endpose16 -> RoboTwin ee16 execution."""

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
import torch

DEFAULT_CONFIG = "examples/embodiment/config/robotwin_handover_block_ee_probe.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("current_qpos_hold", "right_joint_delta"),
        default="current_qpos_hold",
    )
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument("--joint-index", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--planner-backend", choices=("curobo", "mplib"), default="curobo")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_context(args)
    env = None
    try:
        cfg = load_env_cfg(args)
        env = make_env(cfg)
        obs, _ = env.reset()
        task = get_single_task(env)
        current_qpos14 = read_qpos14(obs)
        initial_pose16 = read_pose16(task, current_qpos14)
        qpos14_chunk = build_qpos14_chunk(current_qpos14, args)
        print_vector("current_qpos14", current_qpos14)
        print_vector("target_qpos14_last", qpos14_chunk[0, -1])

        pipeline = build_pipeline(task, args)
        pipeline_out = pipeline.build_zero_residual_action(
            torch.as_tensor(qpos14_chunk, dtype=torch.float32, device=args.device)
        )
        base_endpose16 = pipeline_out.base_endpose16_chunk.detach().cpu().numpy()
        executed_endpose16 = pipeline_out.executed_endpose16_chunk.detach().cpu().numpy()
        print("PIPELINE")
        print("--------")
        print("path=qpos14 chunk -> AlohaFKBridge -> base_endpose16 -> right_xyz_ref -> zero residual -> ee16")
        print(f"base_endpose16_shape={tuple(base_endpose16.shape)}")
        print(f"residual_ref_shape={tuple(pipeline_out.residual_ref_chunk.shape)}")
        print(f"executed_endpose16_shape={tuple(executed_endpose16.shape)}")
        print_vector("target_endpose16_last", executed_endpose16[0, -1])

        execution_endpose16 = collapse_consecutive_duplicate_targets(executed_endpose16)
        print(f"execution_endpose16_shape={tuple(execution_endpose16.shape)}")
        obs_list, rewards, terms, truncs, infos_list = env.chunk_step(execution_endpose16)
        del obs_list
        final_qpos14 = read_task_qpos14(task)
        final_pose16 = read_pose16(task, final_qpos14)
        print_summary(
            args,
            initial_pose16,
            executed_endpose16[0, -1],
            final_pose16,
            rewards,
            terms,
            truncs,
            infos_list[-1] if infos_list else {},
        )
        if is_suspicious(args, executed_endpose16[0, -1], final_pose16):
            print("SMOKE: SUSPICIOUS")
            return 2 if args.strict else 0
        print("SMOKE: PASS")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"SMOKE: FAIL {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().strip().splitlines()[-18:]:
            print(f"  {line}")
        return 1
    finally:
        if env is not None:
            close_env(env)


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
            robotwin_action_mode="ee16",
        ),
    )


def print_context(args: argparse.Namespace) -> None:
    print("Runtime Context")
    print("---------------")
    print(f"python={sys.executable}")
    print(f"cwd={os.getcwd()}")
    print(f"config={args.config}")
    print(f"mode={args.mode}")
    print(f"chunk_len={args.chunk_len}")
    print(f"joint_index={args.joint_index}")
    print(f"delta={args.delta}")
    print(f"device={args.device}")
    print(f"planner_backend={args.planner_backend}")
    for name in ("REPO_PATH", "ROBOTWIN_PATH", "ROBOT_PLATFORM", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


def load_env_cfg(args: argparse.Namespace) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(args.config))
    OmegaConf.resolve(cfg)
    env_cfg = cfg.env.train
    env_cfg.total_num_envs = 1
    env_cfg.robotwin_action_mode = "ee16"
    env_cfg.auto_reset = False
    env_cfg.ignore_terminations = False
    env_cfg.task_config.action_type = "ee"
    env_cfg.task_config.episode_num = 1
    env_cfg.task_config.use_seed = False
    env_cfg.task_config.render_freq = 0
    env_cfg.task_config.planner_backend = args.planner_backend
    env_cfg.task_config.collect_data = False
    env_cfg.task_config.eval_video_log = False
    env_cfg.task_config.data_type.qpos = True
    env_cfg.task_config.data_type.endpose = True
    if args.chunk_len <= 0:
        raise ValueError("chunk-len must be positive.")
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
    try:
        return env.venv.envs[0].task
    except Exception as exc:  # noqa: BLE001
        raise NotImplementedError("Could not access env.venv.envs[0].task.") from exc


def read_qpos14(obs: dict[str, Any]) -> np.ndarray:
    states = to_numpy(obs["states"])
    if states.shape != (1, 14):
        raise ValueError(f"expected obs['states'] shape (1, 14), got {states.shape}.")
    return np.asarray(states[0], dtype=np.float32)


def read_task_qpos14(task: Any) -> np.ndarray:
    left = np.asarray(task.robot.get_left_arm_jointState(), dtype=np.float32).reshape(7)
    right = np.asarray(task.robot.get_right_arm_jointState(), dtype=np.float32).reshape(7)
    return np.concatenate((left, right))


def read_pose16(task: Any, qpos14: np.ndarray) -> np.ndarray:
    left = np.asarray(task.get_arm_pose("left"), dtype=np.float32).reshape(7)
    right = np.asarray(task.get_arm_pose("right"), dtype=np.float32).reshape(7)
    return np.concatenate((left, [qpos14[6]], right, [qpos14[13]])).astype(np.float32)


def build_qpos14_chunk(current_qpos14: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    chunk = np.repeat(current_qpos14.reshape(1, 1, 14), repeats=args.chunk_len, axis=1)
    if args.mode == "right_joint_delta":
        if args.joint_index < 0 or args.joint_index >= 6:
            raise ValueError("joint-index must be in [0, 5] for the right arm.")
        chunk[..., 7 + args.joint_index] += args.delta
    return chunk.astype(np.float32)


def collapse_consecutive_duplicate_targets(actions: np.ndarray) -> np.ndarray:
    """Avoid RoboTwin zero-step planner failures on repeated identical EE targets."""
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 16:
        raise ValueError(f"expected actions shape [1, C, 16], got {actions.shape}.")
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
        print(
            "execution_chunk_dedup="
            f"{actions.shape[1]}->{collapsed.shape[1]} "
            "reason=RoboTwin zero-step EE planner guard"
        )
    return collapsed


def print_summary(
    args: argparse.Namespace,
    initial_pose16: np.ndarray,
    target_pose16: np.ndarray,
    final_pose16: np.ndarray,
    rewards: Any,
    terms: Any,
    truncs: Any,
    info: dict[str, Any],
) -> None:
    print("SUMMARY")
    print("-------")
    print("dispatch_path=RoboTwinEnv.ee16 -> task.take_action(action_type='ee')")
    print_pose_error("target_vs_final", target_pose16, final_pose16)
    print_pose_delta("initial_to_final", initial_pose16, final_pose16)
    if args.mode == "right_joint_delta":
        right_motion = final_pose16[8:11] - initial_pose16[8:11]
        print(f"right_xyz_motion={format_vector(right_motion)}")
    print(f"reward={to_numpy(rewards)}")
    print(f"termination={to_numpy(terms)}")
    print(f"truncation={to_numpy(truncs)}")
    print(f"info_keys={sorted(info.keys())}")


def print_pose_error(label: str, target: np.ndarray, actual: np.ndarray) -> None:
    left_pos = float(np.linalg.norm(actual[0:3] - target[0:3]))
    right_pos = float(np.linalg.norm(actual[8:11] - target[8:11]))
    left_quat = quat_angle_error_wxyz(target[3:7], actual[3:7])
    right_quat = quat_angle_error_wxyz(target[11:15], actual[11:15])
    print(
        f"{label}: left_pos={left_pos:.9g} right_pos={right_pos:.9g} "
        f"left_quat={left_quat:.9g} right_quat={right_quat:.9g}"
    )


def print_pose_delta(label: str, before: np.ndarray, after: np.ndarray) -> None:
    print(f"{label}.left_xyz_delta={format_vector(after[0:3] - before[0:3])}")
    print(f"{label}.right_xyz_delta={format_vector(after[8:11] - before[8:11])}")


def is_suspicious(args: argparse.Namespace, target_pose16: np.ndarray, final_pose16: np.ndarray) -> bool:
    left_pos = float(np.linalg.norm(final_pose16[0:3] - target_pose16[0:3]))
    right_pos = float(np.linalg.norm(final_pose16[8:11] - target_pose16[8:11]))
    left_quat = quat_angle_error_wxyz(target_pose16[3:7], final_pose16[3:7])
    right_quat = quat_angle_error_wxyz(target_pose16[11:15], final_pose16[11:15])
    pos_limit = 0.03 if args.mode == "right_joint_delta" else 0.02
    return left_pos > pos_limit or right_pos > pos_limit or left_quat > 0.08 or right_quat > 0.08


def quat_angle_error_wxyz(a_quat: np.ndarray, b_quat: np.ndarray) -> float:
    a = a_quat / np.linalg.norm(a_quat)
    b = b_quat / np.linalg.norm(b_quat)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


def print_vector(label: str, value: np.ndarray) -> None:
    print(f"{label}: {format_vector(value)}")


def format_vector(value: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x):.9g}" for x in np.asarray(value).reshape(-1)) + "]"


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
    raise SystemExit(main())
