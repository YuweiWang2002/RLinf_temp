"""Probe ALOHA qpos14 -> RoboTwin endpose16 FK bridge feasibility.

This script is intentionally probe-only. It checks whether CuRobo FK can map
ALOHA qpos14 targets to the same world-frame EE pose returned by
``task.get_arm_pose("left"/"right")`` before any training code depends on it.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CONFIG = "examples/embodiment/config/robotwin_handover_block_ee_probe.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("current_pose", "qpos_delta"), default="current_pose")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--close-env", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_context(args)
    if not runtime_available():
        print("SKIP: RoboTwin/CuRobo runtime is unavailable.")
        return 2 if args.strict else 0

    env = None
    try:
        cfg = load_env_config(args.config)
        if args.mode == "qpos_delta":
            cfg.task_config.action_type = "qpos"
        env = make_env(cfg)
        obs, _ = env.reset()
        task = env.venv.envs[0].task
        bridge = AlohaCuRoboFK.from_task(task, device=args.device)
        print_audit(task, bridge)
        current_qpos14 = read_qpos14(task, obs)
        print_vector("current_qpos14", current_qpos14)

        if args.mode == "current_pose":
            compare_current_pose(task, bridge, current_qpos14)
        else:
            run_qpos_delta_probe(task, bridge, current_qpos14)
    except Exception as exc:  # noqa: BLE001
        print(f"PROBE: FAIL {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=12)
        return 1
    finally:
        if env is not None and args.close_env:
            close_env(env)

    print("PROBE: PASS")
    return 0


class AlohaCuRoboFK:
    """Minimal CuRobo FK wrapper for RoboTwin ALOHA probe."""

    def __init__(self, left: "_ArmFK", right: "_ArmFK") -> None:
        self.left = left
        self.right = right

    @classmethod
    def from_task(cls, task: Any, device: str) -> "AlohaCuRoboFK":
        robot = task.robot
        root_path = os.environ.get("ROBOTWIN_PATH") or os.environ.get("ASSETS_PATH")
        if root_path is None:
            raise RuntimeError("ROBOTWIN_PATH or ASSETS_PATH must be set for ALOHA CuRobo FK.")
        root = Path(root_path)
        left_yml = root / "assets" / "embodiments" / "aloha-agilex" / "curobo_left.yml"
        right_yml = root / "assets" / "embodiments" / "aloha-agilex" / "curobo_right.yml"
        left = _ArmFK(
            arm_tag="left",
            yml_path=left_yml,
            arm_joint_names=list(robot.left_arm_joints_name),
            origin_pose=robot.left_entity_origion_pose,
            global_trans_matrix=np.asarray(robot.left_global_trans_matrix, dtype=np.float64),
            delta_matrix=np.asarray(robot.left_delta_matrix, dtype=np.float64),
            frame_bias=np.asarray(_read_frame_bias(left_yml), dtype=np.float64),
            device=device,
        )
        right = _ArmFK(
            arm_tag="right",
            yml_path=right_yml,
            arm_joint_names=list(robot.right_arm_joints_name),
            origin_pose=robot.right_entity_origion_pose,
            global_trans_matrix=np.asarray(robot.right_global_trans_matrix, dtype=np.float64),
            delta_matrix=np.asarray(robot.right_delta_matrix, dtype=np.float64),
            frame_bias=np.asarray(_read_frame_bias(right_yml), dtype=np.float64),
            device=device,
        )
        return cls(left=left, right=right)

    def qpos14_to_endpose16(self, qpos14: np.ndarray) -> np.ndarray:
        qpos14 = np.asarray(qpos14, dtype=np.float64).reshape(14)
        left_pose = self.left.fk_endpose_wxyz(qpos14[0:6])
        right_pose = self.right.fk_endpose_wxyz(qpos14[7:13])
        return np.concatenate(
            (
                left_pose,
                np.asarray([qpos14[6]], dtype=np.float64),
                right_pose,
                np.asarray([qpos14[13]], dtype=np.float64),
            )
        )


class _ArmFK:
    def __init__(
        self,
        *,
        arm_tag: str,
        yml_path: Path,
        arm_joint_names: list[str],
        origin_pose: Any,
        global_trans_matrix: np.ndarray,
        delta_matrix: np.ndarray,
        frame_bias: np.ndarray,
        device: str,
    ) -> None:
        if not yml_path.exists():
            raise FileNotFoundError(f"CuRobo config not found: {yml_path}")

        import torch
        from curobo.cuda_robot_model.cuda_robot_model import (
            CudaRobotModel,
            CudaRobotModelConfig,
        )
        from curobo.types.base import TensorDeviceType

        tensor_args = TensorDeviceType(device=torch.device(device))
        cfg = CudaRobotModelConfig.from_robot_yaml_file(str(yml_path), tensor_args=tensor_args)
        self.model = CudaRobotModel(cfg)
        self.torch = torch
        self.arm_tag = arm_tag
        self.yml_path = yml_path
        self.arm_joint_names = arm_joint_names
        self.curobo_joint_names = list(self.model.kinematics_config.joint_names)
        self.origin_p = np.asarray(origin_pose.p, dtype=np.float64)
        self.origin_q = normalize_quat(np.asarray(origin_pose.q, dtype=np.float64))
        self.global_trans_matrix = global_trans_matrix
        self.delta_matrix = delta_matrix
        self.frame_bias = frame_bias
        self.device = device

    def fk_raw_base_wxyz(self, qpos6: np.ndarray) -> np.ndarray:
        q = self._pack_qpos(qpos6)
        with self.torch.no_grad():
            state = self.model.get_state(q)
        pos = state.ee_position.detach().cpu().numpy()[0].astype(np.float64)
        quat = state.ee_quaternion.detach().cpu().numpy()[0].astype(np.float64)
        return np.concatenate((pos, normalize_quat(quat)))

    def fk_gripper_world_wxyz(self, qpos6: np.ndarray) -> np.ndarray:
        """CuRobo raw FK -> RoboTwin planner gripper pose in world frame."""
        raw = self.fk_raw_base_wxyz(qpos6)
        base_p = raw[:3] - self.frame_bias
        base_q = raw[3:7]
        world_rot = quat2mat(self.origin_q) @ quat2mat(base_q)
        world_p = self.origin_p + quat2mat(self.origin_q) @ base_p
        world_q = mat2quat(world_rot)
        return np.concatenate((world_p, normalize_quat(world_q)))

    def fk_endpose_wxyz(self, qpos6: np.ndarray) -> np.ndarray:
        """CuRobo FK converted to the frame returned by task.get_arm_pose.

        RoboTwin applies ``global_trans_matrix`` when reading SAPIEN joint pose,
        but CuRobo's configured ``ee_link`` orientation already matches the
        resulting get_arm_pose orientation. Applying the matrix again produces a
        pi-radian quaternion error, so the bridge candidate only applies the
        CuRobo planner's frame_bias and the robot root pose.
        """
        return self.fk_gripper_world_wxyz(qpos6)

    def fk_with_robotwin_matrices_wxyz(self, qpos6: np.ndarray) -> np.ndarray:
        gripper_pose = self.fk_gripper_world_wxyz(qpos6)
        rot = quat2mat(gripper_pose[3:7]) @ self.global_trans_matrix @ self.delta_matrix
        return np.concatenate((gripper_pose[:3], mat2quat(rot)))

    def _pack_qpos(self, qpos6: np.ndarray) -> Any:
        qpos6 = np.asarray(qpos6, dtype=np.float32).reshape(6)
        values_by_name = {name: float(qpos6[i]) for i, name in enumerate(self.arm_joint_names)}
        values_by_name.update({name: 0.04 for name in self.curobo_joint_names if name.endswith(("joint7", "joint8"))})
        values = [values_by_name.get(name, 0.0) for name in self.curobo_joint_names]
        return self.torch.tensor([values], dtype=self.torch.float32, device=self.device)


def print_context(args: argparse.Namespace) -> None:
    print("Runtime Context")
    print("---------------")
    print(f"python={sys.executable}")
    print(f"cwd={os.getcwd()}")
    print(f"config={args.config}")
    print(f"mode={args.mode}")
    print(f"device={args.device}")
    for name in ("REPO_PATH", "ROBOTWIN_PATH", "ASSETS_PATH", "ROBOT_PLATFORM", "CUDA_VISIBLE_DEVICES"):
        print(f"{name}={os.environ.get(name, '<unset>')}")


def runtime_available() -> bool:
    try:
        import curobo  # noqa: F401
        import robotwin  # noqa: F401

        import rlinf  # noqa: F401

        ensure_robotwin_vector_env_importable()
    except Exception as exc:  # noqa: BLE001
        print(f"runtime import failed: {type(exc).__name__}: {exc}")
        return False
    return True


def load_env_config(path: str) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(path))
    OmegaConf.resolve(cfg)
    env_cfg = deepcopy(cfg.env.train)
    env_cfg.task_config.data_type.qpos = True
    env_cfg.task_config.data_type.endpose = True
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


def read_qpos14(task: Any, obs: dict[str, Any] | None = None) -> np.ndarray:
    left = np.asarray(task.robot.get_left_arm_jointState(), dtype=np.float64).reshape(7)
    right = np.asarray(task.robot.get_right_arm_jointState(), dtype=np.float64).reshape(7)
    qpos14 = np.concatenate((left, right))
    if obs is not None and "states" in obs:
        obs_state = to_numpy(obs["states"])[0]
        print(f"obs.states shape={to_numpy(obs['states']).shape} first14={format_vector(obs_state[:14])}")
    return qpos14


def read_pose16(task: Any, grippers: np.ndarray) -> np.ndarray:
    left = np.asarray(task.get_arm_pose("left"), dtype=np.float64).reshape(7)
    right = np.asarray(task.get_arm_pose("right"), dtype=np.float64).reshape(7)
    return np.concatenate((left, [grippers[0]], right, [grippers[1]])).astype(np.float64)


def compare_current_pose(task: Any, bridge: AlohaCuRoboFK, qpos14: np.ndarray) -> None:
    gt = read_pose16(task, np.asarray([qpos14[6], qpos14[13]], dtype=np.float64))
    pred = bridge.qpos14_to_endpose16(qpos14)
    print_pose_block("current", bridge, qpos14, pred, gt)


def run_qpos_delta_probe(task: Any, bridge: AlohaCuRoboFK, qpos14: np.ndarray) -> None:
    deltas = (
        ("right_joint1_plus_0.01", 7, 0.01),
        ("right_joint2_plus_0.01", 8, 0.01),
        ("left_joint1_plus_0.01", 0, 0.01),
    )
    for name, index, delta in deltas:
        target = qpos14.copy()
        target[index] += delta
        pred = bridge.qpos14_to_endpose16(target)
        print(f"\nQPOS DELTA {name}: index={index} delta={delta}")
        print_vector("target_qpos14", target)
        task.take_action(target, action_type="qpos")
        observed_qpos14 = read_qpos14(task)
        gt = read_pose16(
            task,
            np.asarray([observed_qpos14[6], observed_qpos14[13]], dtype=np.float64),
        )
        print_pose_block(name, bridge, target, pred, gt)
        qpos14 = observed_qpos14


def print_audit(task: Any, bridge: AlohaCuRoboFK) -> None:
    robot = task.robot
    print("\nAudit")
    print("-----")
    print("direct qpos target -> ee pose interface: NO")
    print("CuRobo CudaRobotModel.get_state available: YES")
    print(f"left arm_joints_name={robot.left_arm_joints_name}")
    print(f"right arm_joints_name={robot.right_arm_joints_name}")
    print(f"left curobo_joint_names={bridge.left.curobo_joint_names}")
    print(f"right curobo_joint_names={bridge.right.curobo_joint_names}")
    print("input qpos14 split: left[0:6], left_gripper[6], right[7:13], right_gripper[13]")
    print("FK quaternion order: wxyz")
    print("output endpose16 quaternion order: wxyz")
    print(f"left frame_bias={bridge.left.frame_bias.tolist()}")
    print(f"right frame_bias={bridge.right.frame_bias.tolist()}")
    print(f"left global_trans_matrix={bridge.left.global_trans_matrix.tolist()}")
    print(f"right global_trans_matrix={bridge.right.global_trans_matrix.tolist()}")
    print(f"left delta_matrix={bridge.left.delta_matrix.tolist()}")
    print(f"right delta_matrix={bridge.right.delta_matrix.tolist()}")
    print("applied frame correction: frame_bias + robot root pose")
    print("global_trans_matrix/delta_matrix correction: NOT applied to CuRobo FK output")


def print_pose_block(
    label: str,
    bridge: AlohaCuRoboFK,
    qpos14: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
) -> None:
    print(f"\nPOSE COMPARE {label}")
    print("----------------")
    print_vector("left_raw_fk_base_wxyz", bridge.left.fk_raw_base_wxyz(qpos14[0:6]))
    print_vector("right_raw_fk_base_wxyz", bridge.right.fk_raw_base_wxyz(qpos14[7:13]))
    print_vector(
        "left_fk_with_global_delta_wxyz",
        bridge.left.fk_with_robotwin_matrices_wxyz(qpos14[0:6]),
    )
    print_vector(
        "right_fk_with_global_delta_wxyz",
        bridge.right.fk_with_robotwin_matrices_wxyz(qpos14[7:13]),
    )
    print_vector("left_fk_endpose_wxyz", pred[0:7])
    print_vector("left_get_arm_pose_wxyz", gt[0:7])
    print_vector("right_fk_endpose_wxyz", pred[8:15])
    print_vector("right_get_arm_pose_wxyz", gt[8:15])
    for arm, pred_pose, gt_pose in (
        ("left", pred[0:7], gt[0:7]),
        ("right", pred[8:15], gt[8:15]),
    ):
        pos_err = float(np.linalg.norm(pred_pose[:3] - gt_pose[:3]))
        ang_err = quat_angle_error(pred_pose[3:7], gt_pose[3:7])
        print(f"{arm}_pos_err_l2={pos_err:.9g}")
        print(f"{arm}_quat_ang_err_rad={ang_err:.9g}")
    print(f"gripper_copy left={pred[7]:.9g} right={pred[15]:.9g}")


def _read_frame_bias(path: Path) -> list[float]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return list(data.get("planner", {}).get("frame_bias", [0.0, 0.0, 0.0]))


def quat_angle_error(a: np.ndarray, b: np.ndarray) -> float:
    qa = normalize_quat(a)
    qb = normalize_quat(b)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError(f"invalid quaternion: {q}")
    return q / norm


def quat2mat(q: np.ndarray) -> np.ndarray:
    import transforms3d as t3d

    return t3d.quaternions.quat2mat(normalize_quat(q))


def mat2quat(mat: np.ndarray) -> np.ndarray:
    import transforms3d as t3d

    return normalize_quat(t3d.quaternions.mat2quat(np.asarray(mat, dtype=np.float64)))


def print_vector(label: str, value: np.ndarray) -> None:
    print(f"{label}: {format_vector(value)}")


def format_vector(value: np.ndarray) -> str:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:.9g}" for x in arr) + "]"


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
        env.close()
        print("ENV CLOSE: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"ENV CLOSE: FAIL {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
