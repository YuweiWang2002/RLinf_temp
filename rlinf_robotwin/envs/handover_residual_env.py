"""Safe residual wrapper for RoboTwin handover tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf_robotwin.control.action_fusion import (
    ActionLayout,
    fuse_relative_ee_action_with_residual,
)
from rlinf_robotwin.geometry.relative_pose import compute_relative_pose_error
from rlinf_robotwin.reward.handover_alignment_reward import (
    HandoverAlignmentRewardConfig,
    compute_handover_alignment_reward,
)
from rlinf_robotwin.safety.residual_safety import (
    ResidualSafetyConfig,
    apply_residual_safety_filter,
)

ActionLayoutType = Literal["ee", "delta_ee", "qpos", "unknown"]
BaseActionMode = Literal["hold", "policy", "scripted", "zero", "random"]
BaseActionProvider = Callable[[Mapping[str, Any], "HandoverResidualEnv"], np.ndarray]

RESIDUAL_DIM = 4
DEFAULT_RESIDUAL_LOW = (-0.02, -0.02, -0.02, -0.10)
DEFAULT_RESIDUAL_HIGH = (0.02, 0.02, 0.02, 0.10)


def default_relative_ee_action_layout() -> ActionLayout:
    """Return the confirmed RoboTwin dual-arm EE action layout."""
    return ActionLayout(
        left_slice=slice(0, 7),
        right_slice=slice(7, 14),
        right_pos_indices=(7, 8, 9),
        right_yaw_index=13,
        action_type="ee",
        action_dim=14,
        pose_format="xyz_quat",
        quat_order="xyzw",
    )


def default_residual_safety_config() -> ResidualSafetyConfig:
    """Return conservative smoke-stage residual clipping limits."""
    return ResidualSafetyConfig(
        per_dim_low=DEFAULT_RESIDUAL_LOW,
        per_dim_high=DEFAULT_RESIDUAL_HIGH,
        max_norm=None,
    )


@dataclass(frozen=True)
class HandoverResidualEnvConfig:
    """Configuration for safe handover residual action execution."""

    action_dim: int = 14
    action_layout_type: ActionLayoutType = "ee"
    action_layout: ActionLayout | None = field(
        default_factory=default_relative_ee_action_layout
    )
    base_action_mode: BaseActionMode = "hold"
    safety_config: ResidualSafetyConfig | None = field(
        default_factory=default_residual_safety_config
    )
    reward_config: HandoverAlignmentRewardConfig | None = None


class HandoverResidualEnv(RoboTwinEnv):
    """RoboTwin handover env for 4D relative EE residual control.

    The residual action contract is ``[dx, dy, dz, dyaw]`` in meters/radians.
    In smoke-stage hold-base mode, the wrapper uses the current ``obs["states"]``
    as the 14D dual-arm EE base action. A later base policy can replace this
    base action provider without changing the relative residual contract.
    """

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
        record_metrics: bool = True,
        residual_cfg: HandoverResidualEnvConfig | None = None,
        base_action_provider: BaseActionProvider | None = None,
    ) -> None:
        require_endpose_enabled(cfg)
        self.residual_cfg = residual_cfg or HandoverResidualEnvConfig()
        self.action_layout_type = self.residual_cfg.action_layout_type.lower()
        self.base_action_provider = base_action_provider
        self._last_obs: dict[str, Any] | None = None
        self._last_handover_error: np.ndarray | None = None
        self._prev_residual: np.ndarray | None = None
        super().__init__(
            cfg=cfg,
            num_envs=num_envs,
            seed_offset=seed_offset,
            total_num_processes=total_num_processes,
            worker_info=worker_info,
            record_metrics=record_metrics,
        )
        self._configure_spaces()

    def reset(
        self,
        env_idx: int | list[int] | None = None,
        env_seeds: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment and cache the initial handover error."""
        obs, infos = super().reset(env_idx=env_idx, env_seeds=env_seeds)
        self._last_obs = obs
        self._last_handover_error = _to_numpy(obs["handover_error"])
        self._prev_residual = None
        return obs, infos

    def step(
        self,
        residual_actions: torch.Tensor | np.ndarray | dict | None = None,
        auto_reset: bool = True,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply a residual action after safety and action-layout validation."""
        residuals = normalize_residual_actions(residual_actions, self.num_envs)
        safe_residuals, safety_infos = self._apply_safety(residuals)
        base_actions, fused_actions = self._build_fused_actions(safe_residuals)

        prev_error = self._last_handover_error
        obs, _, terminations, truncations, infos = super().step(
            fused_actions,
            auto_reset=auto_reset,
        )
        rewards, reward_infos = self._compute_rewards(
            prev_error=prev_error,
            curr_error=_to_numpy(obs["handover_error"]),
            residuals=safe_residuals,
            infos=infos,
        )
        infos["handover_residual"] = {
            **_build_residual_info(residuals, safe_residuals, base_actions, fused_actions),
            "safety": safety_infos,
            "reward": reward_infos,
        }
        self._last_obs = obs
        self._last_handover_error = _to_numpy(obs["handover_error"])
        self._prev_residual = safe_residuals[:, -1, :].copy()
        return obs, rewards, terminations, truncations, infos

    def chunk_step(
        self,
        residual_chunk_actions: torch.Tensor | np.ndarray,
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Apply a residual action chunk after safety and layout validation."""
        residuals = normalize_residual_actions(residual_chunk_actions, self.num_envs)
        safe_residuals, safety_infos = self._apply_safety(residuals)
        base_actions, fused_actions = self._build_fused_actions(safe_residuals)

        prev_error = self._last_handover_error
        obs_list, _, terminations, truncations, infos_list = super().chunk_step(
            fused_actions
        )
        final_obs = obs_list[-1]
        chunk_rewards, reward_infos = self._compute_chunk_rewards(
            prev_error=prev_error,
            curr_error=_to_numpy(final_obs["handover_error"]),
            residuals=safe_residuals,
            infos=infos_list[-1] if infos_list else {},
        )
        if infos_list:
            infos_list[-1]["handover_residual"] = {
                **_build_residual_info(
                    residuals,
                    safe_residuals,
                    base_actions,
                    fused_actions,
                ),
                "safety": safety_infos,
                "reward": reward_infos,
            }
        self._last_obs = final_obs
        self._last_handover_error = _to_numpy(final_obs["handover_error"])
        self._prev_residual = safe_residuals[:, -1, :].copy()
        return obs_list, chunk_rewards, terminations, truncations, infos_list

    def _extract_obs_image(
        self,
        raw_obs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        extracted_obs = super()._extract_obs_image(raw_obs)
        left_poses = []
        right_poses = []
        errors = []
        for obs in raw_obs:
            left_pose = extract_pose_from_raw_obs(obs, "left")
            right_pose = extract_pose_from_raw_obs(obs, "right")
            left_poses.append(left_pose)
            right_poses.append(right_pose)
            errors.append(compute_relative_pose_error(left_pose, right_pose))

        extracted_obs["left_ee_pose"] = torch.as_tensor(
            np.stack(left_poses), dtype=torch.float32
        )
        extracted_obs["right_ee_pose"] = torch.as_tensor(
            np.stack(right_poses), dtype=torch.float32
        )
        extracted_obs["handover_error"] = torch.as_tensor(
            np.stack(errors), dtype=torch.float32
        )
        return extracted_obs

    def _apply_safety(
        self,
        residuals: np.ndarray,
    ) -> tuple[np.ndarray, list[list[dict[str, object]]]]:
        safe_residuals = np.empty_like(residuals, dtype=np.float64)
        safety_infos: list[list[dict[str, object]]] = []
        for env_id in range(residuals.shape[0]):
            env_infos = []
            for step_id in range(residuals.shape[1]):
                safe, info = apply_residual_safety_filter(
                    residual=residuals[env_id, step_id],
                    config=self.residual_cfg.safety_config,
                )
                safe_residuals[env_id, step_id] = safe
                env_infos.append(info)
            safety_infos.append(env_infos)
        return safe_residuals, safety_infos

    def _build_fused_actions(self, residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.action_layout_type == "qpos":
            msg = "Cartesian residual fusion is not supported for qpos actions."
            raise NotImplementedError(msg)
        if self.action_layout_type != "ee":
            msg = (
                "Relative residual fusion requires action_layout_type='ee'; "
                f"got {self.action_layout_type!r}."
            )
            raise ValueError(msg)
        if self.residual_cfg.action_layout is None:
            msg = "action_layout is required for relative EE residual fusion."
            raise ValueError(msg)

        base_actions = self._get_base_actions(residuals)
        if base_actions.shape[:2] != residuals.shape[:2]:
            msg = (
                "base actions must match residual batch and horizon dimensions: "
                f"base={base_actions.shape}, residual={residuals.shape}."
            )
            raise ValueError(msg)

        fused_actions = np.empty_like(base_actions, dtype=np.float64)
        for env_id in range(residuals.shape[0]):
            for step_id in range(residuals.shape[1]):
                fused_actions[env_id, step_id] = fuse_relative_ee_action_with_residual(
                    base_action=np.asarray(base_actions[env_id, step_id]),
                    residual=np.asarray(residuals[env_id, step_id]),
                    layout=self.residual_cfg.action_layout,
                )
        return base_actions, fused_actions

    def _get_base_actions(self, residuals: np.ndarray) -> np.ndarray:
        if self.base_action_provider is not None:
            if self._last_obs is None:
                msg = "reset must be called before requesting base actions."
                raise ValueError(msg)
            base = self.base_action_provider(self._last_obs, self)
            return _normalize_base_actions(
                base,
                self.num_envs,
                residuals.shape[1],
                self.residual_cfg.action_dim,
        )

        if self.residual_cfg.base_action_mode == "hold":
            if self._last_obs is None:
                msg = "reset must be called before using hold-base residual mode."
                raise ValueError(msg)
            base = _extract_states_action(self._last_obs, self.residual_cfg.action_dim)
            return _normalize_base_actions(
                base,
                self.num_envs,
                residuals.shape[1],
                self.residual_cfg.action_dim,
            )
        if self.residual_cfg.base_action_mode == "zero":
            msg = "zero base_action_mode is not valid for absolute EE hold actions."
            raise ValueError(msg)
        if self.residual_cfg.base_action_mode == "random":
            msg = (
                "random base_action_mode is disabled in HandoverResidualEnv; "
                "provide an explicit debug base_action_provider instead."
            )
            raise ValueError(msg)

        msg = (
            "base_action_provider is required when base_action_mode is "
            f"{self.residual_cfg.base_action_mode!r}."
        )
        raise ValueError(msg)

    def _configure_spaces(self) -> None:
        try:
            from gymnasium import spaces
        except ImportError:  # pragma: no cover - depends on runtime env.
            spaces = None

        if spaces is not None:
            low = np.asarray(DEFAULT_RESIDUAL_LOW, dtype=np.float32)
            high = np.asarray(DEFAULT_RESIDUAL_HIGH, dtype=np.float32)
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = getattr(self, "observation_space", None)

    def _compute_rewards(
        self,
        prev_error: np.ndarray | None,
        curr_error: np.ndarray,
        residuals: np.ndarray,
        infos: Mapping[str, Any],
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if prev_error is None:
            prev_error = curr_error
        curr_error = _select_reward_curr_error(curr_error, infos)
        rewards = []
        reward_infos = []
        for env_id in range(self.num_envs):
            reward, reward_info = compute_handover_alignment_reward(
                prev_error=prev_error[env_id],
                curr_error=curr_error[env_id],
                residual=residuals[env_id, -1],
                prev_residual=(
                    self._prev_residual[env_id]
                    if self._prev_residual is not None
                    else None
                ),
                info=_info_for_env(infos, env_id),
                config=self.residual_cfg.reward_config,
            )
            rewards.append(reward)
            reward_infos.append(reward_info)
        return (
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            reward_infos,
        )

    def _compute_chunk_rewards(
        self,
        prev_error: np.ndarray | None,
        curr_error: np.ndarray,
        residuals: np.ndarray,
        infos: Mapping[str, Any],
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        final_rewards, reward_infos = self._compute_rewards(
            prev_error=prev_error,
            curr_error=curr_error,
            residuals=residuals,
            infos=infos,
        )
        chunk_rewards = torch.zeros(
            (self.num_envs, residuals.shape[1]),
            dtype=torch.float32,
            device=self.device,
        )
        chunk_rewards[:, -1] = final_rewards
        return chunk_rewards, reward_infos


def require_endpose_enabled(cfg: Any) -> None:
    """Require RoboTwin ``task_config.data_type.endpose`` to be true."""
    endpose = _nested_get(cfg, ("task_config", "data_type", "endpose"))
    if endpose is not True:
        msg = (
            "HandoverResidualEnv requires task_config.data_type.endpose == True "
            "so raw observations expose left/right EE poses."
        )
        raise ValueError(msg)


def normalize_residual_actions(
    residual_actions: torch.Tensor | np.ndarray | dict | None,
    num_envs: int,
) -> np.ndarray:
    """Normalize residual actions to ``[num_envs, horizon, 4]``."""
    if residual_actions is None:
        msg = "residual_actions must be provided."
        raise ValueError(msg)
    if isinstance(residual_actions, dict):
        residual_actions = residual_actions.get(
            "residual_actions",
            residual_actions.get("residuals", residual_actions.get("actions")),
        )
    if isinstance(residual_actions, torch.Tensor):
        residual_actions = residual_actions.detach().cpu().numpy()

    residuals = np.asarray(residual_actions, dtype=np.float64)
    if residuals.ndim == 1:
        residuals = residuals.reshape(1, 1, RESIDUAL_DIM)
    elif residuals.ndim == 2:
        residuals = residuals[:, None, :]
    if residuals.ndim != 3 or residuals.shape[-1] != RESIDUAL_DIM:
        msg = (
            "residual_actions must have shape (4,), (num_envs, 4), or "
            f"(num_envs, horizon, 4), got {residuals.shape}."
        )
        raise ValueError(msg)
    if residuals.shape[0] != num_envs:
        msg = (
            f"residual num_envs mismatch: expected {num_envs}, "
            f"got {residuals.shape[0]}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(residuals)):
        msg = "residual_actions must contain only finite values."
        raise ValueError(msg)
    return residuals


def extract_pose_from_raw_obs(
    obs: Mapping[str, Any],
    side: Literal["left", "right"],
) -> np.ndarray:
    """Extract a 7D EE pose from one raw RoboTwin observation."""
    state_pose = _extract_pose_from_state(obs, side)
    if state_pose is not None:
        return state_pose

    candidates = _pose_candidates(side)
    for key in candidates:
        found, value = _get_path(obs, key)
        if found:
            return _as_pose7(value, key)

    raw_keys = sorted(str(key) for key in obs.keys())
    msg = (
        f"Could not find {side}_ee_pose in raw RoboTwin observation. "
        f"Tried {candidates}; raw obs keys are {raw_keys}."
    )
    raise ValueError(msg)


def _extract_pose_from_state(
    obs: Mapping[str, Any],
    side: Literal["left", "right"],
) -> np.ndarray | None:
    found, value = _get_path(obs, "state")
    if not found:
        found, value = _get_path(obs, "states")
    if not found:
        return None

    state = np.asarray(_to_numpy(value), dtype=np.float64).reshape(-1)
    if state.shape != (14,):
        return None
    if not np.all(np.isfinite(state)):
        msg = "state must contain only finite values."
        raise ValueError(msg)
    return state[0:7] if side == "left" else state[7:14]


def _pose_candidates(side: str) -> tuple[str, ...]:
    prefix = side
    return (
        f"{prefix}_ee_pose",
        f"{prefix}_end_effector_pose",
        f"{prefix}_eef_pose",
        f"{prefix}_tcp_pose",
        f"{prefix}_gripper_pose",
        f"{prefix}_endpose",
        f"endpose.{prefix}",
        f"ee_pose.{prefix}",
        f"eef_pose.{prefix}",
        f"tcp_pose.{prefix}",
        f"{prefix}.ee_pose",
        f"{prefix}.endpose",
        f"{prefix}.tcp_pose",
    )


def _get_path(mapping: Mapping[str, Any], dotted_key: str) -> tuple[bool, Any]:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _as_pose7(value: Any, key: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,):
        msg = f"{key} must be a 7D pose [x, y, z, qx, qy, qz, qw], got {pose.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(pose)):
        msg = f"{key} must contain only finite values."
        raise ValueError(msg)
    return pose


def _normalize_base_actions(
    base_actions: np.ndarray,
    num_envs: int,
    horizon: int,
    action_dim: int,
) -> np.ndarray:
    base = np.asarray(base_actions, dtype=np.float64)
    if base.ndim == 1:
        base = base.reshape(1, 1, action_dim)
    elif base.ndim == 2:
        base = base[:, None, :]
    if base.ndim != 3 or base.shape[-1] != action_dim:
        msg = (
            "base actions must have shape (action_dim,), (num_envs, action_dim), "
            f"or (num_envs, horizon, action_dim), got {base.shape}."
        )
        raise ValueError(msg)
    if base.shape[0] != num_envs:
        msg = (
            f"base action num_envs mismatch: expected {num_envs}, "
            f"got {base.shape[0]}."
        )
        raise ValueError(msg)
    if base.shape[1] == 1 and horizon != 1:
        base = np.repeat(base, horizon, axis=1)
    if base.shape[1] != horizon:
        msg = f"base action horizon mismatch: expected {horizon}, got {base.shape[1]}."
        raise ValueError(msg)
    if not np.all(np.isfinite(base)):
        msg = "base actions must contain only finite values."
        raise ValueError(msg)
    return base


def _extract_states_action(obs: Mapping[str, Any], action_dim: int) -> np.ndarray:
    if "states" not in obs:
        msg = "hold-base residual mode requires obs['states']."
        raise ValueError(msg)
    states = _to_numpy(obs["states"])
    base = np.asarray(states, dtype=np.float64)
    if base.shape[-1:] != (action_dim,):
        msg = f"obs['states'] must have shape (..., {action_dim}), got {base.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(base)):
        msg = "obs['states'] must contain only finite values."
        raise ValueError(msg)
    return base


def _build_residual_info(
    residual_raw: np.ndarray,
    residual_clipped: np.ndarray,
    base_action: np.ndarray,
    fused_action: np.ndarray,
) -> dict[str, Any]:
    return {
        "residual_raw": residual_raw.copy(),
        "residual_clipped": residual_clipped.copy(),
        "base_action": base_action.copy(),
        "fused_action": fused_action.copy(),
        "action_type": "ee",
        "fusion_type": "relative_ee",
    }


def _nested_get(obj: Any, path: Sequence[str]) -> Any:
    current = obj
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None
    return current


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _info_for_env(infos: Mapping[str, Any], env_id: int) -> dict[str, Any]:
    result = {}
    for key, value in infos.items():
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] > env_id:
                result[key] = value[env_id].item()
            continue
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] > env_id:
                item = value[env_id]
                result[key] = (
                    item.item() if getattr(item, "shape", None) == () else item
                )
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) > env_id:
                result[key] = value[env_id]
            continue
        result[key] = value
    return result


def _select_reward_curr_error(
    curr_error: np.ndarray,
    infos: Mapping[str, Any],
) -> np.ndarray:
    final_obs = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if not isinstance(final_obs, Mapping) or "handover_error" not in final_obs:
        return curr_error
    if final_mask is None:
        return curr_error

    selected = np.array(curr_error, copy=True)
    mask = _to_numpy(final_mask).reshape(-1).astype(bool)
    final_error = _to_numpy(final_obs["handover_error"])
    for env_id, use_final in enumerate(mask):
        if use_final:
            selected[env_id] = final_error[env_id]
    return selected
