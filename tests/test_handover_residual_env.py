"""Unit tests for the RoboTwin handover residual wrapper logic."""

import numpy as np
import pytest

from rlinf_robotwin.envs.handover_residual_env import (
    HandoverResidualEnv,
    HandoverResidualEnvConfig,
    _build_residual_info,
    extract_pose_from_raw_obs,
    normalize_residual_actions,
)


def _identity_states() -> np.ndarray:
    return np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )


def _wrapper(
    *,
    action_layout_type: str = "ee",
    states: np.ndarray | None = None,
) -> HandoverResidualEnv:
    env = object.__new__(HandoverResidualEnv)
    env.num_envs = 1
    env.residual_cfg = HandoverResidualEnvConfig(
        action_layout_type=action_layout_type,
    )
    env.action_layout_type = action_layout_type
    env.base_action_provider = None
    env._last_obs = {"states": np.asarray(states if states is not None else _identity_states())}
    env._prev_residual = None
    return env


def test_residual_action_shape_must_be_4d() -> None:
    with pytest.raises(ValueError, match="shape"):
        normalize_residual_actions(np.zeros(3), num_envs=1)


def test_residual_is_safety_clipped() -> None:
    env = _wrapper()
    residuals = normalize_residual_actions(
        np.array([0.05, -0.05, 0.03, 0.20]),
        num_envs=1,
    )

    clipped, safety_infos = env._apply_safety(residuals)

    np.testing.assert_allclose(clipped[0, 0], np.array([0.02, -0.02, 0.02, 0.10]))
    assert safety_infos[0][0]["per_dim_clipped"] is True


def test_qpos_layout_disallows_cartesian_fusion() -> None:
    env = _wrapper(action_layout_type="qpos")

    with pytest.raises(NotImplementedError, match="not supported for qpos"):
        env._build_fused_actions(np.zeros((1, 1, 4), dtype=np.float64))


def test_hold_base_zero_residual_produces_hold_action() -> None:
    env = _wrapper()

    base_actions, fused_actions = env._build_fused_actions(
        np.zeros((1, 1, 4), dtype=np.float64)
    )

    np.testing.assert_allclose(base_actions[0, 0], _identity_states())
    np.testing.assert_allclose(fused_actions[0, 0], _identity_states())


def test_relative_residual_calls_fusion_and_outputs_14d_action() -> None:
    env = _wrapper()

    _, fused_actions = env._build_fused_actions(
        np.array([[[0.01, 0.0, 0.0, 0.0]]], dtype=np.float64)
    )

    assert fused_actions.shape == (1, 1, 14)
    assert fused_actions[0, 0, 7] == pytest.approx(0.01)
    np.testing.assert_allclose(fused_actions[0, 0, 0:7], _identity_states()[0:7])


def test_residual_info_contains_smoke_fields() -> None:
    residual_raw = np.array([[[0.03, 0.0, 0.0, 0.0]]], dtype=np.float64)
    residual_clipped = np.array([[[0.02, 0.0, 0.0, 0.0]]], dtype=np.float64)
    base_action = _identity_states().reshape(1, 1, 14)
    fused_action = base_action.copy()
    fused_action[0, 0, 7] = 0.02

    info = _build_residual_info(
        residual_raw,
        residual_clipped,
        base_action,
        fused_action,
    )

    assert {
        "residual_raw",
        "residual_clipped",
        "base_action",
        "fused_action",
        "fusion_type",
    } <= set(info)
    assert info["fusion_type"] == "relative_ee"
    np.testing.assert_allclose(info["fused_action"], fused_action)


def test_extract_pose_from_raw_obs_uses_14d_state_fallback() -> None:
    raw_obs = {"state": _identity_states()}

    np.testing.assert_allclose(
        extract_pose_from_raw_obs(raw_obs, "left"),
        _identity_states()[0:7],
    )
    np.testing.assert_allclose(
        extract_pose_from_raw_obs(raw_obs, "right"),
        _identity_states()[7:14],
    )
