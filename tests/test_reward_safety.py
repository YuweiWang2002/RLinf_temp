"""Tests for RoboTwin handover reward and residual safety filter."""

import numpy as np
import pytest

from rlinf_robotwin.reward.handover_alignment_reward import (
    HandoverAlignmentRewardConfig,
    compute_handover_alignment_reward,
)
from rlinf_robotwin.safety.residual_safety import (
    ResidualSafetyConfig,
    apply_residual_safety_filter,
)


def test_error_decrease_reward_increases() -> None:
    """A larger alignment improvement should produce a larger reward."""
    residual = np.zeros(4)
    config = HandoverAlignmentRewardConfig(success_bonus=0.0)

    larger_decrease, _ = compute_handover_alignment_reward(
        prev_error=np.array([1.0, 0.0, 0.0, 0.0]),
        curr_error=np.array([0.5, 0.0, 0.0, 0.0]),
        residual=residual,
        config=config,
    )
    smaller_decrease, _ = compute_handover_alignment_reward(
        prev_error=np.array([1.0, 0.0, 0.0, 0.0]),
        curr_error=np.array([0.8, 0.0, 0.0, 0.0]),
        residual=residual,
        config=config,
    )

    assert larger_decrease > smaller_decrease


def test_error_increase_reward_lowers() -> None:
    """Alignment regression should be rewarded less than alignment progress."""
    residual = np.zeros(4)
    config = HandoverAlignmentRewardConfig(success_bonus=0.0)

    regressed, _ = compute_handover_alignment_reward(
        prev_error=np.array([0.5, 0.0, 0.0, 0.0]),
        curr_error=np.array([0.8, 0.0, 0.0, 0.0]),
        residual=residual,
        config=config,
    )
    improved, _ = compute_handover_alignment_reward(
        prev_error=np.array([0.5, 0.0, 0.0, 0.0]),
        curr_error=np.array([0.4, 0.0, 0.0, 0.0]),
        residual=residual,
        config=config,
    )

    assert regressed < improved


def test_success_bonus_and_standardized_info() -> None:
    """Success metadata should add the configured bonus and be standardized."""
    config = HandoverAlignmentRewardConfig(success_bonus=3.0)
    kwargs = {
        "prev_error": np.array([0.5, 0.0, 0.0, 0.0]),
        "curr_error": np.array([0.4, 0.0, 0.0, 0.0]),
        "residual": np.zeros(4),
        "config": config,
    }

    base_reward, base_info = compute_handover_alignment_reward(
        **kwargs,
        info={"is_success": False},
    )
    success_reward, success_info = compute_handover_alignment_reward(
        **kwargs,
        info={"is_success": True},
    )

    assert success_reward == pytest.approx(base_reward + config.success_bonus)
    assert base_info["success"] is False
    assert success_info["success"] is True
    assert {"e_xy", "e_z", "e_yaw", "residual_norm"} <= set(success_info)


def test_collision_penalty_and_standardized_info() -> None:
    """Collision metadata should subtract the configured penalty."""
    config = HandoverAlignmentRewardConfig(collision_penalty=4.0, success_bonus=0.0)
    kwargs = {
        "prev_error": np.array([0.5, 0.0, 0.0, 0.0]),
        "curr_error": np.array([0.4, 0.0, 0.0, 0.0]),
        "residual": np.zeros(4),
        "config": config,
    }

    base_reward, base_info = compute_handover_alignment_reward(
        **kwargs,
        info={"has_collision": False},
    )
    collision_reward, collision_info = compute_handover_alignment_reward(
        **kwargs,
        info={"has_collision": True},
    )

    assert collision_reward == pytest.approx(base_reward - config.collision_penalty)
    assert base_info["collision"] is False
    assert collision_info["collision"] is True


def test_timeout_penalty_is_configurable() -> None:
    """Timeout penalty should only affect reward when configured."""
    kwargs = {
        "prev_error": np.array([0.5, 0.0, 0.0, 0.0]),
        "curr_error": np.array([0.4, 0.0, 0.0, 0.0]),
        "residual": np.zeros(4),
        "info": {"timeout": True},
    }
    no_penalty_reward, _ = compute_handover_alignment_reward(
        **kwargs,
        config=HandoverAlignmentRewardConfig(success_bonus=0.0),
    )
    penalty_reward, penalty_info = compute_handover_alignment_reward(
        **kwargs,
        config=HandoverAlignmentRewardConfig(
            success_bonus=0.0,
            timeout_penalty=1.5,
        ),
    )

    assert penalty_reward == pytest.approx(no_penalty_reward - 1.5)
    assert penalty_info["timeout"] is True


def test_reward_rejects_non_vector4_inputs() -> None:
    """Reward inputs must match the frozen handover error shape."""
    with pytest.raises(ValueError, match="prev_error must have shape"):
        compute_handover_alignment_reward(
            prev_error=np.zeros(3),
            curr_error=np.zeros(4),
            residual=np.zeros(4),
        )


def test_residual_default_per_dim_clip() -> None:
    """Default per-dim limits must match the frozen residual contract."""
    clipped, safety_info = apply_residual_safety_filter(
        np.array([0.2, -0.2, 0.2, -0.2]),
        config=ResidualSafetyConfig(max_norm=None),
    )

    np.testing.assert_allclose(clipped, np.array([0.01, -0.01, 0.01, -0.087]))
    assert safety_info["per_dim_clipped"] is True


def test_residual_max_norm_clip() -> None:
    """Residual norm should be scaled down after per-dim clipping."""
    clipped, safety_info = apply_residual_safety_filter(
        np.array([0.01, 0.01, 0.01, 0.0]),
        config=ResidualSafetyConfig(max_norm=0.01),
    )

    assert np.linalg.norm(clipped) <= 0.01 + 1e-12
    assert safety_info["norm_clipped"] is True


def test_residual_zero_unchanged() -> None:
    """A zero residual should remain exactly zero."""
    clipped, safety_info = apply_residual_safety_filter(np.zeros(4))

    np.testing.assert_array_equal(clipped, np.zeros(4))
    assert safety_info["per_dim_clipped"] is False
    assert safety_info["norm_clipped"] is False
    assert safety_info["workspace_clipped"] is False


def test_residual_workspace_bound() -> None:
    """Workspace bounds should clip the target pose and return residual delta."""
    clipped, safety_info = apply_residual_safety_filter(
        np.array([0.01, 0.0, 0.0, 0.0]),
        config=ResidualSafetyConfig(
            max_norm=None,
            workspace_low=(-1.0, -1.0, -1.0, -1.0),
            workspace_high=(0.005, 1.0, 1.0, 1.0),
        ),
        current_position=np.zeros(4),
    )

    np.testing.assert_allclose(clipped, np.array([0.005, 0.0, 0.0, 0.0]))
    assert safety_info["workspace_clipped"] is True


def test_residual_shape_is_strictly_vector4() -> None:
    """Residual safety filter should reject any non-``(4,)`` residual."""
    with pytest.raises(ValueError, match="residual must have shape"):
        apply_residual_safety_filter(np.zeros((1, 4)))
