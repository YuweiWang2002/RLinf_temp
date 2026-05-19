import math

import numpy as np
import pytest

from rlinf_robotwin.control.action_fusion import (
    ActionLayout,
    fuse_relative_ee_action_with_residual,
    fuse_residual_action,
)


def _layout() -> ActionLayout:
    return ActionLayout(
        left_slice=slice(0, 4),
        right_slice=slice(4, 9),
        right_pos_indices=(4, 5, 6),
        right_yaw_index=7,
        gripper_indices=(3, 8),
    )


def _ee_layout(action_type: str = "ee", action_dim: int = 14) -> ActionLayout:
    return ActionLayout(
        left_slice=slice(0, 7),
        right_slice=slice(7, 14),
        right_pos_indices=(7, 8, 9),
        right_yaw_index=13,
        action_type=action_type,
        action_dim=action_dim,
        pose_format="xyz_quat",
        quat_order="xyzw",
    )


def _dual_identity_action() -> np.ndarray:
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


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])


def test_zero_residual_returns_base_action_copy() -> None:
    base_action = np.arange(10, dtype=np.float64)
    fused_action = fuse_residual_action(
        base_action,
        np.zeros(4, dtype=np.float64),
        _layout(),
    )

    np.testing.assert_array_equal(fused_action, base_action)
    assert fused_action is not base_action


def test_fuses_residual_only_into_right_arm_dimensions() -> None:
    base_action = np.arange(10, dtype=np.float64)
    residual_action = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)

    fused_action = fuse_residual_action(base_action, residual_action, _layout())

    expected_action = base_action.copy()
    expected_action[[4, 5, 6]] += residual_action[:3]
    expected_action[7] += residual_action[3]
    np.testing.assert_allclose(fused_action, expected_action)
    np.testing.assert_array_equal(fused_action[0:4], base_action[0:4])


def test_gripper_indices_are_unchanged() -> None:
    base_action = np.arange(10, dtype=np.float64)
    residual_action = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    fused_action = fuse_residual_action(base_action, residual_action, _layout())

    np.testing.assert_array_equal(fused_action[[3, 8]], base_action[[3, 8]])


@pytest.mark.parametrize(
    "residual_action",
    [
        np.zeros(3, dtype=np.float64),
        np.zeros((1, 4), dtype=np.float64),
        np.zeros((4, 1), dtype=np.float64),
    ],
)
def test_rejects_invalid_residual_shape(residual_action: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        fuse_residual_action(np.zeros(10, dtype=np.float64), residual_action, _layout())


def test_rejects_layout_indices_out_of_bounds() -> None:
    invalid_layout = ActionLayout(
        left_slice=slice(0, 4),
        right_slice=slice(4, 9),
        right_pos_indices=(4, 5, 10),
        right_yaw_index=7,
    )

    with pytest.raises(ValueError, match="out of bounds"):
        fuse_residual_action(
            np.zeros(10, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            invalid_layout,
        )


def test_rejects_residual_indices_outside_right_slice() -> None:
    invalid_layout = ActionLayout(
        left_slice=slice(0, 4),
        right_slice=slice(4, 7),
        right_pos_indices=(4, 5, 6),
        right_yaw_index=7,
    )

    with pytest.raises(ValueError, match="right_slice"):
        fuse_residual_action(
            np.zeros(10, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            invalid_layout,
        )


def test_rejects_residual_indices_overlapping_gripper_indices() -> None:
    invalid_layout = ActionLayout(
        left_slice=slice(0, 4),
        right_slice=slice(4, 9),
        right_pos_indices=(4, 5, 6),
        right_yaw_index=7,
        gripper_indices=(7, 8),
    )

    with pytest.raises(ValueError, match="gripper"):
        fuse_residual_action(
            np.zeros(10, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            invalid_layout,
        )


def test_layout_is_not_tied_to_robotwin_action_indices() -> None:
    base_action = np.arange(12, dtype=np.float64)
    residual_action = np.array([-0.5, 0.25, 0.75, -1.0], dtype=np.float64)
    layout = ActionLayout(
        left_slice=slice(6, 9),
        right_slice=slice(0, 5),
        right_pos_indices=(1, 2, 3),
        right_yaw_index=4,
        gripper_indices=(0, 10),
    )

    fused_action = fuse_residual_action(base_action, residual_action, layout)

    expected_action = base_action.copy()
    expected_action[[1, 2, 3]] += residual_action[:3]
    expected_action[4] += residual_action[3]
    np.testing.assert_allclose(fused_action, expected_action)
    np.testing.assert_array_equal(fused_action[6:9], base_action[6:9])
    np.testing.assert_array_equal(fused_action[[0, 10]], base_action[[0, 10]])


def test_relative_ee_fusion_identity_dx() -> None:
    base_action = _dual_identity_action()
    residual = np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float64)

    fused = fuse_relative_ee_action_with_residual(base_action, residual, _ee_layout())

    np.testing.assert_allclose(fused[0:7], base_action[0:7])
    assert fused[7] == pytest.approx(0.01)
    np.testing.assert_allclose(fused[10:14], base_action[10:14], atol=1e-12)
    np.testing.assert_allclose(base_action[7], 0.0)


def test_relative_ee_fusion_left_reference_translation() -> None:
    base_action = _dual_identity_action()
    base_action[0:3] = np.array([1.0, 2.0, 3.0])
    base_action[7:10] = np.array([1.0, 2.0, 3.0])
    residual = np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float64)

    fused = fuse_relative_ee_action_with_residual(base_action, residual, _ee_layout())

    assert fused[7] == pytest.approx(1.01)
    assert fused[8] == pytest.approx(2.0)
    assert fused[9] == pytest.approx(3.0)


def test_relative_ee_fusion_rotated_left_frame() -> None:
    base_action = _dual_identity_action()
    left_quat = _yaw_quat(math.pi / 2.0)
    base_action[3:7] = left_quat
    base_action[10:14] = left_quat
    residual = np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float64)

    fused = fuse_relative_ee_action_with_residual(base_action, residual, _ee_layout())

    assert fused[7] == pytest.approx(0.0, abs=1e-12)
    assert fused[8] == pytest.approx(0.01)
    assert not np.isclose(fused[7] - base_action[7], 0.01)


def test_relative_ee_fusion_dyaw() -> None:
    base_action = _dual_identity_action()
    residual = np.array([0.0, 0.0, 0.0, math.pi / 2.0], dtype=np.float64)

    fused = fuse_relative_ee_action_with_residual(base_action, residual, _ee_layout())

    np.testing.assert_allclose(fused[10:14], _yaw_quat(math.pi / 2.0), atol=1e-9)


def test_relative_ee_fusion_left_unchanged() -> None:
    base_action = _dual_identity_action()
    base_action[0:7] = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.1, 0.995])
    residual = np.array([0.0, -0.01, 0.01, 0.02], dtype=np.float64)

    fused = fuse_relative_ee_action_with_residual(base_action, residual, _ee_layout())

    np.testing.assert_array_equal(fused[0:7], base_action[0:7])


def test_relative_ee_fusion_quat_normalized() -> None:
    base_action = _dual_identity_action()
    base_action[10:14] = np.array([0.0, 0.0, 0.2, 2.0])
    residual = np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float64)

    fused = fuse_relative_ee_action_with_residual(base_action, residual, _ee_layout())

    assert np.linalg.norm(fused[10:14]) == pytest.approx(1.0)


def test_relative_ee_fusion_rejects_qpos() -> None:
    with pytest.raises(
        NotImplementedError,
        match="Cartesian residual fusion is not supported for qpos actions",
    ):
        fuse_relative_ee_action_with_residual(
            _dual_identity_action(),
            np.zeros(4, dtype=np.float64),
            _ee_layout(action_type="qpos"),
        )


def test_relative_ee_fusion_rejects_wrong_action_dim() -> None:
    with pytest.raises(ValueError, match="action_dim must be 14"):
        fuse_relative_ee_action_with_residual(
            np.zeros(13, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            _ee_layout(action_dim=13),
        )


def test_relative_ee_fusion_rejects_wrong_residual_dim() -> None:
    with pytest.raises(ValueError, match="residual must have shape"):
        fuse_relative_ee_action_with_residual(
            _dual_identity_action(),
            np.zeros(3, dtype=np.float64),
            _ee_layout(),
        )


def test_relative_ee_fusion_torch_batch() -> None:
    torch = pytest.importorskip("torch")
    base_action = torch.tensor(
        np.stack([_dual_identity_action(), _dual_identity_action()]),
        dtype=torch.float64,
    )
    residual = torch.tensor(
        [[0.01, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 0.0]],
        dtype=torch.float64,
    )

    fused = fuse_relative_ee_action_with_residual(base_action, residual, _ee_layout())

    assert isinstance(fused, torch.Tensor)
    assert fused.device == base_action.device
    assert fused.dtype == base_action.dtype
    torch.testing.assert_close(fused[:, 0:7], base_action[:, 0:7])
    torch.testing.assert_close(fused[:, 7:10], residual[:, 0:3])
