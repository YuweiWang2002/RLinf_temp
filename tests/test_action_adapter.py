import math

import numpy as np
import pytest

from rlinf_robotwin.control.action_adapter import (
    endpose16_to_openpi_ee14,
    endpose16_to_pose14,
    openpi_ee14_to_endpose16,
    openpi_ee14_to_pose14,
    pose14_to_endpose16,
    quat_xyzw_to_rotvec,
    rotvec_to_quat_xyzw,
)


def _identity_pose14() -> np.ndarray:
    return np.array(
        [
            0.1,
            0.2,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
            0.4,
            0.5,
            0.6,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )


def _endpose16() -> np.ndarray:
    return np.array(
        [
            0.1,
            0.2,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
            0.7,
            0.4,
            0.5,
            0.6,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.3,
        ],
        dtype=np.float64,
    )


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])


def _same_quat(actual: np.ndarray, expected: np.ndarray) -> None:
    if np.dot(actual, expected) < 0:
        actual = -actual
    np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_endpose16_to_pose14_drops_grippers() -> None:
    pose14 = endpose16_to_pose14(_endpose16())

    expected = np.concatenate((_endpose16()[0:7], _endpose16()[8:15]))
    np.testing.assert_allclose(pose14, expected)


def test_pose14_to_endpose16_inserts_grippers() -> None:
    endpose16 = pose14_to_endpose16(_identity_pose14(), 0.7, -0.3)

    np.testing.assert_allclose(endpose16, _endpose16())


def test_quat_identity_to_rotvec_zero() -> None:
    rotvec = quat_xyzw_to_rotvec(np.array([0.0, 0.0, 0.0, 1.0]))

    np.testing.assert_allclose(rotvec, np.zeros(3), atol=1e-12)


def test_rotvec_zero_to_quat_identity() -> None:
    quat = rotvec_to_quat_xyzw(np.zeros(3))

    np.testing.assert_allclose(quat, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12)


def test_yaw_90_quat_rotvec_round_trip() -> None:
    quat = _yaw_quat(math.pi / 2.0)
    rotvec = quat_xyzw_to_rotvec(quat)
    round_trip = rotvec_to_quat_xyzw(rotvec)

    np.testing.assert_allclose(rotvec, np.array([0.0, 0.0, math.pi / 2.0]), atol=1e-9)
    _same_quat(round_trip, quat)


def test_endpose16_to_openpi_ee14_shape() -> None:
    openpi_action = endpose16_to_openpi_ee14(_endpose16())

    assert openpi_action.shape == (14,)
    np.testing.assert_allclose(openpi_action[[6, 13]], np.array([0.7, -0.3]))


def test_openpi_ee14_to_pose14_drops_grippers() -> None:
    openpi_action = endpose16_to_openpi_ee14(_endpose16())
    pose14 = openpi_ee14_to_pose14(openpi_action)

    assert pose14.shape == (14,)
    np.testing.assert_allclose(pose14, _identity_pose14(), atol=1e-9)


def test_openpi_ee14_to_endpose16_preserves_grippers() -> None:
    openpi_action = endpose16_to_openpi_ee14(_endpose16())
    endpose16 = openpi_ee14_to_endpose16(openpi_action)

    assert endpose16.shape == (16,)
    np.testing.assert_allclose(endpose16, _endpose16(), atol=1e-9)


def test_batch_inputs_are_supported() -> None:
    batch = np.stack([_endpose16(), _endpose16()])

    openpi_action = endpose16_to_openpi_ee14(batch)
    pose14 = openpi_ee14_to_pose14(openpi_action)
    endpose16 = openpi_ee14_to_endpose16(openpi_action)

    assert openpi_action.shape == (2, 14)
    assert pose14.shape == (2, 14)
    assert endpose16.shape == (2, 16)
    np.testing.assert_allclose(endpose16, batch, atol=1e-9)


@pytest.mark.parametrize(
    ("func", "bad_input"),
    [
        (endpose16_to_pose14, np.zeros(15)),
        (endpose16_to_openpi_ee14, np.zeros(15)),
        (openpi_ee14_to_pose14, np.zeros(16)),
        (openpi_ee14_to_endpose16, np.zeros(16)),
        (quat_xyzw_to_rotvec, np.zeros(3)),
        (rotvec_to_quat_xyzw, np.zeros(4)),
    ],
)
def test_shape_errors_raise_value_error(func, bad_input: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        func(bad_input)


def test_pose14_to_endpose16_shape_error() -> None:
    with pytest.raises(ValueError, match="shape"):
        pose14_to_endpose16(np.zeros(13), 0.0, 0.0)


def test_pose14_to_endpose16_gripper_shape_error() -> None:
    with pytest.raises(ValueError, match="left_gripper"):
        pose14_to_endpose16(np.zeros((2, 14)), np.zeros(3), np.zeros(2))
