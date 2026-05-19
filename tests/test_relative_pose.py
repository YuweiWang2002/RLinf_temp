"""Tests for RobotWin relative pose geometry utilities."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf_robotwin.geometry.relative_pose import compute_relative_pose_error


def test_same_pose_returns_zero() -> None:
    pose = np.array([0.2, -0.1, 0.4, 0.0, 0.0, 0.0, 1.0])

    relative_error = compute_relative_pose_error(pose, pose)

    assert relative_error.shape == (4,)
    np.testing.assert_allclose(relative_error, np.zeros(4), atol=1e-9)


def test_right_pose_x_offset_relative_to_left() -> None:
    left_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    right_pose = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    relative_error = compute_relative_pose_error(left_pose, right_pose)

    np.testing.assert_allclose(
        relative_error,
        np.array([0.1, 0.0, 0.0, 0.0]),
        atol=1e-9,
    )


def test_right_pose_yaw_rotated_90deg_relative_to_left() -> None:
    left_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    yaw_quat_xyzw = Rotation.from_euler("z", math.pi / 2).as_quat()
    right_pose = np.concatenate([np.zeros(3), yaw_quat_xyzw])

    relative_error = compute_relative_pose_error(left_pose, right_pose)

    np.testing.assert_allclose(
        relative_error,
        np.array([0.0, 0.0, 0.0, math.pi / 2]),
        atol=1e-9,
    )


def test_pose_shape_must_be_seven() -> None:
    left_pose = np.zeros(6)
    right_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="Expected pose shape"):
        compute_relative_pose_error(left_pose, right_pose)
