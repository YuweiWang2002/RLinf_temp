import numpy as np

from rlinf.algorithms.residual_td3.gate_labeler import (
    BinaryGateLabelerConfig,
    BinaryResidualGateLabeler,
)


def _label(
    left_pos,
    right_pos,
    left_gripper=None,
    right_gripper=None,
    **kwargs,
):
    left_pos = np.asarray(left_pos, dtype=np.float64)
    right_pos = np.asarray(right_pos, dtype=np.float64)
    length = left_pos.shape[0]
    if left_gripper is None:
        left_gripper = np.zeros(length, dtype=np.float64)
    if right_gripper is None:
        right_gripper = np.zeros(length, dtype=np.float64)
    cfg = BinaryGateLabelerConfig(
        distance_small_threshold=kwargs.pop("distance_small_threshold", 0.2),
        gripper_closed_threshold=kwargs.pop("gripper_closed_threshold", 0.5),
        contact_dilation=kwargs.pop("contact_dilation", 0),
        min_positive_segment_len=kwargs.pop("min_positive_segment_len", 1),
        **kwargs,
    )
    return BinaryResidualGateLabeler(cfg).label_episode(
        left_ee_pos=left_pos,
        right_ee_pos=right_pos,
        left_gripper=np.asarray(left_gripper, dtype=np.float64),
        right_gripper=np.asarray(right_gripper, dtype=np.float64),
    )


def test_single_arm_motion_with_open_grippers_is_all_zero():
    left_pos = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
    right_pos = np.zeros((4, 3))

    result = _label(left_pos, right_pos)

    assert np.all(result["w_binary"] == 0)


def test_both_arms_far_distance_motion_is_zero():
    left_pos = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
    right_pos = left_pos + np.array([1.0, 0.0, 0.0])

    result = _label(left_pos, right_pos, distance_small_threshold=0.2)

    assert np.all(result["w_binary"] == 0)


def test_both_arms_near_distance_motion_is_positive():
    left_pos = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
    right_pos = left_pos + np.array([0.05, 0.0, 0.0])

    result = _label(left_pos, right_pos, distance_small_threshold=0.2)

    np.testing.assert_array_equal(result["w_binary"], np.array([0, 1, 1, 1]))


def test_both_closed_grippers_are_positive():
    left_pos = np.zeros((4, 3))
    right_pos = np.ones((4, 3))

    result = _label(
        left_pos,
        right_pos,
        left_gripper=np.ones(4),
        right_gripper=np.ones(4),
    )

    assert np.all(result["w_binary"] == 1)


def test_contact_dilation_expands_single_contact_frame():
    left_pos = np.zeros((7, 3))
    right_pos = np.ones((7, 3))
    left_gripper = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
    right_gripper = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)

    result = _label(
        left_pos,
        right_pos,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        contact_dilation=2,
    )

    np.testing.assert_array_equal(result["w_binary"], np.array([0, 1, 1, 1, 1, 1, 0]))


def test_min_positive_segment_len_removes_short_segments():
    left_pos = np.zeros((5, 3))
    right_pos = np.ones((5, 3))
    left_gripper = np.array([0, 1, 0, 0, 0], dtype=np.float64)
    right_gripper = np.array([0, 1, 0, 0, 0], dtype=np.float64)

    result = _label(
        left_pos,
        right_pos,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
        min_positive_segment_len=2,
    )

    assert np.all(result["w_binary"] == 0)


def test_output_is_binary():
    left_pos = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
    right_pos = left_pos + np.array([0.05, 0.0, 0.0])

    result = _label(left_pos, right_pos)

    assert set(np.unique(result["w_binary"])).issubset({0, 1})
