import numpy as np

from scripts.convert_residual_bc_targets_to_replay import (
    build_next_obs_and_done,
    has_nan_or_inf,
)


def test_build_next_obs_and_done_uses_next_frame_within_episode():
    obs = np.arange(4 * 21, dtype=np.float32).reshape(4, 21)
    episode_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
    frame_index = np.asarray([0, 1, 0, 2], dtype=np.int64)

    next_obs, done = build_next_obs_and_done(obs, episode_id=episode_id, frame_index=frame_index)

    np.testing.assert_array_equal(next_obs[0], obs[1])
    np.testing.assert_array_equal(next_obs[1], obs[1])
    np.testing.assert_array_equal(next_obs[2], obs[2])
    assert done.tolist() == [False, True, True, True]


def test_has_nan_or_inf_flags_rows():
    first = np.zeros((3, 2), dtype=np.float32)
    second = np.zeros((3, 1), dtype=np.float32)
    second[1, 0] = np.inf

    flags = has_nan_or_inf(first, second)

    assert flags.tolist() == [0, 1, 0]
