from pathlib import Path

import numpy as np

from scripts.convert_hard_expert_hdf5_to_residual_bc_targets import (
    load_success_metadata,
    make_state_action_pairs,
    resolve_demo_data_dir,
    select_episode_paths,
)


def test_resolve_demo_data_dir_accepts_successful_demo_root(tmp_path):
    data_dir = tmp_path / "successful_demos" / "data"
    data_dir.mkdir(parents=True)

    assert resolve_demo_data_dir(tmp_path / "successful_demos") == data_dir


def test_resolve_demo_data_dir_accepts_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    assert resolve_demo_data_dir(data_dir) == data_dir


def test_load_success_metadata_keeps_only_success_rows(tmp_path):
    path = tmp_path / "expert_success_summary.csv"
    path.write_text(
        "episode_id,seed,expert_success,failure_reason\n"
        "0,100,true,\n"
        "1,101,false,expert_plan_or_task_failed\n"
        "2,102,1,\n",
        encoding="utf-8",
    )

    rows = load_success_metadata(path)

    assert sorted(rows) == [0, 2]
    assert rows[0]["seed"] == "100"
    assert rows[2]["seed"] == "102"


def test_make_state_action_pairs_uses_next_qpos_as_action():
    qpos = np.arange(4 * 14, dtype=np.float32).reshape(4, 14)

    state, action = make_state_action_pairs(qpos)

    np.testing.assert_array_equal(state, qpos[:-1])
    np.testing.assert_array_equal(action, qpos[1:])


def test_select_episode_paths_shards_by_selected_order():
    paths = [Path(f"episode{i}.hdf5") for i in range(6)]

    selected = select_episode_paths(paths, num_shards=2, shard_id=1, episode_limit=2)

    assert selected == [Path("episode1.hdf5"), Path("episode3.hdf5")]
