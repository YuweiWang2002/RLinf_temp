from scripts.create_hard_eval_set import select_hard_eval_rows


def test_select_hard_eval_rows_prioritizes_all_fail_then_regression():
    summary = {
        "all_fail": {"seeds": [3, 4]},
        "qpos_success_zero_fail": {"seeds": [2, 4]},
        "candidate_hard_eval_seeds": {"seeds": [1, 2, 3, 4, 5]},
    }

    rows = select_hard_eval_rows(
        summary,
        priority=["all_fail", "qpos_success_zero_fail", "candidate_hard_eval_seeds"],
        num_seeds=4,
    )

    assert [row["seed"] for row in rows] == [3, 4, 2, 1]
    assert rows[0]["source_group"] == "all_fail"
    assert rows[2]["source_group"] == "qpos_success_zero_fail"
