import json

from scripts.collect_hard_seed_expert_demos import select_hard_seed_rows


def test_select_hard_seed_rows_prefers_all_fail_rows(tmp_path):
    path = tmp_path / "hard_seeds.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"rank": 1, "seed": 10, "source_group": "all_fail"},
                    {"rank": 2, "seed": 20, "source_group": "candidate_hard_eval_seeds"},
                    {"rank": 3, "seed": 30, "source_group": "all_fail"},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = select_hard_seed_rows(path, ["all_fail", "candidate_hard_eval_seeds"], limit=None)

    assert [row["seed"] for row in rows] == [10, 30, 20]


def test_select_hard_seed_rows_supports_summary_groups(tmp_path):
    path = tmp_path / "hard_seed_summary.json"
    path.write_text(
        json.dumps(
            {
                "all_fail": [10, 20],
                "candidate_hard_eval_seeds": [20, 30],
            }
        ),
        encoding="utf-8",
    )

    rows = select_hard_seed_rows(path, ["all_fail", "candidate_hard_eval_seeds"], limit=2)

    assert [row["seed"] for row in rows] == [10, 20]
    assert [row["source_group"] for row in rows] == ["all_fail", "all_fail"]
