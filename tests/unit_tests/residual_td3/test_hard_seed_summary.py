import csv

from scripts.summarize_hard_seeds import build_groups, load_summary_rows


def write_summary(path, rows):
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "condition", "success", "rollout_status", "return_code"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_hard_seed_summary_groups_from_mining_logs(tmp_path):
    write_summary(
        tmp_path / "run" / "summary.csv",
        [
            {"seed": 1, "condition": "qpos14", "success": "False", "rollout_status": "completed", "return_code": "0"},
            {
                "seed": 1,
                "condition": "zero_k10",
                "success": "True",
                "rollout_status": "completed",
                "return_code": "0",
            },
            {"seed": 2, "condition": "qpos14", "success": "True", "rollout_status": "completed", "return_code": "0"},
            {
                "seed": 2,
                "condition": "zero_k10",
                "success": "False",
                "rollout_status": "completed",
                "return_code": "0",
            },
            {
                "seed": 3,
                "condition": "zero_k10",
                "success": "False",
                "rollout_status": "completed",
                "return_code": "0",
            },
            {"seed": 3, "condition": "bc_s05", "success": "True", "rollout_status": "completed", "return_code": "0"},
            {
                "seed": 4,
                "condition": "zero_k10",
                "success": "True",
                "rollout_status": "completed",
                "return_code": "0",
            },
            {"seed": 4, "condition": "bc_s05", "success": "False", "rollout_status": "completed", "return_code": "0"},
        ],
    )

    rows = load_summary_rows(tmp_path)
    groups = build_groups(rows)

    assert groups["qpos_fail"]["seeds"] == [1]
    assert groups["zero_k10_fail"]["seeds"] == [2, 3]
    assert groups["qpos_fail_zero_success"]["seeds"] == [1]
    assert groups["qpos_success_zero_fail"]["seeds"] == [2]
    assert groups["zero_fail_bc_success"]["seeds"] == [3]
    assert groups["zero_success_bc_fail"]["seeds"] == [4]


def test_hard_seed_summary_prefers_k50_conditions(tmp_path):
    write_summary(
        tmp_path / "run" / "summary.csv",
        [
            {"seed": 1, "condition": "qpos14", "success": "False", "rollout_status": "completed", "return_code": "0"},
            {
                "seed": 1,
                "condition": "zero_k50",
                "success": "True",
                "rollout_status": "completed",
                "return_code": "0",
            },
            {"seed": 2, "condition": "qpos14", "success": "True", "rollout_status": "completed", "return_code": "0"},
            {
                "seed": 2,
                "condition": "zero_k50",
                "success": "False",
                "rollout_status": "completed",
                "return_code": "0",
            },
            {"seed": 3, "condition": "bc_k50", "success": "False", "rollout_status": "completed", "return_code": "0"},
        ],
    )

    rows = load_summary_rows(tmp_path)
    groups = build_groups(rows)

    assert groups["zero_k50_fail"]["seeds"] == [2]
    assert groups["bc_k50_fail"]["seeds"] == [3]
    assert groups["zero_k10_fail"]["seeds"] == []
    assert groups["bc_s05_fail"]["seeds"] == []
    assert groups["qpos_fail_zero_success"]["seeds"] == [1]
    assert groups["qpos_success_zero_fail"]["seeds"] == [2]
    assert groups["candidate_hard_eval_seeds"]["seeds"] == [1, 2, 3]
