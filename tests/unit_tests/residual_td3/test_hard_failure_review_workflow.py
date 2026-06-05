import csv
from pathlib import Path

from scripts.build_hard_failure_review_pack import (
    build_actionable_rows,
    build_manual_master,
    build_review_candidates,
    load_manual_labels,
    load_master_manual_labels,
)
from scripts.update_manual_failure_labels import apply_updates


def test_manual_label_merge_preserves_existing_human_label():
    candidate = {
        "seed": "100100005",
        "auto_stage_guess": "no_gate_or_no_intervention",
        "auto_bucket": "pre_gate_or_no_gate",
    }
    manual = {
        100100005: {
            "manual_stage": "handover_grasp",
            "manual_bucket": "right_xyz_misaligned",
            "actionable_for_current_residual": "yes",
            "reviewer_notes": "human note",
            "labeled_at": "2026-01-01T00:00:00+00:00",
            "label_source": "manual.csv",
        }
    }

    rows = build_manual_master([candidate], manual)

    assert rows[0]["manual_status"] == "labeled"
    assert rows[0]["manual_stage"] == "handover_grasp"
    assert rows[0]["manual_bucket"] == "right_xyz_misaligned"
    assert rows[0]["reviewer_notes"] == "human note"
    assert rows[0]["auto_stage_guess"] == "no_gate_or_no_intervention"


def test_auto_fields_and_manual_fields_stay_separate_for_pending_review(tmp_path):
    zero_rows = {
        7: {
            "seed": "7",
            "success": "False",
            "failure_reason": "timeout",
            "episode_length": "600",
            "first_gate_step": "150",
            "num_interventions": "2",
            "total_ee_intervention_steps": "100",
        }
    }

    rows = build_review_candidates(
        zero_rows=zero_rows,
        bc_rows={},
        hard_eval_dir=tmp_path,
        manual_labels={},
        seed_order=[],
    )

    assert rows[0]["auto_stage_guess"] == "gate_triggered_timeout"
    assert rows[0]["manual_status"] == "pending_review"
    assert rows[0]["manual_stage"] == "unknown"
    assert rows[0]["manual_bucket"] == "unknown"
    assert rows[0]["actionable_for_current_residual"] == "unknown"


def test_pending_review_does_not_enter_actionable_set_by_default():
    master_rows = [
        {
            "seed": "1",
            "manual_status": "pending_review",
            "actionable_for_current_residual": "maybe",
        },
        {
            "seed": "2",
            "manual_status": "labeled",
            "actionable_for_current_residual": "yes",
        },
    ]

    rows = build_actionable_rows(master_rows)

    assert [row["seed"] for row in rows] == ["2"]


def test_existing_five_seed_labels_load_correctly(tmp_path):
    label_dir = tmp_path / "manual_failure_labels"
    label_dir.mkdir()
    write_csv(
        label_dir / "manual_failure_labels_k50_first5.csv",
        [
            {
                "seed": "100100005",
                "manual_stage": "handover",
                "manual_bucket": "handover_xyz_or_gripper_open_issue",
                "actionable_for_current_residual": "True",
                "note": "handover failure",
            },
            {
                "seed": "100100029",
                "manual_stage": "pre_gate_left_grasp",
                "manual_bucket": "left_arm_grasp_failure_before_gate",
                "actionable_for_current_residual": "False",
                "note": "pre gate",
            },
            {
                "seed": "100100033",
                "manual_stage": "pre_gate_left_grasp",
                "manual_bucket": "left_arm_grasp_failure_before_gate",
                "actionable_for_current_residual": "False",
                "note": "pre gate",
            },
            {
                "seed": "100100034",
                "manual_stage": "pre_gate_left_grasp",
                "manual_bucket": "left_arm_grasp_failure_before_gate",
                "actionable_for_current_residual": "False",
                "note": "pre gate",
            },
            {
                "seed": "100100035",
                "manual_stage": "pre_gate_left_grasp",
                "manual_bucket": "left_arm_grasp_failure_before_gate",
                "actionable_for_current_residual": "False",
                "note": "pre gate",
            },
        ],
    )

    labels = load_manual_labels(label_dir)

    assert labels[100100005]["actionable_for_current_residual"] == "yes"
    assert labels[100100029]["actionable_for_current_residual"] == "no"
    assert {100100005, 100100029, 100100033, 100100034, 100100035} == set(labels)


def test_existing_master_labels_are_reusable_manual_source(tmp_path):
    master = tmp_path / "manual_failure_labels_master.csv"
    write_csv(
        master,
        [
            {
                "seed": "42",
                "manual_status": "labeled",
                "manual_stage": "handover_grasp",
                "manual_bucket": "right_xyz_misaligned",
                "actionable_for_current_residual": "maybe",
                "reviewer_notes": "keep across rebuilds",
                "labeled_at": "2026-01-01T00:00:00+00:00",
                "label_source": "cli",
                "auto_stage_guess": "gate_triggered_timeout",
                "auto_bucket": "handover_review_candidate",
                "auto_manual_conflict": "false",
            },
            {
                "seed": "43",
                "manual_status": "pending_review",
                "manual_stage": "unknown",
                "manual_bucket": "unknown",
                "actionable_for_current_residual": "unknown",
                "reviewer_notes": "",
                "labeled_at": "",
                "label_source": "",
                "auto_stage_guess": "gate_triggered_timeout",
                "auto_bucket": "handover_review_candidate",
                "auto_manual_conflict": "false",
            },
        ],
    )

    labels = load_master_manual_labels(master)

    assert set(labels) == {42}
    assert labels[42]["actionable_for_current_residual"] == "maybe"
    assert labels[42]["reviewer_notes"] == "keep across rebuilds"


def test_update_manual_failure_labels_adds_one_label():
    rows, summary = apply_updates(
        [],
        [
            {
                "seed": "9",
                "manual_stage": "handover_grasp",
                "manual_bucket": "right_xyz_misaligned",
                "actionable_for_current_residual": "yes",
                "reviewer_notes": "right hand misses",
                "label_source": "unit",
            }
        ],
        overwrite=False,
        default_label_source="unit",
    )

    assert summary["added"] == 1
    assert rows[0]["manual_status"] == "labeled"
    assert rows[0]["manual_stage"] == "handover_grasp"
    assert rows[0]["reviewer_notes"] == "right hand misses"


def test_update_does_not_overwrite_existing_notes_without_overwrite():
    rows, _ = apply_updates(
        [
            {
                "seed": "9",
                "manual_status": "labeled",
                "manual_stage": "handover_grasp",
                "manual_bucket": "right_xyz_misaligned",
                "actionable_for_current_residual": "yes",
                "reviewer_notes": "keep me",
                "labeled_at": "old",
                "label_source": "old",
                "auto_stage_guess": "gate_triggered_timeout",
                "auto_bucket": "handover_review_candidate",
                "auto_manual_conflict": "false",
            }
        ],
        [
            {
                "seed": "9",
                "manual_stage": "handover_grasp",
                "manual_bucket": "right_xyz_misaligned",
                "actionable_for_current_residual": "yes",
                "reviewer_notes": "new note",
                "label_source": "unit",
            }
        ],
        overwrite=False,
        default_label_source="unit",
    )

    assert rows[0]["reviewer_notes"] == "keep me"


def test_conflict_flag_detects_auto_manual_disagreement():
    candidate = {
        "seed": "100100029",
        "auto_stage_guess": "gate_triggered_timeout",
        "auto_bucket": "handover_review_candidate",
    }
    manual = {
        100100029: {
            "manual_stage": "pre_gate_left_grasp",
            "manual_bucket": "left_arm_grasp_failure_before_gate",
            "actionable_for_current_residual": "no",
            "reviewer_notes": "pre gate",
            "labeled_at": "",
            "label_source": "manual.csv",
        }
    }

    rows = build_manual_master([candidate], manual)

    assert rows[0]["auto_manual_conflict"] == "true"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
