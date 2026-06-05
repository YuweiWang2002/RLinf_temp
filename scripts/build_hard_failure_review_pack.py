"""Build hard-eval failure review candidates and manual-label artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CANDIDATE_FIELDS = [
    "seed",
    "source_condition",
    "zero_success",
    "bc_success",
    "failure_reason_zero",
    "failure_reason_bc",
    "episode_len_zero",
    "episode_len_bc",
    "gate_triggered_zero",
    "gate_triggered_bc",
    "first_gate_step_zero",
    "first_gate_step_bc",
    "num_interventions_zero",
    "num_interventions_bc",
    "total_ee_steps_zero",
    "total_ee_steps_bc",
    "applied_norm_mean_zero",
    "applied_norm_mean_bc",
    "applied_norm_max_zero",
    "applied_norm_max_bc",
    "auto_stage_guess",
    "auto_bucket",
    "review_priority",
    "video_path_zero",
    "video_path_bc",
    "debug_path_zero",
    "debug_path_bc",
    "intervention_log_path_zero",
    "intervention_log_path_bc",
    "manual_status",
    "manual_stage",
    "manual_bucket",
    "actionable_for_current_residual",
    "reviewer_notes",
]

MASTER_FIELDS = [
    "seed",
    "manual_status",
    "manual_stage",
    "manual_bucket",
    "actionable_for_current_residual",
    "reviewer_notes",
    "labeled_at",
    "label_source",
    "auto_stage_guess",
    "auto_bucket",
    "auto_manual_conflict",
]

REVIEW_INDEX_FIELDS = [
    "rank",
    "seed",
    "source_condition",
    "review_priority",
    "manual_status",
    "auto_stage_guess",
    "auto_bucket",
    "why_selected",
    "video_path_zero",
    "video_path_bc",
    "debug_path_zero",
    "debug_path_bc",
    "intervention_log_path_zero",
    "intervention_log_path_bc",
]

ACTIONABLE_SUMMARY_FIELDS = [
    "rank",
    "seed",
    "manual_stage",
    "manual_bucket",
    "actionable_for_current_residual",
    "reviewer_notes",
    "label_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-eval-dir", default="logs/hard_eval_v1")
    parser.add_argument("--zero-eval-dir", default=None)
    parser.add_argument("--bc-eval-dir", default=None)
    parser.add_argument("--hard-seeds-json", default=None)
    parser.add_argument("--manual-label-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--review-pack-name", default="review_pack_v1")
    parser.add_argument("--review-limit", type=int, default=20)
    parser.add_argument("--include-pending-maybe", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hard_eval_dir = Path(args.hard_eval_dir)
    zero_eval_dir = Path(args.zero_eval_dir) if args.zero_eval_dir else hard_eval_dir / "zero_k50_eval50"
    bc_eval_dir = Path(args.bc_eval_dir) if args.bc_eval_dir else hard_eval_dir / "bc_k50_eval50"
    output_dir = Path(args.output_dir) if args.output_dir else hard_eval_dir / "failure_review"
    manual_label_dir = (
        Path(args.manual_label_dir)
        if args.manual_label_dir
        else hard_eval_dir / "manual_failure_labels"
    )

    zero_rows = load_condition_rows(resolve_summary_csv(zero_eval_dir, hard_eval_dir, "zero_k50"))
    bc_rows = load_condition_rows(resolve_summary_csv(bc_eval_dir, hard_eval_dir, "bc_k50"))
    seed_order = load_seed_order(Path(args.hard_seeds_json) if args.hard_seeds_json else hard_eval_dir / "hard_seeds.json")
    master_path = output_dir / "manual_failure_labels_master.csv"
    manual_labels = load_manual_labels(manual_label_dir)
    manual_labels.update(load_master_manual_labels(master_path))

    candidates = build_review_candidates(
        zero_rows=zero_rows,
        bc_rows=bc_rows,
        hard_eval_dir=hard_eval_dir,
        manual_labels=manual_labels,
        seed_order=seed_order,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "review_candidates.csv", CANDIDATE_FIELDS, candidates)

    master_rows = build_manual_master(candidates, manual_labels)
    write_csv(output_dir / "manual_failure_labels_master.csv", MASTER_FIELDS, master_rows)

    review_rows = select_review_rows(candidates, limit=args.review_limit)
    pack_dir = output_dir / args.review_pack_name
    pack_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pack_dir / "review_index.csv", REVIEW_INDEX_FIELDS, add_ranks(review_rows))
    write_review_markdown(pack_dir / "review_index.md", review_rows)

    actionable_dir = hard_eval_dir / "actionable_hard_eval_v1"
    actionable_rows = build_actionable_rows(
        master_rows,
        include_pending_maybe=bool(args.include_pending_maybe),
    )
    write_actionable_set(actionable_dir, actionable_rows)

    print(f"REVIEW CANDIDATES: {output_dir / 'review_candidates.csv'} count={len(candidates)}")
    print(f"REVIEW PACK: {pack_dir}")
    print(f"MANUAL MASTER: {output_dir / 'manual_failure_labels_master.csv'}")
    print(f"ACTIONABLE SET: {actionable_dir / 'hard_seeds.json'} count={len(actionable_rows)}")
    print([int(row["seed"]) for row in review_rows])
    return 0


def resolve_summary_csv(eval_dir: Path, hard_eval_dir: Path, condition: str) -> Path:
    direct = eval_dir / "summary.csv"
    if direct.exists():
        return direct
    matches = sorted(
        path
        for path in hard_eval_dir.rglob("summary.csv")
        if condition in str(path.parent) and "failure_review" not in path.parts
    )
    if not matches:
        raise FileNotFoundError(f"Could not find summary.csv for {condition} under {hard_eval_dir}")
    return matches[-1]


def load_condition_rows(summary_csv: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed_value = row.get("seed")
            if not seed_value:
                continue
            row = dict(row)
            row["_summary_csv"] = str(summary_csv)
            rows[int(seed_value)] = row
    return rows


def load_seed_order(path: Path) -> list[int]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        seeds = payload.get("selected") or payload.get("seeds") or []
    else:
        seeds = payload
    return [int(seed) for seed in seeds]


def load_manual_labels(manual_label_dir: Path) -> dict[int, dict[str, str]]:
    labels: dict[int, dict[str, str]] = {}
    if not manual_label_dir.exists():
        return labels
    for path in sorted(manual_label_dir.glob("*.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("seed"):
                    continue
                seed = int(row["seed"])
                normalized = normalize_manual_row(row)
                normalized["label_source"] = str(path)
                labels[seed] = normalized
    return labels


def load_master_manual_labels(master_path: Path) -> dict[int, dict[str, str]]:
    labels: dict[int, dict[str, str]] = {}
    if not master_path.exists():
        return labels
    with master_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("manual_status") != "labeled" or not row.get("seed"):
                continue
            seed = int(row["seed"])
            normalized = normalize_manual_row(row)
            normalized["label_source"] = str(row.get("label_source") or master_path)
            labels[seed] = normalized
    return labels


def normalize_manual_row(row: dict[str, Any]) -> dict[str, str]:
    notes = first_non_empty(row, "reviewer_notes", "notes", "note")
    return {
        "manual_stage": str(row.get("manual_stage") or "unknown"),
        "manual_bucket": str(row.get("manual_bucket") or "unknown"),
        "actionable_for_current_residual": normalize_actionable(
            first_non_empty(row, "actionable_for_current_residual", "actionable")
        ),
        "reviewer_notes": str(notes or ""),
        "labeled_at": str(row.get("labeled_at") or ""),
    }


def build_review_candidates(
    *,
    zero_rows: dict[int, dict[str, Any]],
    bc_rows: dict[int, dict[str, Any]],
    hard_eval_dir: Path,
    manual_labels: dict[int, dict[str, str]],
    seed_order: list[int],
) -> list[dict[str, Any]]:
    seeds = sorted(set(zero_rows) | set(bc_rows))
    seed_rank = {seed: index for index, seed in enumerate(seed_order)}
    candidates = [
        build_candidate(seed, zero_rows.get(seed), bc_rows.get(seed), hard_eval_dir, manual_labels)
        for seed in seeds
    ]
    candidates.sort(
        key=lambda row: (
            -int(row["review_priority"]),
            seed_rank.get(int(row["seed"]), len(seed_rank) + int(row["seed"])),
            int(row["seed"]),
        )
    )
    return candidates


def build_candidate(
    seed: int,
    zero_row: dict[str, Any] | None,
    bc_row: dict[str, Any] | None,
    hard_eval_dir: Path,
    manual_labels: dict[int, dict[str, str]],
) -> dict[str, Any]:
    zero_success = parse_bool(value(zero_row, "success"))
    bc_success = parse_bool(value(bc_row, "success"))
    auto_stage_guess, auto_bucket = auto_classify(zero_row, bc_row)
    videos = {
        "zero": resolve_video_path(zero_row, hard_eval_dir, seed, "zero"),
        "bc": resolve_video_path(bc_row, hard_eval_dir, seed, "bc"),
    }
    manual = manual_labels.get(seed)
    row = {
        "seed": seed,
        "source_condition": source_condition(zero_row, bc_row, zero_success, bc_success),
        "zero_success": bool_to_str(zero_success) if zero_row else "",
        "bc_success": bool_to_str(bc_success) if bc_row else "",
        "failure_reason_zero": value(zero_row, "failure_reason"),
        "failure_reason_bc": value(bc_row, "failure_reason"),
        "episode_len_zero": value(zero_row, "episode_length"),
        "episode_len_bc": value(bc_row, "episode_length"),
        "gate_triggered_zero": bool_to_str(gate_triggered(zero_row)) if zero_row else "",
        "gate_triggered_bc": bool_to_str(gate_triggered(bc_row)) if bc_row else "",
        "first_gate_step_zero": value(zero_row, "first_gate_step"),
        "first_gate_step_bc": value(bc_row, "first_gate_step"),
        "num_interventions_zero": value(zero_row, "num_interventions"),
        "num_interventions_bc": value(bc_row, "num_interventions"),
        "total_ee_steps_zero": value(zero_row, "total_ee_intervention_steps"),
        "total_ee_steps_bc": value(bc_row, "total_ee_intervention_steps"),
        "applied_norm_mean_zero": value(zero_row, "applied_norm_mean"),
        "applied_norm_mean_bc": value(bc_row, "applied_norm_mean"),
        "applied_norm_max_zero": value(zero_row, "applied_norm_max"),
        "applied_norm_max_bc": value(bc_row, "applied_norm_max"),
        "auto_stage_guess": auto_stage_guess,
        "auto_bucket": auto_bucket,
        "review_priority": 0,
        "video_path_zero": videos["zero"],
        "video_path_bc": videos["bc"],
        "debug_path_zero": resolve_debug_path(zero_row),
        "debug_path_bc": resolve_debug_path(bc_row),
        "intervention_log_path_zero": value(zero_row, "intervention_records_path"),
        "intervention_log_path_bc": value(bc_row, "intervention_records_path"),
        "manual_status": "labeled" if manual else "pending_review",
        "manual_stage": manual["manual_stage"] if manual else "unknown",
        "manual_bucket": manual["manual_bucket"] if manual else "unknown",
        "actionable_for_current_residual": manual["actionable_for_current_residual"] if manual else "unknown",
        "reviewer_notes": manual["reviewer_notes"] if manual else "",
    }
    row["review_priority"] = review_priority(row, zero_row, bc_row)
    if auto_bucket != "infra_fail" and not videos["zero"] and not videos["bc"]:
        row["auto_bucket"] = f"{auto_bucket}|missing_artifact"
        row["review_priority"] = max(0, int(row["review_priority"]) - 25)
    return row


def source_condition(
    zero_row: dict[str, Any] | None,
    bc_row: dict[str, Any] | None,
    zero_success: bool,
    bc_success: bool,
) -> str:
    zero_fail = zero_row is not None and not zero_success
    bc_fail = bc_row is not None and not bc_success
    if zero_fail and bc_fail:
        return "both"
    if zero_fail:
        return "zero_k50"
    if bc_fail:
        return "bc_k50"
    return "none"


def auto_classify(
    zero_row: dict[str, Any] | None,
    bc_row: dict[str, Any] | None,
) -> tuple[str, str]:
    rows = [row for row in (zero_row, bc_row) if row]
    failing_rows = [row for row in rows if not parse_bool(value(row, "success"))]
    if not failing_rows:
        return "success", "success"
    if any(is_infra_fail(row) for row in failing_rows):
        return "infra_or_safety_failure", "infra_fail"
    if any(gate_triggered(row) for row in failing_rows):
        if any(str(value(row, "failure_reason")) == "timeout" for row in failing_rows):
            return "gate_triggered_timeout", "handover_review_candidate"
        return "gate_triggered_failure", "handover_review_candidate"
    return "no_gate_or_no_intervention", "pre_gate_or_no_gate"


def review_priority(
    candidate: dict[str, Any],
    zero_row: dict[str, Any] | None,
    bc_row: dict[str, Any] | None,
) -> int:
    score = 0
    if candidate["source_condition"] == "both":
        score += 100
    elif candidate["source_condition"] in {"zero_k50", "bc_k50"}:
        score += 45
    if gate_triggered(zero_row) or gate_triggered(bc_row):
        score += 35
    if number(value(zero_row, "num_interventions")) > 0 or number(value(bc_row, "num_interventions")) > 0:
        score += 20
    if number(value(zero_row, "total_ee_intervention_steps")) > 0 or number(value(bc_row, "total_ee_intervention_steps")) > 0:
        score += 20
    if value(zero_row, "failure_reason") == "timeout" or value(bc_row, "failure_reason") == "timeout":
        score += 15
    if number(value(zero_row, "episode_length")) >= 590 or number(value(bc_row, "episode_length")) >= 590:
        score += 10
    if candidate["video_path_zero"] or candidate["video_path_bc"]:
        score += 30
    if candidate["manual_status"] == "labeled":
        score += 5
    score += min(10, int(1000 * max(number(candidate["applied_norm_max_zero"]), number(candidate["applied_norm_max_bc"]))))
    return score


def build_manual_master(
    candidates: list[dict[str, Any]],
    manual_labels: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    seen: set[int] = set()
    for candidate in candidates:
        seed = int(candidate["seed"])
        seen.add(seed)
        manual = manual_labels.get(seed)
        rows.append(master_row(seed, candidate, manual))
    for seed, manual in sorted(manual_labels.items()):
        if seed in seen:
            continue
        rows.append(master_row(seed, None, manual))
    rows.sort(key=lambda row: int(row["seed"]))
    return rows


def master_row(
    seed: int,
    candidate: dict[str, Any] | None,
    manual: dict[str, str] | None,
) -> dict[str, Any]:
    auto_stage_guess = str(candidate.get("auto_stage_guess", "unknown")) if candidate else "unknown"
    auto_bucket = str(candidate.get("auto_bucket", "unknown")) if candidate else "unknown"
    manual_status = "labeled" if manual else "pending_review"
    row = {
        "seed": seed,
        "manual_status": manual_status,
        "manual_stage": manual["manual_stage"] if manual else "unknown",
        "manual_bucket": manual["manual_bucket"] if manual else "unknown",
        "actionable_for_current_residual": manual["actionable_for_current_residual"] if manual else "unknown",
        "reviewer_notes": manual["reviewer_notes"] if manual else "",
        "labeled_at": manual.get("labeled_at", "") if manual else "",
        "label_source": manual.get("label_source", "") if manual else "",
        "auto_stage_guess": auto_stage_guess,
        "auto_bucket": auto_bucket,
        "auto_manual_conflict": bool_to_str(auto_manual_conflict(auto_stage_guess, auto_bucket, manual)),
    }
    return row


def auto_manual_conflict(
    auto_stage_guess: str,
    auto_bucket: str,
    manual: dict[str, str] | None,
) -> bool:
    if not manual:
        return False
    manual_stage = manual.get("manual_stage", "")
    actionable = manual.get("actionable_for_current_residual", "unknown")
    auto_text = f"{auto_stage_guess}|{auto_bucket}"
    if manual_stage.startswith("pre_gate") and "handover" in auto_text:
        return True
    if actionable == "yes" and ("pre_gate" in auto_text or "no_gate" in auto_text):
        return True
    return actionable == "no" and auto_bucket == "success"


def select_review_rows(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    with_artifacts = [row for row in candidates if row["video_path_zero"] or row["video_path_bc"]]
    missing = [row for row in candidates if not row["video_path_zero"] and not row["video_path_bc"]]
    selected = with_artifacts[:limit]
    if len(selected) < limit:
        selected.extend(missing[: limit - len(selected)])
    return selected


def add_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for rank, row in enumerate(rows, start=1):
        out = dict(row)
        out["rank"] = rank
        out["why_selected"] = why_selected(row)
        ranked.append(out)
    return ranked


def why_selected(row: dict[str, Any]) -> str:
    reasons = []
    if row["source_condition"] == "both":
        reasons.append("zero_k50 and bc_k50 both failed")
    if row["gate_triggered_zero"] == "true" or row["gate_triggered_bc"] == "true":
        reasons.append("gate/intervention was triggered")
    if row["failure_reason_zero"] == "timeout" or row["failure_reason_bc"] == "timeout":
        reasons.append("timeout near max episode length")
    if row["video_path_zero"] or row["video_path_bc"]:
        reasons.append("video artifact available")
    else:
        reasons.append("missing video artifact; included after available-video seeds")
    return "; ".join(reasons)


def write_review_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Hard Eval Failure Review Pack v1",
        "",
        "Manual labels must come from human review only. Auto guesses below are triage hints.",
        "",
        "Question template for each seed:",
        "",
        "- Failure stage: pre_gate_left_grasp / handover_approach / handover_grasp / post_handover_place / unknown",
        "- Primary cause: left_grasp_failed / right_xyz_misaligned / gripper_open_close_timing / wrist_orientation / object_knocked / timeout_after_handover_attempt / other / unknown",
        "- Can current right_xyz residual fix it: yes / no / maybe",
        "- Notes: free text",
        "",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {rank}. Seed {row['seed']}",
                "",
                f"- Why selected: {why_selected(row)}",
                f"- Auto guess: {row['auto_stage_guess']} / {row['auto_bucket']}",
                f"- Manual status: {row['manual_status']}",
                f"- zero video: {row['video_path_zero'] or 'missing_artifact'}",
                f"- bc video: {row['video_path_bc'] or 'missing_artifact'}",
                f"- zero debug: {row['debug_path_zero'] or 'missing_artifact'}",
                f"- bc debug: {row['debug_path_bc'] or 'missing_artifact'}",
                f"- zero intervention log: {row['intervention_log_path_zero'] or 'missing_artifact'}",
                f"- bc intervention log: {row['intervention_log_path_bc'] or 'missing_artifact'}",
                "",
                "Please answer:",
                "",
                "- Failure stage:",
                "- Primary cause:",
                "- Can current right_xyz residual fix it:",
                "- Notes:",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_actionable_rows(
    master_rows: list[dict[str, Any]],
    *,
    include_pending_maybe: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for row in master_rows:
        actionable = normalize_actionable(row.get("actionable_for_current_residual", "unknown"))
        if row.get("manual_status") == "labeled" and actionable in {"yes", "maybe"}:
            rows.append(dict(row))
        elif include_pending_maybe and row.get("manual_status") == "pending_review":
            rows.append(dict(row))
    rows.sort(key=lambda row: int(row["seed"]))
    return rows


def write_actionable_set(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for rank, row in enumerate(rows, start=1):
        summary = {field: row.get(field, "") for field in ACTIONABLE_SUMMARY_FIELDS}
        summary["rank"] = rank
        summary_rows.append(summary)
    write_csv(output_dir / "summary.csv", ACTIONABLE_SUMMARY_FIELDS, summary_rows)
    payload = {
        "selected": [int(row["seed"]) for row in rows],
        "count": len(rows),
        "selection_rule": {
            "source": "manual_failure_labels_master.csv",
            "include": "manual actionable_for_current_residual in {yes, maybe}",
            "pending_review_included": False,
        },
        "rows": summary_rows,
    }
    (output_dir / "hard_seeds.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_video_path(
    row: dict[str, Any] | None,
    hard_eval_dir: Path,
    seed: int,
    condition_hint: str,
) -> str:
    explicit = value(row, "video_path")
    if explicit and Path(explicit).exists():
        return explicit
    organized = hard_eval_dir / "failure_review" / "review_videos_v1" / f"seed_{seed}" / f"{condition_hint}_k50.mp4"
    if organized.exists():
        return str(organized)
    matches = sorted(hard_eval_dir.rglob(f"*seed_{seed}*.mp4"))
    if not matches:
        return ""
    condition_matches = [path for path in matches if condition_hint in str(path)]
    if not condition_matches:
        return ""
    return str(condition_matches[0])


def resolve_debug_path(row: dict[str, Any] | None) -> str:
    explicit = value(row, "hybrid_log_path")
    if not explicit:
        return ""
    csv_path = Path(explicit).with_suffix(".csv")
    if csv_path.exists():
        return str(csv_path)
    return explicit


def is_infra_fail(row: dict[str, Any]) -> bool:
    return (
        parse_bool(value(row, "simulator_crash"))
        or number(value(row, "nan_inf_count")) > 0
        or number(value(row, "saturation_ratio")) > 0
    )


def gate_triggered(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    return (
        value(row, "first_gate_step") not in {"", "None", "null"}
        or number(value(row, "num_interventions")) > 0
        or number(value(row, "total_ee_intervention_steps")) > 0
    )


def value(row: dict[str, Any] | None, key: str) -> str:
    if not row:
        return ""
    raw = row.get(key, "")
    return "" if raw is None else str(raw)


def first_non_empty(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def parse_bool(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes"}


def bool_to_str(value: bool) -> str:
    return "true" if value else "false"


def number(raw: Any) -> float:
    if raw in (None, ""):
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def normalize_actionable(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "yes"
    if text in {"0", "false", "no", "n"}:
        return "no"
    if text == "maybe":
        return "maybe"
    return "unknown"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
