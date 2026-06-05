"""Build a compact failure review table for K50 eval runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

FIELDS = [
    "seed",
    "condition",
    "success",
    "failure_reason",
    "episode_length",
    "first_gate_step",
    "num_interventions",
    "total_ee_intervention_steps",
    "gate_prob_mean",
    "gate_prob_max",
    "applied_norm_mean",
    "applied_norm_max",
    "nan_inf_count",
    "saturation_ratio",
    "auto_bucket",
    "human_review_focus",
    "video_path",
    "run_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    rows = load_episode_rows(eval_dir)
    out = Path(args.output_csv) if args.output_csv else eval_dir / "failure_review.csv"
    write_csv(out, rows)
    print(f"FAILURE REVIEW CSV: {out}")
    return 0


def load_episode_rows(eval_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(eval_dir.rglob("summary.json")):
        data = json.loads(summary_path.read_text())
        condition = infer_condition(summary_path, eval_dir)
        for episode in data.get("episodes") or []:
            row = dict(episode)
            row["condition"] = condition
            row["run_dir"] = str(summary_path.parent)
            bucket, focus = classify(row)
            row["auto_bucket"] = bucket
            row["human_review_focus"] = focus
            rows.append(row)
    return sorted(rows, key=lambda row: (str(row.get("condition", "")), int(row.get("seed", 0))))


def infer_condition(summary_path: Path, eval_dir: Path) -> str:
    try:
        rel = summary_path.relative_to(eval_dir)
    except ValueError:
        return ""
    parts = rel.parts
    if len(parts) >= 4 and parts[0] == "per_condition":
        return parts[1]
    return eval_dir.name


def classify(row: dict[str, Any]) -> tuple[str, str]:
    if bool(row.get("success")):
        return "success", ""
    if as_float(row, "nan_inf_count") > 0 or as_float(row, "saturation_ratio") > 0:
        return "safety_anomaly", "check NaN/Inf or action saturation before visual review"
    reason = str(row.get("failure_reason") or "")
    total_ee_steps = as_float(row, "total_ee_intervention_steps")
    num_interventions = as_float(row, "num_interventions")
    applied_max = as_float(row, "applied_norm_max")
    if reason == "timeout" and total_ee_steps == 0:
        return "timeout_no_intervention", "check whether gate never activates or activates too late"
    if reason == "timeout" and num_interventions <= 2 and total_ee_steps <= 100:
        return (
            "timeout_after_short_intervention",
            "watch handover: decide xyz offset, gripper timing, pose mismatch, or object already displaced",
        )
    if reason == "timeout" and applied_max < 0.003:
        return (
            "timeout_small_residual",
            "watch whether residual is too weak versus base qpos trajectory",
        )
    if reason == "timeout":
        return (
            "timeout_progress_issue",
            "watch whether robot reaches handover, misses grasp, or drops object after grasp",
        )
    return "non_timeout_failure", "inspect video and debug logs"


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
