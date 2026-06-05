#!/usr/bin/env python3
"""Build a compact video review index for pre-grasp K50 failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", default="logs/pregrasp_k50_review/review_pack_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pack_dir = Path(args.pack_dir)
    rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []

    for summary_path in sorted(pack_dir.glob("per_condition/*/seed_*/summary.json")):
        condition = summary_path.parents[1].name
        seed = strip_prefix(summary_path.parent.name, "seed_")
        summary = json.loads(summary_path.read_text())
        episode = (summary.get("episodes") or [{}])[0]
        video_path = episode.get("video_path") or ""
        hybrid_log_path = episode.get("hybrid_log_path") or ""
        hybrid_log_csv_path = csv_log_path(Path(hybrid_log_path)) if hybrid_log_path else Path("")
        intervention_records_path = episode.get("intervention_records_path") or ""
        events = read_events(hybrid_log_csv_path, condition, seed)
        timeline_rows.extend(events)
        rows.append(
            {
                "seed": seed,
                "condition": condition,
                "success": episode.get("success"),
                "failure_reason": episode.get("failure_reason"),
                "episode_length": episode.get("episode_length"),
                "first_pregrasp_trigger_step": episode.get("first_pregrasp_trigger_step"),
                "first_handover_trigger_step": episode.get("first_handover_trigger_step"),
                "num_pregrasp_interventions": episode.get("num_pregrasp_interventions"),
                "num_handover_interventions": episode.get("num_handover_interventions"),
                "total_pregrasp_ee_steps": episode.get("total_pregrasp_ee_steps"),
                "total_handover_ee_steps": episode.get("total_handover_ee_steps"),
                "video_path": video_path,
                "summary_path": str(summary_path),
                "hybrid_log_path": str(hybrid_log_csv_path) if hybrid_log_csv_path else hybrid_log_path,
                "intervention_records_path": intervention_records_path,
                "event_summary": format_events(events),
            }
        )

    write_csv(pack_dir / "review_index.csv", rows)
    write_csv(pack_dir / "trigger_timeline.csv", timeline_rows)
    write_markdown(pack_dir / "review_index.md", rows, timeline_rows)


def read_events(hybrid_log_path: Path, condition: str, seed: str) -> list[dict[str, Any]]:
    if not hybrid_log_path.exists():
        return []
    events = []
    with hybrid_log_path.open(newline="") as f:
        for row in csv.DictReader(f):
            stage = row.get("intervention_stage") or "none"
            if stage == "none":
                continue
            end_step = as_int(row.get("env_step"))
            horizon_k = as_int(row.get("K")) or 50
            start_step = end_step - horizon_k if end_step is not None else None
            first_closing = row.get("first_closing_chunk_index") or ""
            closing_step = start_step + int(first_closing) if start_step is not None and first_closing else ""
            events.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "replan_id": row.get("replan_id"),
                    "intervention_stage": stage,
                    "trigger_source": row.get("trigger_source"),
                    "start_env_step": start_step,
                    "end_env_step": end_step,
                    "K": horizon_k,
                    "first_closing_chunk_index": first_closing,
                    "estimated_closing_env_step": closing_step,
                    "pregrasp_trigger_score": row.get("pregrasp_trigger_score"),
                    "pregrasp_trigger_reason": row.get("pregrasp_trigger_reason"),
                    "left_gripper_start": row.get("left_gripper_start"),
                    "left_gripper_end": row.get("left_gripper_end"),
                    "left_gripper_min": row.get("left_gripper_min"),
                    "left_gripper_max": row.get("left_gripper_max"),
                    "gate_prob": row.get("gate_prob"),
                    "gate_binary": row.get("gate_binary"),
                }
            )
    return events


def csv_log_path(path: Path) -> Path:
    if path.suffix == ".parquet":
        candidate = path.with_suffix(".csv")
        if candidate.exists():
            return candidate
    return path


def format_events(events: list[dict[str, Any]]) -> str:
    parts = []
    for event in events:
        parts.append(
            "{stage}@{start}-{end}, close_idx={close_idx}, score={score}, gate={gate}".format(
                stage=event["intervention_stage"],
                start=event["start_env_step"],
                end=event["end_env_step"],
                close_idx=event["first_closing_chunk_index"],
                score=event["pregrasp_trigger_score"],
                gate=event["gate_prob"],
            )
        )
    return " | ".join(parts)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], timeline_rows: list[dict[str, Any]]) -> None:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in timeline_rows:
        by_key.setdefault((str(row["seed"]), str(row["condition"])), []).append(row)

    lines = [
        "# Pre-grasp K50 Review Pack v1",
        "",
        "Compare `handover_only_k50` as the later EE switch baseline against "
        "`pregrasp_plus_handover_k50` as the early pre-grasp EE switch.",
        "",
        "Video frame mode is `executed_step`; `start_env_step` is the first step of the EE chunk, "
        "and `end_env_step` is the step after the K-step chunk finishes.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## seed {row['seed']} / {row['condition']}",
                "",
                f"- success: {row['success']}",
                f"- failure_reason: {row['failure_reason']}",
                f"- video: `{row['video_path']}`",
                f"- hybrid log: `{row['hybrid_log_path']}`",
                f"- intervention records: `{row['intervention_records_path']}`",
                f"- summary: `{row['summary_path']}`",
                "- events:",
            ]
        )
        for event in by_key.get((str(row["seed"]), str(row["condition"])), []):
            lines.append(
                "  - {stage}: env_step {start}-{end}, source={source}, "
                "first_closing_chunk_index={close_idx}, estimated_closing_env_step={close_step}, "
                "score={score}, left_gripper={g0}->{g1} min={gmin} max={gmax}, gate_prob={gate}".format(
                    stage=event["intervention_stage"],
                    start=event["start_env_step"],
                    end=event["end_env_step"],
                    source=event["trigger_source"],
                    close_idx=event["first_closing_chunk_index"],
                    close_step=event["estimated_closing_env_step"],
                    score=event["pregrasp_trigger_score"],
                    g0=event["left_gripper_start"],
                    g1=event["left_gripper_end"],
                    gmin=event["left_gripper_min"],
                    gmax=event["left_gripper_max"],
                    gate=event["gate_prob"],
                )
            )
        lines.extend(
            [
                "- review questions: Does the early pre-grasp EE switch move the left arm before a stable approach? "
                "Does it damage contact/grasp timing compared with handover-only?",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def strip_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


if __name__ == "__main__":
    main()
