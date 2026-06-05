"""Update the hard-failure manual label master CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="logs/hard_eval_v1/failure_review/manual_failure_labels_master.csv")
    parser.add_argument("--input", default=None, help="CSV containing labels to add.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--manual-stage", default=None)
    parser.add_argument("--manual-bucket", default=None)
    parser.add_argument("--actionable", default=None, choices=["yes", "no", "maybe", "unknown"])
    parser.add_argument("--notes", default=None)
    parser.add_argument("--label-source", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    master_path = Path(args.master)
    rows = read_master(master_path)
    updates = load_updates(args)
    if not updates:
        raise ValueError("Provide --input or --seed with manual label fields.")
    updated_rows, summary = apply_updates(
        rows,
        updates,
        overwrite=bool(args.overwrite),
        default_label_source=args.label_source or ("csv_import" if args.input else "cli"),
    )
    write_master(master_path, updated_rows)
    print(f"MASTER: {master_path}")
    print(
        "UPDATED SUMMARY: "
        f"input={summary['input']} added={summary['added']} changed={summary['changed']} "
        f"preserved={summary['preserved']} total={len(updated_rows)}"
    )
    return 0


def read_master(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [normalize_master_row(dict(row)) for row in csv.DictReader(f)]


def write_master(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_updates(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.input:
        with Path(args.input).open("r", newline="", encoding="utf-8") as f:
            return [normalize_update_row(dict(row)) for row in csv.DictReader(f)]
    if args.seed is None:
        return []
    return [
        normalize_update_row(
            {
                "seed": str(args.seed),
                "manual_stage": args.manual_stage or "unknown",
                "manual_bucket": args.manual_bucket or "unknown",
                "actionable_for_current_residual": args.actionable or "unknown",
                "reviewer_notes": args.notes or "",
                "label_source": args.label_source or "cli",
            }
        )
    ]


def apply_updates(
    rows: list[dict[str, str]],
    updates: list[dict[str, str]],
    *,
    overwrite: bool = False,
    default_label_source: str = "manual_update",
) -> tuple[list[dict[str, str]], dict[str, int]]:
    by_seed = {int(row["seed"]): normalize_master_row(row) for row in rows}
    summary = {"input": len(updates), "added": 0, "changed": 0, "preserved": 0}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for update in updates:
        seed = int(update["seed"])
        existing = by_seed.get(seed)
        if existing is None:
            row = new_master_row(seed)
            by_seed[seed] = row
            summary["added"] += 1
        else:
            row = existing
        changed = merge_update(
            row,
            update,
            overwrite=overwrite,
            labeled_at=now,
            default_label_source=default_label_source,
        )
        if changed:
            summary["changed"] += 1
        else:
            summary["preserved"] += 1
    return list(by_seed.values()), summary


def merge_update(
    row: dict[str, str],
    update: dict[str, str],
    *,
    overwrite: bool,
    labeled_at: str,
    default_label_source: str,
) -> bool:
    changed = False
    for field in ("manual_stage", "manual_bucket", "actionable_for_current_residual"):
        value = update.get(field, "")
        if not value or value == "unknown":
            continue
        if overwrite or row.get(field, "unknown") in {"", "unknown"}:
            changed = set_field(row, field, value) or changed
    notes = update.get("reviewer_notes", "")
    if notes and (overwrite or not row.get("reviewer_notes")):
        changed = set_field(row, "reviewer_notes", notes) or changed
    if changed or row.get("manual_status") != "labeled":
        row["manual_status"] = "labeled"
        row["labeled_at"] = update.get("labeled_at") or labeled_at
        row["label_source"] = update.get("label_source") or default_label_source
        row["auto_manual_conflict"] = bool_to_str(auto_manual_conflict(row))
        return True
    return False


def set_field(row: dict[str, str], field: str, value: str) -> bool:
    if row.get(field, "") == value:
        return False
    row[field] = value
    return True


def normalize_update_row(row: dict[str, Any]) -> dict[str, str]:
    seed = row.get("seed")
    if seed in (None, ""):
        raise ValueError("Each update row must include seed.")
    return {
        "seed": str(seed),
        "manual_stage": str(row.get("manual_stage") or "unknown"),
        "manual_bucket": str(row.get("manual_bucket") or "unknown"),
        "actionable_for_current_residual": normalize_actionable(
            first_non_empty(row, "actionable_for_current_residual", "actionable")
        ),
        "reviewer_notes": first_non_empty(row, "reviewer_notes", "notes", "note"),
        "labeled_at": str(row.get("labeled_at") or ""),
        "label_source": str(row.get("label_source") or ""),
    }


def normalize_master_row(row: dict[str, Any]) -> dict[str, str]:
    out = {field: str(row.get(field) or "") for field in MASTER_FIELDS}
    out["manual_status"] = out["manual_status"] or "pending_review"
    out["manual_stage"] = out["manual_stage"] or "unknown"
    out["manual_bucket"] = out["manual_bucket"] or "unknown"
    out["actionable_for_current_residual"] = normalize_actionable(
        out["actionable_for_current_residual"]
    )
    out["auto_stage_guess"] = out["auto_stage_guess"] or "unknown"
    out["auto_bucket"] = out["auto_bucket"] or "unknown"
    out["auto_manual_conflict"] = out["auto_manual_conflict"] or bool_to_str(auto_manual_conflict(out))
    return out


def new_master_row(seed: int) -> dict[str, str]:
    return {
        "seed": str(seed),
        "manual_status": "pending_review",
        "manual_stage": "unknown",
        "manual_bucket": "unknown",
        "actionable_for_current_residual": "unknown",
        "reviewer_notes": "",
        "labeled_at": "",
        "label_source": "",
        "auto_stage_guess": "unknown",
        "auto_bucket": "unknown",
        "auto_manual_conflict": "false",
    }


def auto_manual_conflict(row: dict[str, str]) -> bool:
    manual_stage = row.get("manual_stage", "")
    actionable = normalize_actionable(row.get("actionable_for_current_residual", "unknown"))
    auto_text = f"{row.get('auto_stage_guess', '')}|{row.get('auto_bucket', '')}"
    if manual_stage.startswith("pre_gate") and "handover" in auto_text:
        return True
    if actionable == "yes" and ("pre_gate" in auto_text or "no_gate" in auto_text):
        return True
    return actionable == "no" and row.get("auto_bucket") == "success"


def first_non_empty(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def normalize_actionable(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "yes"
    if text in {"0", "false", "no", "n"}:
        return "no"
    if text == "maybe":
        return "maybe"
    return "unknown"


def bool_to_str(value: bool) -> str:
    return "true" if value else "false"


if __name__ == "__main__":
    raise SystemExit(main())
