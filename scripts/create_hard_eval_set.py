"""Create a fixed hard-eval seed set from hard seed summary groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_PRIORITY = (
    "all_fail",
    "qpos_success_zero_fail",
    "zero_success_bc_fail",
    "zero_k10_fail",
    "qpos_fail_zero_success",
    "qpos_fail",
    "candidate_hard_eval_seeds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", default="logs/hard_seed_mining/hard_seed_summary.json")
    parser.add_argument("--output-dir", default="logs/hard_eval_v1")
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument(
        "--priority",
        default=",".join(DEFAULT_PRIORITY),
        help="Comma-separated hard seed groups in selection priority order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    priority = [item.strip() for item in args.priority.split(",") if item.strip()]
    rows = select_hard_eval_rows(summary, priority=priority, num_seeds=args.num_seeds)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_rows_csv(output / "summary.csv", rows)
    write_json(
        output / "hard_seeds.json",
        {
            "selected": [row["seed"] for row in rows],
            "count": len(rows),
            "selection_rule": {
                "summary_json": str(args.summary_json),
                "priority": priority,
                "num_seeds": int(args.num_seeds),
            },
            "rows": rows,
        },
    )
    print(f"HARD EVAL SET: {output / 'hard_seeds.json'}")
    print(f"SUMMARY CSV: {output / 'summary.csv'}")
    print([row["seed"] for row in rows])
    return 0


def select_hard_eval_rows(
    summary: dict[str, Any],
    *,
    priority: list[str],
    num_seeds: int,
) -> list[dict[str, Any]]:
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    selected: dict[int, dict[str, Any]] = {}
    rank = 1
    for group in priority:
        seeds = group_seeds(summary, group)
        for seed in seeds:
            seed = int(seed)
            if seed in selected:
                continue
            selected[seed] = {"rank": rank, "seed": seed, "source_group": group}
            rank += 1
            if len(selected) >= num_seeds:
                return list(selected.values())
    return list(selected.values())


def group_seeds(summary: dict[str, Any], group: str) -> list[int]:
    value = summary.get(group)
    if isinstance(value, dict):
        return [int(seed) for seed in value.get("seeds", [])]
    if isinstance(value, list):
        return [int(seed) for seed in value]
    groups = summary.get("groups", {})
    if group in groups:
        return [int(seed) for seed in groups[group].get("seeds", [])]
    return []


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "seed", "source_group"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
