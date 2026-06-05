"""Summarize hard seed mining runs into one compact report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

GROUP_FIELDS = (
    "qpos_fail",
    "zero_k50_fail",
    "bc_k50_fail",
    "zero_k10_fail",
    "bc_s05_fail",
    "qpos_fail_zero_success",
    "qpos_success_zero_fail",
    "zero_fail_bc_success",
    "zero_success_bc_fail",
    "all_fail",
    "candidate_hard_eval_seeds",
    "infra_fail",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="logs/hard_seed_mining")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--include-infra-failures", action="store_true")
    parser.add_argument(
        "--exclude-run-prefix",
        action="append",
        default=["codex_"],
        help="Skip run directories whose name starts with this prefix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = load_summary_rows(input_dir, exclude_prefixes=tuple(args.exclude_run_prefix))
    groups = build_groups(rows, include_infra_failures=bool(args.include_infra_failures))
    output_json = Path(args.output_json) if args.output_json else input_dir / "hard_seed_summary.json"
    output_csv = Path(args.output_csv) if args.output_csv else input_dir / "hard_seed_summary.csv"
    write_json(output_json, groups)
    write_csv(output_csv, groups)
    print(f"HARD SEED SUMMARY JSON: {output_json}")
    print(f"HARD SEED SUMMARY CSV: {output_csv}")
    for name in GROUP_FIELDS:
        values = groups.get(name, {}).get("seeds", [])
        print(f"{name}: count={len(values)} seeds={values}")
    return 0


def load_summary_rows(input_dir: Path, *, exclude_prefixes: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("summary.csv")):
        if path.name in {"hard_seed_summary.csv"}:
            continue
        if should_skip(path, input_dir, exclude_prefixes):
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("seed") or not row.get("condition"):
                    continue
                row = dict(row)
                row["summary_path"] = str(path)
                rows.append(row)
    return rows


def should_skip(path: Path, input_dir: Path, prefixes: tuple[str, ...]) -> bool:
    try:
        relative = path.relative_to(input_dir)
    except ValueError:
        return False
    if not relative.parts:
        return False
    first = relative.parts[0]
    return any(first.startswith(prefix) for prefix in prefixes)


def build_groups(
    rows: list[dict[str, Any]],
    *,
    include_infra_failures: bool = False,
) -> dict[str, Any]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        seed = int(row["seed"])
        condition = normalize_condition(row["condition"])
        if condition not in {"qpos14", "zero_k50", "bc_k50", "zero_k10", "bc_s05"}:
            continue
        current = by_seed.setdefault(seed, {}).get(condition)
        if current is None or row_rank(row) >= row_rank(current):
            by_seed[seed][condition] = row

    def has(seed: int, condition: str) -> bool:
        row = by_seed.get(seed, {}).get(condition)
        return row is not None and (include_infra_failures or is_evaluated(row))

    def success(seed: int, condition: str) -> bool:
        return has(seed, condition) and parse_bool(by_seed[seed][condition].get("success"))

    def fail(seed: int, condition: str) -> bool:
        return has(seed, condition) and not success(seed, condition)

    zero_condition = preferred_condition(by_seed, "zero_k50", "zero_k10")
    bc_condition = preferred_condition(by_seed, "bc_k50", "bc_s05")
    qpos_fail = sorted(seed for seed in by_seed if fail(seed, "qpos14"))
    zero_fail = sorted(seed for seed in by_seed if fail(seed, zero_condition))
    bc_fail = sorted(seed for seed in by_seed if fail(seed, bc_condition))
    qpos_fail_zero_success = sorted(
        seed for seed in by_seed if fail(seed, "qpos14") and success(seed, zero_condition)
    )
    qpos_success_zero_fail = sorted(
        seed for seed in by_seed if success(seed, "qpos14") and fail(seed, zero_condition)
    )
    zero_fail_bc_success = sorted(
        seed for seed in by_seed if fail(seed, zero_condition) and success(seed, bc_condition)
    )
    zero_success_bc_fail = sorted(
        seed for seed in by_seed if success(seed, zero_condition) and fail(seed, bc_condition)
    )
    all_fail = sorted(
        seed
        for seed in by_seed
        if any(has(seed, condition) for condition in ("qpos14", zero_condition, bc_condition))
        and all(
            fail(seed, condition)
            for condition in ("qpos14", zero_condition, bc_condition)
            if has(seed, condition)
        )
    )
    infra_fail = sorted(
        seed
        for seed, per_condition in by_seed.items()
        if any(not is_evaluated(row) for row in per_condition.values())
    )
    candidate_hard_eval_seeds = sorted(
        set(qpos_fail)
        | set(zero_fail)
        | set(bc_fail)
        | set(qpos_success_zero_fail)
        | set(zero_success_bc_fail)
    )
    payload = {
        "input_rows": len(rows),
        "num_unique_seeds": len(by_seed),
        "conditions": {
            condition: summarize_condition(by_seed, condition)
            for condition in ("qpos14", "zero_k50", "bc_k50", "zero_k10", "bc_s05")
        },
        "groups": {},
    }
    group_values = {name: [] for name in GROUP_FIELDS}
    group_values.update(
        {
        "qpos_fail": qpos_fail,
        f"{zero_condition}_fail": zero_fail,
        f"{bc_condition}_fail": bc_fail,
        "qpos_fail_zero_success": qpos_fail_zero_success,
        "qpos_success_zero_fail": qpos_success_zero_fail,
        "zero_fail_bc_success": zero_fail_bc_success,
        "zero_success_bc_fail": zero_success_bc_fail,
        "all_fail": all_fail,
        "candidate_hard_eval_seeds": candidate_hard_eval_seeds,
        "infra_fail": infra_fail,
        }
    )
    for name in GROUP_FIELDS:
        seeds = group_values[name]
        payload["groups"][name] = {"count": len(seeds), "seeds": seeds}
        payload[name] = payload["groups"][name]
    return payload


def normalize_condition(condition: str) -> str:
    aliases = {
        "qpos14_baseline": "qpos14",
        "zero": "zero_k50",
        "pointwise_zero_k10": "zero_k10",
        "pointwise_zero_k50": "zero_k50",
        "bc": "bc_k50",
        "bc_control_s05": "bc_s05",
        "bc_hard_k50": "bc_k50",
    }
    return aliases.get(condition, condition)


def preferred_condition(
    by_seed: dict[int, dict[str, dict[str, Any]]],
    preferred: str,
    fallback: str,
) -> str:
    """Prefer K50 condition names while keeping legacy K10 summaries readable."""

    if any(preferred in per_condition for per_condition in by_seed.values()):
        return preferred
    return fallback


def row_rank(row: dict[str, Any]) -> tuple[int, int]:
    return (int(is_evaluated(row)), int(row.get("return_code") in ("0", 0)))


def is_evaluated(row: dict[str, Any]) -> bool:
    return row.get("rollout_status") == "completed"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def summarize_condition(
    by_seed: dict[int, dict[str, dict[str, Any]]],
    condition: str,
) -> dict[str, Any]:
    rows = [per_condition[condition] for per_condition in by_seed.values() if condition in per_condition]
    evaluated = [row for row in rows if is_evaluated(row)]
    successes = [row for row in evaluated if parse_bool(row.get("success"))]
    return {
        "rows": len(rows),
        "evaluated": len(evaluated),
        "success_count": len(successes),
        "failure_count": len(evaluated) - len(successes),
        "success_rate": len(successes) / len(evaluated) if evaluated else 0.0,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "count", "seeds"])
        writer.writeheader()
        for name in GROUP_FIELDS:
            row = payload["groups"][name]
            writer.writerow(
                {
                    "group": name,
                    "count": row["count"],
                    "seeds": ";".join(str(seed) for seed in row["seeds"]),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
