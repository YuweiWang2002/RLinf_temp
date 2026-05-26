"""Build a compact CSV summary for gate-controlled hybrid rollout sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=SUMMARY_OR_DIR",
        help="Condition label and either a summary.json path or a rollout save directory.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [summarize_condition(spec) for spec in args.condition]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "summary_path",
        "save_dir",
        "rollout_status",
        "execution_mode",
        "ee16_execution_strategy",
        "residual_actor",
        "residual_horizon_k",
        "gate_threshold",
        "num_episodes",
        "success_count",
        "success_rate",
        "mean_return",
        "mean_episode_length",
        "mean_replan_steps",
        "mean_wall_time_s",
        "first_gate_steps",
        "first_intervention_steps",
        "num_interventions_total",
        "total_ee_intervention_steps",
        "failure_reasons",
        "simulator_crash_count",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"SUMMARY CSV: {output}")
    return 0


def summarize_condition(spec: str) -> dict[str, Any]:
    name, path = parse_condition(spec)
    summary_path = resolve_summary_path(path)
    if not summary_path.exists():
        return {
            "condition": name,
            "summary_path": str(summary_path),
            "save_dir": str(path),
            "rollout_status": "missing_summary",
            "execution_mode": "",
            "ee16_execution_strategy": "",
            "residual_actor": "",
            "residual_horizon_k": "",
            "gate_threshold": "",
            "num_episodes": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "mean_return": 0.0,
            "mean_episode_length": 0.0,
            "mean_replan_steps": 0.0,
            "mean_wall_time_s": 0.0,
            "first_gate_steps": "",
            "first_intervention_steps": "",
            "num_interventions_total": 0,
            "total_ee_intervention_steps": 0,
            "failure_reasons": "missing_summary",
            "simulator_crash_count": 1,
        }
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    episodes = list(summary.get("episodes", []))
    successes = [bool(ep.get("success", False)) for ep in episodes]
    inferred = [infer_episode_from_log(ep) for ep in episodes]
    first_gate_steps = [
        coalesce(ep.get("first_gate_step"), ep.get("gate_first_activation_env_step"), info["first_gate_step"])
        for ep, info in zip(episodes, inferred, strict=True)
    ]
    first_intervention_steps = [
        coalesce(ep.get("first_intervention_step"), info["first_intervention_step"])
        for ep, info in zip(episodes, inferred, strict=True)
    ]
    num_interventions = [
        coalesce(ep.get("num_interventions"), info["num_interventions"], default=0)
        for ep, info in zip(episodes, inferred, strict=True)
    ]
    total_ee_steps = [
        coalesce(ep.get("total_ee_intervention_steps"), info["total_ee_intervention_steps"], default=0)
        for ep, info in zip(episodes, inferred, strict=True)
    ]
    return {
        "condition": name,
        "summary_path": str(summary_path),
        "save_dir": summary.get("save_dir", str(summary_path.parent)),
        "rollout_status": summary.get("rollout_status", "completed"),
        "execution_mode": summary.get("execution_mode", ""),
        "ee16_execution_strategy": summary.get("ee16_execution_strategy", ""),
        "residual_actor": summary.get("residual_actor", ""),
        "residual_horizon_k": summary.get("residual_horizon_k", ""),
        "gate_threshold": summary.get("gate_threshold", ""),
        "num_episodes": len(episodes),
        "success_count": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_return": mean_field(episodes, "return"),
        "mean_episode_length": mean_field(episodes, "episode_length"),
        "mean_replan_steps": mean_field(episodes, "replan_steps"),
        "mean_wall_time_s": mean_field(episodes, "wall_time_s"),
        "first_gate_steps": join_values(first_gate_steps),
        "first_intervention_steps": join_values(first_intervention_steps),
        "num_interventions_total": int(sum(int(value) for value in num_interventions)),
        "total_ee_intervention_steps": int(sum(int(value) for value in total_ee_steps)),
        "failure_reasons": join_values(ep.get("failure_reason") for ep in episodes if ep.get("failure_reason")),
        "simulator_crash_count": int(sum(int(bool(ep.get("simulator_crash", False))) for ep in episodes)),
    }


def parse_condition(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--condition must be NAME=SUMMARY_OR_DIR, got {spec!r}.")
    name, raw_path = spec.split("=", 1)
    if not name:
        raise ValueError("--condition name cannot be empty.")
    return name, Path(raw_path)


def resolve_summary_path(path: Path) -> Path:
    if path.name == "summary.json":
        return path
    return path / "summary.json"


def mean_field(episodes: list[dict[str, Any]], field: str) -> float:
    values = [float(ep[field]) for ep in episodes if ep.get(field) is not None]
    return float(np.mean(values)) if values else 0.0


def infer_episode_from_log(episode: dict[str, Any]) -> dict[str, Any]:
    path = episode.get("hybrid_log_path")
    if not path:
        return empty_episode_inference()
    csv_path = Path(path).with_suffix(".csv")
    if not csv_path.exists():
        return empty_episode_inference()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gate_rows = [row for row in rows if parse_boolish(row.get("gate_binary"))]
    intervention_rows = [row for row in rows if row.get("execution_mode") == "ee16_zero_residual"]
    return {
        "first_gate_step": first_row_step(gate_rows),
        "first_intervention_step": first_row_step(intervention_rows),
        "num_interventions": len(intervention_rows),
        "total_ee_intervention_steps": sum(infer_selected_index_count(row) for row in intervention_rows),
    }


def empty_episode_inference() -> dict[str, Any]:
    return {
        "first_gate_step": None,
        "first_intervention_step": None,
        "num_interventions": 0,
        "total_ee_intervention_steps": 0,
    }


def first_row_step(rows: list[dict[str, str]]) -> int | None:
    if not rows:
        return None
    row = rows[0]
    value = row.get("env_step_start") or row.get("env_step")
    return int(float(value)) if value not in (None, "") else None


def infer_selected_index_count(row: dict[str, str]) -> int:
    if row.get("num_intervention_steps"):
        return int(float(row["num_intervention_steps"]))
    selected = row.get("selected_indices")
    if not selected or selected in ("None", "[]"):
        return 0
    return selected.count(",") + 1


def parse_boolish(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def coalesce(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def join_values(values) -> str:
    return ";".join("none" if value is None else str(value) for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
