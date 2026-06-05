"""Convert saved residual intervention rollout logs into TD3 replay artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rlinf.algorithms.residual_td3.residual_replay import (
    ReplayRewardConfig,
    build_replay_from_rollout,
    load_baseline_success_by_seed,
    load_intervention_records,
    load_rollout_summary,
    save_replay_artifacts,
)

REQUIRED_RECORD_FIELDS = (
    "obs_vector",
    "applied_delta_local_xyz",
    "base_ee16",
    "exec_ee16",
    "applied_delta_world_xyz",
    "gate_score",
    "residual_scale",
)
OPTIONAL_METADATA_FIELDS = (
    "env_step",
    "intervention_id",
    "intervention_step_i",
    "pred_delta_local_xyz",
    "noise_std",
    "saturation",
    "has_nan_or_inf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-dir",
        action="append",
        required=True,
        help="Rollout directory containing summary.json and intervention_records parquet files. Repeatable.",
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--action-source",
        default="auto",
        help="Replay action_source value, or auto from rollout directory name.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("sparse_success", "baseline_comparison"),
        default="sparse_success",
    )
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--reward-discount-to-steps", choices=("all", "last"), default="all")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.save_dir)
    failures: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []
    baseline_success = (
        load_baseline_success_by_seed(args.baseline_summary)
        if args.reward_mode == "baseline_comparison"
        else None
    )

    for raw_rollout_dir in args.rollout_dir:
        rollout_dir = Path(raw_rollout_dir)
        action_source = resolve_action_source(args.action_source, rollout_dir)
        try:
            summary, records_by_episode, inspection = inspect_rollout_dir(rollout_dir)
        except (FileNotFoundError, ValueError) as exc:
            failures.append({"rollout_dir": str(rollout_dir), "error": str(exc)})
            continue
        if inspection["missing_fields"]:
            failures.append(
                {
                    "rollout_dir": str(rollout_dir),
                    "error": "missing required replay fields",
                    "missing_fields": inspection["missing_fields"],
                    "record_files": inspection["record_files"],
                }
            )
            continue
        if inspection["num_records"] == 0 and not args.allow_empty:
            failures.append(
                {
                    "rollout_dir": str(rollout_dir),
                    "error": "no residual intervention records",
                    "missing_fields": [],
                    "record_files": inspection["record_files"],
                }
            )
            continue
        if args.dry_run:
            converted.append(
                {
                    "rollout_dir": str(rollout_dir),
                    "action_source": action_source,
                    "num_records": inspection["num_records"],
                    "dry_run": True,
                }
            )
            continue

        replay, episode_rows, reward_summary = build_replay_from_rollout(
            summary,
            records_by_episode,
            action_source=action_source,
            reward_config=ReplayRewardConfig(
                mode=args.reward_mode,
                discount_to_steps=args.reward_discount_to_steps,
                baseline_summary_path=args.baseline_summary,
            ),
            baseline_success_by_seed=baseline_success,
        )
        target_dir = out_dir / rollout_dir.name if len(args.rollout_dir) > 1 else out_dir
        paths = save_replay_artifacts(
            target_dir,
            replay,
            episode_rows,
            reward_summary,
            {
                **vars(args),
                "rollout_dir": str(rollout_dir),
                "action_source": action_source,
                "inspection": inspection,
            },
        )
        converted.append(
            {
                "rollout_dir": str(rollout_dir),
                "save_dir": str(target_dir),
                "action_source": action_source,
                "num_records": inspection["num_records"],
                "reward_summary": reward_summary,
                "paths": paths,
            }
        )

    report = {"converted": converted, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures and not converted else 0


def inspect_rollout_dir(
    rollout_dir: Path,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    summary_path = rollout_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    summary = load_rollout_summary(summary_path)
    records_by_episode: dict[int, list[dict[str, Any]]] = {}
    record_files: list[str] = []
    missing_by_file: dict[str, list[str]] = {}
    num_records = 0

    for episode in summary.get("episodes", []):
        episode_id = int(episode["episode_id"])
        record_path = Path(str(episode.get("intervention_records_path") or ""))
        if not record_path.is_absolute():
            record_path = Path.cwd() / record_path
        if not record_path.exists():
            fallback = rollout_dir / f"intervention_records_episode_{episode_id:04d}.parquet"
            record_path = fallback if fallback.exists() else record_path
        if not record_path.exists():
            missing_by_file[str(record_path)] = list(REQUIRED_RECORD_FIELDS)
            records_by_episode[episode_id] = []
            continue

        records = load_intervention_records(record_path)
        record_files.append(str(record_path))
        missing = missing_required_fields(records)
        if missing:
            missing_by_file[str(record_path)] = missing
        records_by_episode[episode_id] = records
        num_records += len(records)

    inspection = {
        "num_episodes": len(summary.get("episodes", [])),
        "num_records": num_records,
        "record_files": record_files,
        "required_fields": list(REQUIRED_RECORD_FIELDS),
        "optional_metadata_fields": list(OPTIONAL_METADATA_FIELDS),
        "missing_fields": missing_by_file,
    }
    return summary, records_by_episode, inspection


def missing_required_fields(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    present = set().union(*(record.keys() for record in records))
    return [field for field in REQUIRED_RECORD_FIELDS if field not in present]


def resolve_action_source(raw: str, rollout_dir: Path) -> str:
    if raw != "auto":
        return raw
    name = rollout_dir.name.lower()
    if "bc" in name:
        return "bc"
    if "random_noise" in name or "noise" in name:
        return "random_noise"
    if "zero" in name:
        return "zero"
    if "qpos" in name:
        return "qpos14"
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
