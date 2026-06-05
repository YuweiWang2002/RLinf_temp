"""Mine hard RoboTwin handover seeds across qpos and hybrid conditions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlinf.algorithms.residual_td3.residual_replay import (
    ReplayRewardConfig,
    build_replay_from_rollout,
    save_replay_artifacts,
)
from scripts.convert_rollout_logs_to_residual_replay import inspect_rollout_dir

DEFAULT_CHECKPOINT = "/nfs/data3/rlinf_data/pytorch_checkpoint/model.safetensors"
DEFAULT_NORM_STATS = (
    "/nfs/data3/rlinf_data/pytorch_checkpoint/handover_expert/norm_stats.json"
)
DEFAULT_GATE_CHECKPOINT = (
    "/home/user/wyw/RLinf/logs/chunk_aware_gate/full/gate_head/"
    "chunk_aware_gate_head.pt"
)
DEFAULT_BC_ACTOR_CHECKPOINT = (
    "/home/user/wyw/RLinf/logs/residual_actor_bc/hard_pi05_cached_base_k50_s001/"
    "residual_actor_bc.pt"
)
ROLLOUT_SCRIPT = Path("scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py")
SUMMARY_FIELDS = [
    "seed",
    "condition",
    "success",
    "failure_reason",
    "episode_len",
    "first_gate_step",
    "num_interventions",
    "total_ee_steps",
    "video_path",
    "debug_path",
    "log_path",
    "rollout_status",
    "return_code",
]
CONDITIONS = ("qpos14", "zero_k50", "bc_k50", "zero_k10", "bc_s05")
LARGE_SUCCESS_PATTERNS = (
    "hybrid_log_episode_*.parquet",
    "intervention_records_episode_*.parquet",
    "hybrid_log_episode_*.csv",
    "episode_*_debug.json",
    "plots/episode_*.png",
    "videos/episode_*.mp4",
)


@dataclass(frozen=True)
class EpisodeResult:
    """Compact episode result for cross-condition seed mining."""

    seed: int
    condition: str
    success: bool
    failure_reason: str
    episode_len: int | None
    first_gate_step: int | None
    num_interventions: int
    total_ee_steps: int
    video_path: str
    debug_path: str
    log_path: str
    rollout_status: str
    return_code: int

    @property
    def evaluated(self) -> bool:
        """Whether this row reflects a rollout result, not an infra failure."""
        return self.rollout_status == "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=100100000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument(
        "--seed-list",
        default=None,
        help="Comma-separated seeds or a text file with one seed per line.",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--conditions",
        default="qpos14,zero_k50",
        help="Comma-separated subset of qpos14,zero_k50,bc_k50,zero_k10,bc_s05.",
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--save-videos-for",
        choices=("failures", "all", "none"),
        default="failures",
    )
    parser.add_argument("--cleanup-success-artifacts", action="store_true")
    parser.add_argument("--cleanup-dry-run", action="store_true")
    parser.add_argument("--save-residual-replay", action="store_true")
    parser.add_argument(
        "--replay-reward-mode",
        choices=("sparse_success",),
        default="sparse_success",
    )
    parser.add_argument("--replay-reward-discount-to-steps", choices=("all", "last"), default="all")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-stats-path", default=DEFAULT_NORM_STATS)
    parser.add_argument("--chunk-aware-gate-checkpoint", default=DEFAULT_GATE_CHECKPOINT)
    parser.add_argument("--bc-actor-checkpoint", default=DEFAULT_BC_ACTOR_CHECKPOINT)
    parser.add_argument("--config", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--env-split", choices=("train", "eval"), default="eval")
    parser.add_argument("--planner-backend", choices=("curobo", "mplib"), default="curobo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--video-frame-mode",
        choices=("replan", "executed_step"),
        default="replan",
    )
    parser.set_defaults(save_debug=True)
    parser.add_argument("--save-debug", dest="save_debug", action="store_true")
    parser.add_argument("--no-save-debug", dest="save_debug", action="store_false")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_args(args)
    save_dir = Path(args.save_dir)
    write_run_config(save_dir, args)
    seeds = shard_seeds(resolve_seeds(args), args.num_shards, args.shard_id)
    conditions = parse_conditions(args.conditions)

    print(f"HARD SEED MINING seeds={len(seeds)} conditions={conditions}")
    if args.num_workers != 1:
        print("NOTE: --num-workers is recorded only; use --num-shards/--shard-id for parallel runs.")

    for seed in seeds:
        for condition in conditions:
            run_one(seed, condition, args)
            refresh_outputs(save_dir, conditions)
            maybe_cleanup(save_dir, args)

    refresh_outputs(save_dir, conditions)
    maybe_cleanup(save_dir, args)
    print(f"SUMMARY CSV: {save_dir / 'summary.csv'}")
    print(f"HARD SEEDS: {save_dir / 'hard_seeds.json'}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.num_seeds < 0:
        raise ValueError("--num-seeds must be non-negative.")
    if args.seed_stride <= 0:
        raise ValueError("--seed-stride must be positive.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards).")
    parse_conditions(args.conditions)


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed_list:
        path = Path(args.seed_list)
        if path.exists():
            raw_items = [
                item
                for line in path.read_text(encoding="utf-8").splitlines()
                for item in line.replace(",", " ").split()
            ]
        else:
            raw_items = args.seed_list.replace(",", " ").split()
        return [int(item) for item in raw_items]
    return [args.seed_start + idx * args.seed_stride for idx in range(args.num_seeds)]


def shard_seeds(seeds: list[int], num_shards: int, shard_id: int) -> list[int]:
    return [seed for index, seed in enumerate(seeds) if index % num_shards == shard_id]


def parse_conditions(raw: str) -> list[str]:
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Supported: {CONDITIONS}.")
    if not conditions:
        raise ValueError("--conditions must include at least one condition.")
    return conditions


def run_one(seed: int, condition: str, args: argparse.Namespace) -> None:
    rollout_dir = condition_seed_dir(Path(args.save_dir), condition, seed)
    summary_path = rollout_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        print(f"SKIP existing condition={condition} seed={seed}")
        maybe_save_residual_replay(seed, condition, rollout_dir, args)
        return
    rollout_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_rollout_command(seed, condition, rollout_dir, args)
    (rollout_dir / "command.json").write_text(
        json.dumps(cmd, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"RUN condition={condition} seed={seed} dir={rollout_dir}")
    if args.dry_run:
        print(" ".join(cmd))
        return
    with (rollout_dir / "runner.log").open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    (rollout_dir / "return_code.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
    if proc.returncode != 0:
        print(f"ROLL OUT FAILED condition={condition} seed={seed} rc={proc.returncode}")
        return
    maybe_save_residual_replay(seed, condition, rollout_dir, args)


def build_rollout_command(
    seed: int,
    condition: str,
    rollout_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        args.python,
        str(ROLLOUT_SCRIPT),
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--norm-stats-path",
        args.norm_stats_path,
        "--num-episodes",
        "1",
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(seed),
        "--seed-offset",
        "0",
        "--env-split",
        args.env_split,
        "--save-dir",
        str(rollout_dir),
        "--device",
        args.device,
        "--planner-backend",
        args.planner_backend,
        "--video-fps",
        str(args.video_fps),
        "--video-frame-mode",
        args.video_frame_mode,
    ]
    if args.save_debug:
        cmd.append("--save-debug")
    if args.save_videos_for != "none":
        cmd.append("--save-video")
    if condition == "qpos14":
        cmd.extend(["--execution-mode", "qpos14_baseline"])
    else:
        cmd.extend(
            [
                "--execution-mode",
                "gate_controlled_hybrid",
                "--chunk-aware-gate-checkpoint",
                args.chunk_aware_gate_checkpoint,
                "--enable-residual-intervention",
                "--ee16-execution-strategy",
                "pointwise",
                "--residual-horizon-k",
                str(condition_horizon_k(condition)),
            ]
        )
        if condition in {"zero_k50", "zero_k10"}:
            cmd.extend(["--residual-actor", "zero", "--residual-scale", "1.0"])
        elif condition == "bc_k50":
            cmd.extend(
                [
                    "--residual-actor",
                    "bc",
                    "--residual-actor-checkpoint",
                    args.bc_actor_checkpoint,
                    "--residual-scale",
                    "1.0",
                    "--enable-learned-residual-control",
                ]
            )
        elif condition == "bc_s05":
            cmd.extend(
                [
                    "--residual-actor",
                    "bc",
                    "--residual-actor-checkpoint",
                    args.bc_actor_checkpoint,
                    "--residual-scale",
                    "0.5",
                    "--enable-learned-residual-control",
                ]
            )
    return cmd


def condition_horizon_k(condition: str) -> int:
    """Return the residual intervention horizon for a mining condition."""

    if condition.endswith("_k10") or condition == "bc_s05":
        return 10
    return 50


def maybe_save_residual_replay(
    seed: int,
    condition: str,
    rollout_dir: Path,
    args: argparse.Namespace,
) -> None:
    if not args.save_residual_replay:
        return
    replay_dir = rollout_dir / "residual_replay"
    if args.skip_existing and (replay_dir / "replay.npz").exists():
        print(f"SKIP existing replay condition={condition} seed={seed}")
        return
    try:
        summary, records_by_episode, inspection = inspect_rollout_dir(rollout_dir)
    except (FileNotFoundError, ValueError) as exc:
        write_json(
            replay_dir / "conversion_failed.json",
            {
                "condition": condition,
                "seed": seed,
                "error": str(exc),
            },
        )
        print(f"REPLAY CONVERSION FAILED condition={condition} seed={seed}: {exc}")
        return
    if inspection["missing_fields"]:
        write_json(
            replay_dir / "conversion_failed.json",
            {
                "condition": condition,
                "seed": seed,
                "error": "missing required replay fields",
                "inspection": inspection,
            },
        )
        print(f"REPLAY MISSING FIELDS condition={condition} seed={seed}: {inspection['missing_fields']}")
        return

    replay, episode_rows, reward_summary = build_replay_from_rollout(
        summary,
        records_by_episode,
        action_source=condition_action_source(condition),
        reward_config=ReplayRewardConfig(
            mode=args.replay_reward_mode,
            discount_to_steps=args.replay_reward_discount_to_steps,
        ),
    )
    paths = save_replay_artifacts(
        replay_dir,
        replay,
        episode_rows,
        reward_summary,
        {
            "condition": condition,
            "seed": seed,
            "rollout_dir": str(rollout_dir),
            "inspection": inspection,
            "action_source": condition_action_source(condition),
            "replay_reward_mode": args.replay_reward_mode,
            "replay_reward_discount_to_steps": args.replay_reward_discount_to_steps,
        },
    )
    print(
        "REPLAY SAVED "
        f"condition={condition} seed={seed} transitions={reward_summary['num_transitions']} "
        f"path={paths['npz_path']}"
    )


def condition_action_source(condition: str) -> str:
    if condition in {"zero_k50", "zero_k10"}:
        return "zero"
    if condition in {"bc_k50", "bc_s05"}:
        return "bc"
    return condition


def refresh_outputs(save_dir: Path, conditions: list[str]) -> None:
    results = collect_results(save_dir, conditions)
    write_summary_csv(save_dir / "summary.csv", results)
    hard_seeds = build_hard_seed_groups(results)
    write_json(save_dir / "hard_seeds.json", hard_seeds)
    write_failure_records(save_dir / "failures", results, hard_seeds)


def collect_results(save_dir: Path, conditions: list[str]) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for condition in conditions:
        condition_dir = save_dir / "per_condition" / condition
        for rollout_dir in sorted(condition_dir.glob("seed_*")):
            if not rollout_dir.is_dir():
                continue
            seed = int(rollout_dir.name.removeprefix("seed_"))
            results.append(load_episode_result(rollout_dir, condition, seed))
    return sorted(results, key=lambda row: (row.seed, row.condition))


def load_episode_result(rollout_dir: Path, condition: str, seed: int) -> EpisodeResult:
    summary_path = rollout_dir / "summary.json"
    return_code = read_return_code(rollout_dir)
    if not summary_path.exists():
        return EpisodeResult(
            seed=seed,
            condition=condition,
            success=False,
            failure_reason="missing_summary",
            episode_len=None,
            first_gate_step=None,
            num_interventions=0,
            total_ee_steps=0,
            video_path="",
            debug_path="",
            log_path=str(rollout_dir / "runner.log"),
            rollout_status="missing_summary",
            return_code=return_code,
        )
    summary = read_json(summary_path)
    episodes = list(summary.get("episodes", []))
    if not episodes:
        return EpisodeResult(
            seed=seed,
            condition=condition,
            success=False,
            failure_reason="empty_summary",
            episode_len=None,
            first_gate_step=None,
            num_interventions=0,
            total_ee_steps=0,
            video_path="",
            debug_path="",
            log_path=str(rollout_dir / "runner.log"),
            rollout_status=str(summary.get("rollout_status", "empty_summary")),
            return_code=return_code,
        )
    episode = episodes[0]
    success = bool(episode.get("success", False))
    debug_path = rollout_dir / f"episode_{int(episode.get('episode_id', 0)):04d}_debug.json"
    return EpisodeResult(
        seed=seed,
        condition=condition,
        success=success,
        failure_reason=str(episode.get("failure_reason") or ""),
        episode_len=none_or_int(episode.get("episode_length")),
        first_gate_step=none_or_int(
            episode.get("first_gate_step", episode.get("gate_first_activation_env_step"))
        ),
        num_interventions=int(episode.get("num_interventions") or 0),
        total_ee_steps=int(episode.get("total_ee_intervention_steps") or 0),
        video_path=str(episode.get("video_path") or ""),
        debug_path=str(debug_path) if debug_path.exists() else "",
        log_path=str(episode.get("hybrid_log_path") or ""),
        rollout_status=str(summary.get("rollout_status", "completed")),
        return_code=return_code,
    )


def read_return_code(rollout_dir: Path) -> int:
    path = rollout_dir / "return_code.txt"
    if not path.exists():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return 1


def build_hard_seed_groups(results: list[EpisodeResult]) -> dict[str, Any]:
    by_seed: dict[int, dict[str, EpisodeResult]] = {}
    for result in results:
        by_seed.setdefault(result.seed, {})[result.condition] = result

    def failed(seed: int, condition: str) -> bool:
        result = by_seed.get(seed, {}).get(condition)
        return result is not None and result.evaluated and not result.success

    def succeeded(seed: int, condition: str) -> bool:
        result = by_seed.get(seed, {}).get(condition)
        return result is not None and result.evaluated and result.success

    zero_condition = preferred_condition(by_seed, "zero_k50", "zero_k10")
    bc_condition = preferred_condition(by_seed, "bc_k50", "bc_s05")
    qpos_fail = sorted(seed for seed in by_seed if failed(seed, "qpos14"))
    zero_fail = sorted(seed for seed in by_seed if failed(seed, zero_condition))
    bc_fail = sorted(seed for seed in by_seed if failed(seed, bc_condition))
    qpos_fail_zero_success = sorted(
        seed for seed in by_seed if failed(seed, "qpos14") and succeeded(seed, zero_condition)
    )
    qpos_success_zero_fail = sorted(
        seed for seed in by_seed if succeeded(seed, "qpos14") and failed(seed, zero_condition)
    )
    all_fail = sorted(
        seed
        for seed, per_condition in by_seed.items()
        if per_condition
        and all(result.evaluated for result in per_condition.values())
        and all(not result.success for result in per_condition.values())
    )
    infra_fail = sorted(
        seed
        for seed, per_condition in by_seed.items()
        if any(not result.evaluated for result in per_condition.values())
    )
    candidate_hard_eval_seeds = sorted(
        set(qpos_fail) | set(zero_fail) | set(bc_fail) | set(qpos_success_zero_fail)
    )
    payload = {
        "zero_k50_fail": [],
        "bc_k50_fail": [],
        "zero_k10_fail": [],
        "bc_s05_fail": [],
        "qpos_fail": qpos_fail,
        f"{zero_condition}_fail": zero_fail,
        f"{bc_condition}_fail": bc_fail,
        "qpos_fail_zero_success": qpos_fail_zero_success,
        "qpos_success_zero_fail": qpos_success_zero_fail,
        "all_fail": all_fail,
        "infra_fail": infra_fail,
        "candidate_hard_eval_seeds": candidate_hard_eval_seeds,
    }
    return payload


def preferred_condition(
    by_seed: dict[int, dict[str, EpisodeResult]],
    preferred: str,
    fallback: str,
) -> str:
    """Prefer K50 condition names while keeping legacy K10 summaries readable."""

    if any(preferred in per_condition for per_condition in by_seed.values()):
        return preferred
    return fallback


def write_summary_csv(path: Path, results: list[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = temporary_sibling(path)
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(result_to_row(result))
    tmp_path.replace(path)


def result_to_row(result: EpisodeResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "condition": result.condition,
        "success": result.success,
        "failure_reason": result.failure_reason,
        "episode_len": result.episode_len if result.episode_len is not None else "",
        "first_gate_step": (
            result.first_gate_step if result.first_gate_step is not None else ""
        ),
        "num_interventions": result.num_interventions,
        "total_ee_steps": result.total_ee_steps,
        "video_path": result.video_path,
        "debug_path": result.debug_path,
        "log_path": result.log_path,
        "rollout_status": result.rollout_status,
        "return_code": result.return_code,
    }


def write_failure_records(
    failures_dir: Path,
    results: list[EpisodeResult],
    hard_seeds: dict[str, Any],
) -> None:
    failures_dir.mkdir(parents=True, exist_ok=True)
    failure_rows = [result for result in results if not result.success]
    write_json(
        failures_dir / "failure_records.json",
        [result_to_row(result) for result in failure_rows],
    )
    write_json(failures_dir / "hard_seed_groups.json", hard_seeds)
    failures_csv = failures_dir / "failures.csv"
    tmp_path = temporary_sibling(failures_csv)
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in failure_rows:
            writer.writerow(result_to_row(result))
    tmp_path.replace(failures_csv)


def maybe_cleanup(save_dir: Path, args: argparse.Namespace) -> None:
    if args.save_videos_for != "failures" and not args.cleanup_success_artifacts:
        return
    results = collect_results(save_dir, parse_conditions(args.conditions))
    hard_seeds = build_hard_seed_groups(results)
    paths: list[Path] = []
    if args.save_videos_for == "failures":
        paths.extend(success_video_cleanup_candidates(results))
    if args.cleanup_success_artifacts:
        paths.extend(cleanup_candidates(results, hard_seeds))
    paths = unique_paths(paths)
    if not paths:
        return
    mode = "DRY-RUN" if args.cleanup_dry_run else "DELETE"
    for path in paths:
        print(f"CLEANUP {mode}: {path}")
        if not args.cleanup_dry_run:
            remove_path(path)


def cleanup_candidates(
    results: list[EpisodeResult],
    hard_seeds: dict[str, Any],
) -> list[Path]:
    hard_seed_set = set(hard_seeds["candidate_hard_eval_seeds"]) | set(hard_seeds["all_fail"])
    paths: list[Path] = []
    for result in results:
        if not result.success:
            continue
        if result.seed in hard_seed_set:
            continue
        rollout_dir = condition_seed_dir_from_result(result)
        paths.extend(existing_large_artifacts(rollout_dir))
    return unique_paths(paths)


def success_video_cleanup_candidates(results: list[EpisodeResult]) -> list[Path]:
    paths: list[Path] = []
    for result in results:
        if result.success and result.video_path:
            video_path = Path(result.video_path)
            if video_path.exists():
                paths.append(video_path)
    return unique_paths(paths)


def existing_large_artifacts(rollout_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in LARGE_SUCCESS_PATTERNS:
        paths.extend(path for path in rollout_dir.glob(pattern) if path.exists())
    return paths


def condition_seed_dir(save_dir: Path, condition: str, seed: int) -> Path:
    return save_dir / "per_condition" / condition / f"seed_{seed}"


def condition_seed_dir_from_result(result: EpisodeResult) -> Path:
    marker = f"per_condition/{result.condition}/seed_{result.seed}"
    for value in (result.log_path, result.debug_path, result.video_path):
        if marker in value.replace(os.sep, "/"):
            normalized = Path(value)
            parts = normalized.parts
            for index in range(len(parts) - 2):
                if (
                    parts[index] == "per_condition"
                    and parts[index + 1] == result.condition
                    and parts[index + 2] == f"seed_{result.seed}"
                ):
                    return Path(*parts[: index + 3])
    return Path("per_condition") / result.condition / f"seed_{result.seed}"


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def write_run_config(save_dir: Path, args: argparse.Namespace) -> None:
    config_dir = save_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_dir / "config.json", vars(args))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = temporary_sibling(path)
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def temporary_sibling(path: Path) -> Path:
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    return Path(raw_path)


def none_or_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
