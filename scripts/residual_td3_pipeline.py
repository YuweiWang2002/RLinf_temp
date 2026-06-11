"""Thin residual TD3 pipeline wrapper around the current prototype scripts."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG_ROOT = REPO_ROOT / "logs" / "residual_td3"


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    for name in ("rollout-eval", "collect-replay", "train-td3"):
        subparser = subparsers.add_parser(name)
        add_common_args(subparser)
        if name == "rollout-eval":
            add_rollout_args(subparser)
        elif name == "collect-replay":
            add_collect_args(subparser)
        elif name == "train-td3":
            add_train_td3_args(subparser)
    return parser.parse_known_args(argv)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--fail-on-task-failure",
        action="store_true",
        help="Return non-zero when the task itself fails even if program artifacts are complete.",
    )
    parser.add_argument(
        "--mock-runtime",
        action="store_true",
        help="Write no-GPU mock artifacts instead of launching real env/model scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write metadata and print the wrapped command without executing it.",
    )


def add_rollout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution-mode", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-base-dir", default=None)
    parser.add_argument("--num-save-videos", type=int, default=None)
    parser.add_argument("--video-temp-subsample", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--save-actual-state-trace", action="store_true")


def add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action-source", choices=("zero", "bc", "random_noise"), default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)


def add_train_td3_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--replay", default=None)
    parser.add_argument("--bc-actor-checkpoint", default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    passthrough = normalize_passthrough(passthrough)
    try:
        task_cfg = load_task_config(args.task_config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    task_name = require_task_name(task_cfg)
    output_dir = output_dir_for(task_name, args.subcommand, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_wrapped_command(args, task_cfg, output_dir, passthrough)
    write_run_metadata(output_dir, args, task_cfg, command, passthrough)
    if args.dry_run:
        print(json.dumps({"output_dir": str(output_dir), "command": command}, indent=2))
        return 0
    if args.mock_runtime and args.subcommand in {"rollout-eval", "collect-replay"}:
        summary = run_mock_subcommand(args, task_cfg, output_dir)
        result = evaluate_run_result(args, output_dir, raw_exit_code=0)
        update_run_metadata(output_dir, result)
        print(json.dumps({"output_dir": str(output_dir), "mock_runtime": True, "summary": summary}, indent=2))
        return int(result["exit_code"])
    completed = subprocess.run(command, cwd=REPO_ROOT, env=subprocess_env())
    result = evaluate_run_result(args, output_dir, raw_exit_code=int(completed.returncode))
    update_run_metadata(output_dir, result)
    return int(result["exit_code"])


def output_dir_for(task_name: str, subcommand: str, run_name: str) -> Path:
    return LOG_ROOT / task_name / subcommand / run_name


def subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    repo_root = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not existing else repo_root + os.pathsep + existing
    env.setdefault("REPO_PATH", repo_root)
    return env


def normalize_passthrough(passthrough: list[str]) -> list[str]:
    return [arg for arg in passthrough if arg != "--"]


def build_wrapped_command(
    args: argparse.Namespace,
    task_cfg: Any,
    output_dir: Path,
    passthrough: list[str],
) -> list[str]:
    if args.subcommand == "rollout-eval":
        command = [
            sys.executable,
            "scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py",
            "--task-config",
            str(task_cfg.path),
            "--save-dir",
            str(output_dir),
        ]
        append_optional(command, "--execution-mode", args.execution_mode)
        append_optional(command, "--num-episodes", args.num_episodes)
        append_optional(command, "--max-steps", args.max_steps)
        append_optional(command, "--seed", args.seed)
        append_optional(command, "--device", args.device)
        if args.save_video:
            command.append("--save-video")
        append_optional(command, "--video-base-dir", args.video_base_dir)
        append_optional(command, "--num-save-videos", args.num_save_videos)
        append_optional(command, "--video-temp-subsample", args.video_temp_subsample)
        append_optional(command, "--video-fps", args.video_fps)
        if args.save_actual_state_trace:
            command.append("--save-actual-state-trace")
        return command + passthrough

    if args.subcommand == "collect-replay":
        defaults = task_config_to_script_defaults(task_cfg)
        command = [
            sys.executable,
            "scripts/collect_residual_replay.py",
            "--config",
            defaults["config"],
            "--env-config",
            defaults["env_config"],
            "--checkpoint",
            defaults["checkpoint"],
            "--norm-stats-path",
            defaults["norm_stats_path"],
            "--chunk-aware-gate-checkpoint",
            defaults["chunk_aware_gate_checkpoint"],
            "--save-dir",
            str(output_dir),
            "--rollout-save-dir",
            str(output_dir / "rollout"),
            "--action-source",
            str(args.action_source or get_path(task_cfg.data, "replay.action_source") or "zero"),
        ]
        append_optional(command, "--task-name", get_path(task_cfg.data, "task.robotwin_task_name"))
        append_optional(command, "--gate-threshold", get_path(task_cfg.data, "gates.chunk_aware.threshold"))
        append_optional(command, "--gate-chunk-len", get_path(task_cfg.data, "gates.chunk_aware.chunk_len"))
        append_optional(command, "--residual-scale", get_path(task_cfg.data, "intervention.residual.scale"))
        append_optional(command, "--residual-horizon-k", get_path(task_cfg.data, "intervention.horizon_k"))
        append_optional(
            command,
            "--residual-max-delta-local-xyz",
            get_path(task_cfg.data, "intervention.delta_max"),
        )
        append_optional(command, "--num-episodes", args.num_episodes)
        append_optional(command, "--max-steps", args.max_steps)
        append_optional(command, "--seed", args.seed)
        append_optional(command, "--device", args.device)
        return command + passthrough

    if args.subcommand == "train-td3":
        bc_actor_checkpoint = args.bc_actor_checkpoint
        if bc_actor_checkpoint is None:
            bc_actor_checkpoint = str(
                mock_runtime_module().ensure_mock_bc_actor_checkpoint(output_dir / "mock_bc_actor.pt")
            )
        command = [
            sys.executable,
            "scripts/train_residual_td3_from_replay.py",
            "--output-dir",
            str(output_dir),
        ]
        append_optional(command, "--replay", args.replay)
        append_optional(command, "--bc-actor-checkpoint", bc_actor_checkpoint)
        append_optional(command, "--updates", args.updates)
        append_optional(command, "--batch-size", args.batch_size)
        append_optional(command, "--device", args.device)
        return command + passthrough

    raise ValueError(f"Unsupported subcommand: {args.subcommand}")


def task_config_to_script_defaults(task_cfg: Any) -> dict[str, str]:
    values = {
        "config": get_path(task_cfg.data, "pi05.config"),
        "env_config": get_path(task_cfg.data, "robotwin.env_config"),
        "checkpoint": get_path(task_cfg.data, "pi05.checkpoint"),
        "norm_stats_path": get_path(task_cfg.data, "pi05.norm_stats_path"),
        "chunk_aware_gate_checkpoint": get_path(task_cfg.data, "gates.chunk_aware.checkpoint"),
    }
    missing = [key for key, value in values.items() if is_missing(value)]
    if missing:
        raise ValueError(f"Task config is missing runtime fields for wrapper: {missing}")
    return {key: str(value) for key, value in values.items()}


def append_optional(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def run_mock_subcommand(args: argparse.Namespace, task_cfg: Any, output_dir: Path) -> dict[str, Any]:
    mock_runtime = mock_runtime_module()
    seed = int(args.seed or get_path(task_cfg.data, "replay.seed") or 100100000)
    num_episodes = int(args.num_episodes or get_path(task_cfg.data, "eval.num_episodes") or 1)
    max_steps = int(args.max_steps or get_path(task_cfg.data, "replay.max_steps") or 100)
    if args.subcommand == "rollout-eval":
        return mock_runtime.write_mock_rollout_eval(
            output_dir,
            execution_mode=str(args.execution_mode or "handover_only_k50"),
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
            gate_threshold=float(get_path(task_cfg.data, "gates.chunk_aware.threshold") or 0.6),
            residual_horizon_k=int(get_path(task_cfg.data, "intervention.horizon_k") or 50),
        )
    if args.subcommand == "collect-replay":
        return mock_runtime.write_mock_collect_replay(
            output_dir,
            action_source=str(args.action_source or get_path(task_cfg.data, "replay.action_source") or "zero"),
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
        )
    raise ValueError(f"mock runtime does not support subcommand: {args.subcommand}")


def mock_runtime_module() -> Any:
    return importlib.import_module("rlinf.algorithms.residual_td3.mock_runtime")


def evaluate_run_result(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    raw_exit_code: int,
) -> dict[str, Any]:
    artifact_complete = artifacts_complete(args.subcommand, output_dir)
    task_success = read_task_success(args.subcommand, output_dir)
    if artifact_complete:
        program_success = True
        failure_type = "task_failure" if not task_success else "none"
    else:
        program_success = False
        failure_type = "program_error" if raw_exit_code != 0 else "artifact_missing"
    if not program_success:
        exit_code = raw_exit_code if raw_exit_code != 0 else 1
    elif bool(args.fail_on_task_failure) and not task_success:
        exit_code = 1
    else:
        exit_code = 0
    return {
        "program_success": bool(program_success),
        "task_success": bool(task_success),
        "artifact_complete": bool(artifact_complete),
        "raw_exit_code": int(raw_exit_code),
        "exit_code": int(exit_code),
        "failure_type": failure_type,
    }


def artifacts_complete(subcommand: str, output_dir: Path) -> bool:
    if subcommand == "rollout-eval":
        summary = load_json_if_exists(output_dir / "summary.json")
        if summary is None:
            return False
        episodes = summary.get("episodes", [])
        return all(paths_exist(ep.get("hybrid_log_path"), ep.get("intervention_records_path")) for ep in episodes)
    if subcommand == "collect-replay":
        required = ("replay.npz", "episodes_summary.csv", "reward_summary.json", "config.json")
        return all((output_dir / name).exists() for name in required)
    if subcommand == "train-td3":
        required = ("actor_td3.pt", "critic_td3.pt", "metrics.csv", "summary.json")
        return all((output_dir / name).exists() for name in required)
    return False


def read_task_success(subcommand: str, output_dir: Path) -> bool:
    if subcommand == "rollout-eval":
        summary = load_json_if_exists(output_dir / "summary.json")
        return bool(summary and float(summary.get("success_rate", 0.0)) > 0.0)
    if subcommand == "collect-replay":
        summary = load_json_if_exists(output_dir / "reward_summary.json")
        return bool(summary and float(summary.get("success_rate", 0.0)) > 0.0)
    if subcommand == "train-td3":
        summary = load_json_if_exists(output_dir / "summary.json")
        return bool(summary and summary.get("actor_smoke", {}).get("finite", False))
    return False


def paths_exist(*paths: object) -> bool:
    return all(isinstance(path, str) and bool(path) and Path(path).exists() for path in paths)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_run_metadata(output_dir: Path, result: dict[str, Any]) -> None:
    metadata_path = output_dir / "run_metadata.json"
    metadata = load_json_if_exists(metadata_path) or {}
    metadata.update(result)
    write_json(metadata_path, metadata)


def write_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    task_cfg: Any,
    command: list[str],
    passthrough: list[str],
) -> None:
    write_config_snapshot(task_cfg.path, output_dir / "config_snapshot.yaml")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_dir / "git_commit.txt").write_text(git_commit_text(), encoding="utf-8")
    write_json(output_dir / "seed_info.json", seed_info(args, task_cfg))
    write_json(
        output_dir / "run_metadata.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subcommand": args.subcommand,
            "task_config": str(task_cfg.path),
            "task_name": require_task_name(task_cfg),
            "run_name": args.run_name,
            "output_dir": str(output_dir),
            "wrapped_command": command,
            "passthrough_args": passthrough,
            "dry_run": bool(args.dry_run),
            "mock_runtime": bool(args.mock_runtime),
            "real_env": not bool(args.mock_runtime),
            "real_model": not bool(args.mock_runtime),
            "program_success": None,
            "task_success": None,
            "artifact_complete": None,
            "exit_code": None,
            "failure_type": None,
        },
    )


def write_config_snapshot(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


def seed_info(args: argparse.Namespace, task_cfg: Any) -> dict[str, Any]:
    return {
        "seed": getattr(args, "seed", None) or get_path(task_cfg.data, "replay.seed"),
        "seed_list": get_path(task_cfg.data, "replay.seed_list"),
        "hard_seed_set": get_path(task_cfg.data, "replay.hard_seed_set"),
    }


def git_commit_text() -> str:
    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    dirty = "dirty" if status.strip() else "clean"
    return f"{commit.strip()}\n{dirty}\n"


def run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    if completed.returncode != 0:
        return f"unavailable: {completed.stderr.strip()}"
    return completed.stdout


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_task_config(path: str) -> Any:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    module = load_task_config_module()
    return module.load_residual_task_config(config_path)


def load_task_config_module() -> Any:
    import importlib.util

    module_path = REPO_ROOT / "rlinf" / "algorithms" / "residual_td3" / "task_config.py"
    spec = importlib.util.spec_from_file_location("_residual_task_config", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load task config module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_task_name(task_cfg: Any) -> str:
    task_name = get_path(task_cfg.data, "task.name")
    if is_missing(task_name):
        raise ValueError(f"Task config is missing task.name: {task_cfg.path}")
    return str(task_name)


def get_path(cfg: dict[str, Any], dotted_path: str) -> Any:
    value: Any = cfg
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def is_missing(value: Any) -> bool:
    return value is None or value == "TODO"


if __name__ == "__main__":
    raise SystemExit(main())
