import json
from pathlib import Path

import numpy as np
import pytest

from scripts import residual_td3_pipeline as pipeline

TASK_CONFIG = "configs/residual_td3/tasks/handover_block.yaml"


def test_pipeline_output_dir_path():
    path = pipeline.output_dir_for("handover_block", "rollout-eval", "smoke")

    assert path.as_posix().endswith("logs/residual_td3/handover_block/rollout-eval/smoke")


def test_pipeline_builds_rollout_command_with_cli_override():
    args, passthrough = pipeline.parse_args(
        [
            "rollout-eval",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "smoke",
            "--execution-mode",
            "qpos14_baseline",
            "--num-episodes",
            "1",
            "--max-steps",
            "100",
            "--",
            "--save-debug",
        ]
    )
    passthrough = pipeline.normalize_passthrough(passthrough)
    task_cfg = pipeline.load_task_config(TASK_CONFIG)
    command = pipeline.build_wrapped_command(
        args,
        task_cfg,
        Path("logs/residual_td3/handover_block/rollout-eval/smoke"),
        passthrough,
    )

    assert "--task-config" in command
    assert "--save-dir" in command
    assert command[command.index("--execution-mode") + 1] == "qpos14_baseline"
    assert command[command.index("--num-episodes") + 1] == "1"
    assert command[command.index("--max-steps") + 1] == "100"
    assert "--save-debug" in command


def test_pipeline_dry_run_writes_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOG_ROOT", tmp_path)
    code = pipeline.main(
        [
            "rollout-eval",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "dry",
            "--execution-mode",
            "qpos14_baseline",
            "--dry-run",
        ]
    )

    out_dir = tmp_path / "handover_block" / "rollout-eval" / "dry"
    assert code == 0
    assert (out_dir / "config_snapshot.yaml").exists()
    assert (out_dir / "command.txt").exists()
    assert (out_dir / "git_commit.txt").exists()
    assert (out_dir / "seed_info.json").exists()
    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    assert metadata["task_name"] == "handover_block"
    assert metadata["subcommand"] == "rollout-eval"
    assert metadata["dry_run"]
    assert metadata["mock_runtime"] is False
    assert metadata["program_success"] is None


def test_pipeline_missing_task_config_has_clear_error(capsys):
    code = pipeline.main(
        [
            "rollout-eval",
            "--task-config",
            "configs/residual_td3/tasks/does_not_exist.yaml",
            "--run-name",
            "missing",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "Task config not found" in captured.err


def test_collect_replay_command_uses_unified_output_dir():
    args, passthrough = pipeline.parse_args(
        [
            "collect-replay",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "collect",
            "--action-source",
            "random_noise",
        ]
    )
    task_cfg = pipeline.load_task_config(TASK_CONFIG)
    output_dir = Path("logs/residual_td3/handover_block/collect-replay/collect")

    command = pipeline.build_wrapped_command(args, task_cfg, output_dir, passthrough)

    assert command[command.index("--save-dir") + 1] == str(output_dir)
    assert command[command.index("--rollout-save-dir") + 1] == str(output_dir / "rollout")
    assert command[command.index("--action-source") + 1] == "random_noise"


def test_load_task_config_raises_for_missing_file():
    with pytest.raises(FileNotFoundError, match="Task config not found"):
        pipeline.load_task_config("configs/residual_td3/tasks/does_not_exist.yaml")


def test_mock_rollout_eval_writes_summary_without_subprocess(tmp_path, monkeypatch):
    real_subprocess_run = pipeline.subprocess.run

    def fail_subprocess(*args, **kwargs):
        command = args[0] if args else kwargs.get("args", [])
        if command and command[0] == "git":
            return real_subprocess_run(*args, **kwargs)
        raise AssertionError("mock runtime must not launch wrapped rollout subprocess")

    monkeypatch.setattr(pipeline, "LOG_ROOT", tmp_path)
    monkeypatch.setattr(pipeline.subprocess, "run", fail_subprocess)

    code = pipeline.main(
        [
            "rollout-eval",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "mock_rollout",
            "--execution-mode",
            "handover_only_k50",
            "--num-episodes",
            "2",
            "--max-steps",
            "100",
            "--mock-runtime",
        ]
    )

    out_dir = tmp_path / "handover_block" / "rollout-eval" / "mock_rollout"
    summary = json.loads((out_dir / "summary.json").read_text())
    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    assert code == 0
    assert summary["mock_runtime"] is True
    assert summary["real_env"] is False
    assert summary["real_model"] is False
    assert len(summary["episodes"]) == 2
    assert (out_dir / "hybrid_log_episode_0000.parquet").exists()
    assert (out_dir / "intervention_records_episode_0000.parquet").exists()
    assert metadata["mock_runtime"] is True


def test_mock_collect_replay_writes_expected_npz_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOG_ROOT", tmp_path)

    code = pipeline.main(
        [
            "collect-replay",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "mock_collect",
            "--action-source",
            "zero",
            "--num-episodes",
            "2",
            "--max-steps",
            "100",
            "--mock-runtime",
        ]
    )

    out_dir = tmp_path / "handover_block" / "collect-replay" / "mock_collect"
    assert code == 0
    assert (out_dir / "episodes_summary.csv").exists()
    assert (out_dir / "reward_summary.json").exists()
    assert (out_dir / "config.json").exists()
    with np.load(out_dir / "replay.npz") as replay:
        assert replay["obs"].shape[1:] == (21,)
        assert replay["action"].shape[1:] == (3,)
        assert replay["next_obs"].shape[1:] == (21,)
        assert replay["reward"].shape == replay["done"].shape
        assert replay["obs"].shape[0] > 0
    reward_summary = json.loads((out_dir / "reward_summary.json").read_text())
    assert reward_summary["mock_runtime"] is True


def test_mock_train_td3_cpu_runs_from_mock_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOG_ROOT", tmp_path)
    collect_code = pipeline.main(
        [
            "collect-replay",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "mock_collect_train",
            "--action-source",
            "zero",
            "--num-episodes",
            "1",
            "--max-steps",
            "50",
            "--mock-runtime",
        ]
    )
    replay_path = tmp_path / "handover_block" / "collect-replay" / "mock_collect_train" / "replay.npz"

    train_code = pipeline.main(
        [
            "train-td3",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "mock_train",
            "--replay",
            str(replay_path),
            "--updates",
            "2",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ]
    )

    out_dir = tmp_path / "handover_block" / "train-td3" / "mock_train"
    assert collect_code == 0
    assert train_code == 0
    assert (out_dir / "actor_td3.pt").exists()
    assert (out_dir / "critic_td3.pt").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "summary.json").exists()


def test_task_failure_with_complete_rollout_artifacts_exits_zero(tmp_path):
    out_dir = tmp_path / "rollout"
    out_dir.mkdir()
    hybrid = out_dir / "hybrid_log_episode_0000.parquet"
    records = out_dir / "intervention_records_episode_0000.parquet"
    hybrid.write_text("", encoding="utf-8")
    records.write_text("", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "success_rate": 0.0,
                "episodes": [
                    {
                        "hybrid_log_path": str(hybrid),
                        "intervention_records_path": str(records),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args, _ = pipeline.parse_args(
        [
            "rollout-eval",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "task_fail",
        ]
    )

    result = pipeline.evaluate_run_result(args, out_dir, raw_exit_code=1)

    assert result["program_success"] is True
    assert result["task_success"] is False
    assert result["artifact_complete"] is True
    assert result["failure_type"] == "task_failure"
    assert result["exit_code"] == 0


def test_fail_on_task_failure_exits_one(tmp_path):
    out_dir = tmp_path / "rollout"
    out_dir.mkdir()
    hybrid = out_dir / "hybrid_log_episode_0000.parquet"
    records = out_dir / "intervention_records_episode_0000.parquet"
    hybrid.write_text("", encoding="utf-8")
    records.write_text("", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "success_rate": 0.0,
                "episodes": [
                    {
                        "hybrid_log_path": str(hybrid),
                        "intervention_records_path": str(records),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args, _ = pipeline.parse_args(
        [
            "rollout-eval",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "task_fail",
            "--fail-on-task-failure",
        ]
    )

    result = pipeline.evaluate_run_result(args, out_dir, raw_exit_code=0)

    assert result["failure_type"] == "task_failure"
    assert result["exit_code"] == 1


def test_artifact_missing_exits_one(tmp_path):
    args, _ = pipeline.parse_args(
        [
            "collect-replay",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "missing_artifact",
        ]
    )

    result = pipeline.evaluate_run_result(args, tmp_path, raw_exit_code=0)

    assert result["program_success"] is False
    assert result["artifact_complete"] is False
    assert result["failure_type"] == "artifact_missing"
    assert result["exit_code"] == 1


def test_program_exception_exits_nonzero(tmp_path):
    args, _ = pipeline.parse_args(
        [
            "rollout-eval",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "program_error",
        ]
    )

    result = pipeline.evaluate_run_result(args, tmp_path, raw_exit_code=2)

    assert result["program_success"] is False
    assert result["failure_type"] == "program_error"
    assert result["exit_code"] == 2


def test_collect_replay_complete_artifacts_exit_zero(tmp_path):
    for name in ("replay.npz", "episodes_summary.csv", "config.json"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "reward_summary.json").write_text(
        json.dumps({"success_rate": 0.0}),
        encoding="utf-8",
    )
    args, _ = pipeline.parse_args(
        [
            "collect-replay",
            "--task-config",
            TASK_CONFIG,
            "--run-name",
            "collect_complete",
        ]
    )

    result = pipeline.evaluate_run_result(args, tmp_path, raw_exit_code=1)

    assert result["program_success"] is True
    assert result["task_success"] is False
    assert result["artifact_complete"] is True
    assert result["exit_code"] == 0


def test_update_run_metadata_records_program_and_task_status(tmp_path):
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(json.dumps({"run_name": "x"}), encoding="utf-8")

    pipeline.update_run_metadata(
        tmp_path,
        {
            "program_success": True,
            "task_success": False,
            "artifact_complete": True,
            "exit_code": 0,
            "failure_type": "task_failure",
        },
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["program_success"] is True
    assert metadata["task_success"] is False
    assert metadata["artifact_complete"] is True
    assert metadata["exit_code"] == 0
    assert metadata["failure_type"] == "task_failure"
