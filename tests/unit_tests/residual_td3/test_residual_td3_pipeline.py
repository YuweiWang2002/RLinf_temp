import json
from pathlib import Path

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
