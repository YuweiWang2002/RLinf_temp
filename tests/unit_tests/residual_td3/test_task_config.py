import pytest

from rlinf.algorithms.residual_td3.task_config import (
    load_residual_task_config,
    task_config_to_rollout_defaults,
)
from scripts.rollout_pi05_gate_controlled_hybrid_zero_residual import parse_args


def test_handover_block_task_config_loads():
    cfg = load_residual_task_config("configs/residual_td3/tasks/handover_block.yaml")

    assert cfg.data["task"]["name"] == "handover_block"
    assert cfg.is_runtime_ready
    defaults = task_config_to_rollout_defaults(cfg)
    assert defaults["config"] == "pi05_aloha_robotwin_handover"
    assert defaults["task_name"] == "handover_block"
    assert defaults["chunk_aware_gate_checkpoint"].endswith("chunk_aware_gate_head.pt")


@pytest.mark.parametrize(
    "task_name",
    ["grab_roller", "lift_pot", "handover_mic", "scan_object"],
)
def test_todo_task_configs_parse_but_report_missing_runtime_fields(task_name):
    cfg = load_residual_task_config(f"configs/residual_td3/tasks/{task_name}.yaml")

    assert cfg.data["task"]["name"] == task_name
    assert not cfg.is_runtime_ready
    assert "pi05.checkpoint" in cfg.missing_runtime_fields
    assert "gates.chunk_aware.checkpoint" in cfg.missing_runtime_fields


def test_rollout_task_config_fills_defaults():
    args = parse_args(["--task-config", "configs/residual_td3/tasks/handover_block.yaml"])

    assert args.checkpoint == "/nfs/data3/rlinf_data/pytorch_checkpoint/model.safetensors"
    assert args.save_dir == "logs/residual_td3/handover_block/eval/config_smoke"
    assert args.task_name == "handover_block"
    assert args.task_config_missing_runtime_fields == []


def test_rollout_cli_explicit_args_override_task_config():
    args = parse_args(
        [
            "--task-config",
            "configs/residual_td3/tasks/handover_block.yaml",
            "--checkpoint",
            "local_model.safetensors",
            "--save-dir",
            "local_out",
            "--gate-threshold",
            "0.9",
        ]
    )

    assert args.checkpoint == "local_model.safetensors"
    assert args.save_dir == "local_out"
    assert args.gate_threshold == 0.9


def test_rollout_without_task_config_keeps_required_legacy_args():
    with pytest.raises(SystemExit):
        parse_args([])
