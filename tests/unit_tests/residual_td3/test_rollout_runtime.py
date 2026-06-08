from argparse import Namespace

import pytest

from rlinf.algorithms.residual_td3.rollout_runtime import (
    build_collect_replay_rollout_args,
    build_env_args,
    resolve_residual_control_flags,
)


def _base_args(**overrides):
    values = {
        "env_config": "env.yaml",
        "env_split": "eval",
        "task_name": "handover_block",
        "seed": 100,
        "seed_offset": 0,
        "num_episodes": 2,
        "max_steps": 200,
        "chunk_len": 50,
        "use_eval_success_seeds": False,
        "save_video": False,
        "save_dir": "logs/out",
        "robotwin_assets_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_env_args_preserves_legacy_qpos_defaults():
    env_args = build_env_args(_base_args())

    assert env_args.config == "env.yaml"
    assert env_args.execution_mode == "qpos14"
    assert env_args.data_dir == "logs/out/data"
    assert env_args.chunk_len == 50


def test_collect_replay_rollout_args_maps_action_source_and_rollout_dir():
    args = _base_args(
        config="pi05",
        checkpoint="model.safetensors",
        norm_stats_path="norm_stats.json",
        chunk_aware_gate_checkpoint="gate.pt",
        action_source="random_noise",
        residual_actor_checkpoint=None,
        residual_scale=0.5,
        residual_noise_std=0.01,
        residual_horizon_k=25,
        residual_max_delta_local_xyz=0.02,
        residual_target_horizon_offset=3,
        gate_threshold=0.7,
        gate_chunk_len=50,
        model_num_action_chunks=50,
        ee16_execution_strategy="pointwise",
        planner_backend="curobo",
        robotwin_runtime_bootstrap="auto",
        robotwin_path=None,
        curobo_src_path=None,
        robotwin_setup_max_retries=8,
        device="cpu",
        num_images_in_input=3,
        noise_level=0.3,
        save_debug=False,
        video_fps=10,
        video_source="third_view",
        video_frame_mode="replan",
        rollout_save_dir=None,
    )

    rollout_args = build_collect_replay_rollout_args(args)

    assert rollout_args.execution_mode == "gate_controlled_hybrid"
    assert rollout_args.enable_residual_intervention is True
    assert rollout_args.enable_learned_residual_control is True
    assert rollout_args.residual_actor == "random_noise"
    assert rollout_args.save_dir == "logs/out/rollout"
    assert rollout_args.residual_horizon_k == 25
    assert rollout_args.residual_target_horizon_offset == 3


def test_residual_control_flags_reject_conflicting_modes():
    args = Namespace(
        residual_dry_run=True,
        enable_learned_residual_control=True,
        execution_mode="handover_only_k50",
        enable_pregrasp_intervention=False,
    )

    with pytest.raises(ValueError, match="cannot be enabled together"):
        resolve_residual_control_flags(args)
