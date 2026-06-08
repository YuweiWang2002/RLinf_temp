"""Residual actor runtime helpers for rollout-time EE intervention."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rlinf.algorithms.residual_td3.fk_bridge import AlohaFKBridge
from rlinf.algorithms.residual_td3.residual_ee_intervention import (
    BCResidualActor,
    ConstantResidualActor,
    RandomNoiseResidualActor,
    ResidualEEInterventionConfig,
    ResidualEEInterventionRunner,
    TD3ResidualActor,
    ZeroInitResidualActor,
    ZeroResidualActor,
)


def build_residual_actor(args: Any):
    """Build the residual actor selected by rollout CLI args."""

    if args.residual_actor == "zero":
        return ZeroResidualActor()
    if args.residual_actor == "zero_init":
        return ZeroInitResidualActor(
            chunk_len=args.residual_horizon_k,
            delta_max=args.residual_max_delta_local_xyz,
            device=args.device,
        )
    if args.residual_actor == "bc":
        if not args.residual_actor_checkpoint:
            raise ValueError("--residual-actor-checkpoint is required for --residual-actor bc.")
        return BCResidualActor.load_from_checkpoint(args.residual_actor_checkpoint, device=args.device)
    if args.residual_actor == "td3":
        if not args.residual_actor_checkpoint:
            raise ValueError("--residual-actor-checkpoint is required for --residual-actor td3.")
        return TD3ResidualActor.load_from_checkpoint(args.residual_actor_checkpoint, device=args.device)
    if args.residual_actor == "random_noise":
        return RandomNoiseResidualActor(
            noise_std=args.residual_noise_std,
            max_delta_local_xyz=args.residual_max_delta_local_xyz,
            seed=args.seed,
        )
    return ConstantResidualActor(tuple(float(v) for v in args.residual_constant_delta_local_xyz))


def build_intervention_config(args: Any, control_flags: dict[str, bool]) -> ResidualEEInterventionConfig:
    """Build rollout-time intervention config from CLI args."""

    return ResidualEEInterventionConfig(
        horizon_k=args.residual_horizon_k,
        target_horizon_offset=args.residual_target_horizon_offset,
        max_delta_local_xyz=args.residual_max_delta_local_xyz,
        left_stabilization_mode=args.left_stabilization_mode,
        left_deadband_xyz=args.left_deadband_xyz,
        left_lowpass_alpha=args.left_lowpass_alpha,
        residual_scale=args.residual_scale,
        dry_run=control_flags["dry_run"],
        learned_residual_control_enabled=control_flags["learned_residual_control_enabled"],
    )


def build_intervention_runner_from_args(
    task: Any,
    args: Any,
    control_flags: dict[str, bool],
) -> ResidualEEInterventionRunner:
    """Build the FK bridge, residual actor, and intervention runner."""

    return ResidualEEInterventionRunner(
        AlohaFKBridge.from_robotwin_task(task, device=args.device),
        build_residual_actor(args),
        build_intervention_config(args, control_flags),
    )


def build_pregrasp_intervention_runner(
    handover_runner: ResidualEEInterventionRunner,
) -> ResidualEEInterventionRunner:
    """Build the zero-residual pregrasp runner sharing the handover FK bridge."""

    cfg = replace(
        handover_runner.config,
        dry_run=False,
        learned_residual_control_enabled=False,
        residual_scale=1.0,
    )
    return ResidualEEInterventionRunner(handover_runner.bridge, ZeroResidualActor(), cfg)
