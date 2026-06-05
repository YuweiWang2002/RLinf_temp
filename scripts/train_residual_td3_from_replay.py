"""Train a residual TD3 actor from offline replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.residual_td3.residual_td3 import (
    NpzReplayDataset,
    ResidualCritic,
    ResidualTD3Config,
    actor_action,
    clone_module,
    load_actor_from_checkpoint,
    save_td3_checkpoints,
    td3_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", default="logs/residual_replay/mainline_4i_merged/replay.npz")
    parser.add_argument(
        "--bc-actor-checkpoint",
        default="logs/residual_actor_bc/pi05_cached_base_k10/residual_actor_bc.pt",
    )
    parser.add_argument("--output-dir", default="logs/residual_td3/mainline_4i_smoke")
    parser.add_argument("--obs-dim", type=int, default=21)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--delta-max", type=float, default=0.05)
    parser.add_argument("--actor-delta-max", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--target-noise", type=float, default=0.001)
    parser.add_argument("--target-noise-clip", type=float, default=0.003)
    parser.add_argument("--actor-bc-weight", type=float, default=100.0)
    parser.add_argument("--actor-l2-weight", type=float, default=0.0)
    parser.add_argument(
        "--actor-bc-mode",
        choices=("none", "all", "q_filter", "advantage_weighted"),
        default="all",
    )
    parser.add_argument("--awbc-temperature", type=float, default=1.0)
    parser.add_argument("--q-filter-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = train_residual_td3_from_replay(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def train_residual_td3_from_replay(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    dataset = NpzReplayDataset.load(args.replay)
    actor = load_actor_from_checkpoint(
        args.bc_actor_checkpoint,
        device=device,
        delta_max=args.actor_delta_max,
    )
    if actor.cfg.obs_dim != args.obs_dim:
        raise ValueError(f"Actor obs_dim {actor.cfg.obs_dim} does not match --obs-dim {args.obs_dim}.")
    if actor.cfg.residual_dim != args.action_dim:
        raise ValueError(
            f"Actor residual_dim {actor.cfg.residual_dim} does not match --action-dim {args.action_dim}."
        )
    actor_target = clone_module(actor).to(device)
    critic1 = ResidualCritic(args.obs_dim, args.action_dim, args.hidden_dim).to(device)
    critic2 = ResidualCritic(args.obs_dim, args.action_dim, args.hidden_dim).to(device)
    critic1_target = clone_module(critic1).to(device)
    critic2_target = clone_module(critic2).to(device)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.AdamW(
        list(critic1.parameters()) + list(critic2.parameters()),
        lr=args.critic_lr,
    )
    cfg = ResidualTD3Config(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        delta_max=args.delta_max,
        gamma=args.gamma,
        tau=args.tau,
        policy_delay=args.policy_delay,
        target_noise=args.target_noise,
        target_noise_clip=args.target_noise_clip,
        actor_bc_weight=args.actor_bc_weight,
        actor_l2_weight=args.actor_l2_weight,
        actor_bc_mode=args.actor_bc_mode,
        awbc_temperature=args.awbc_temperature,
        q_filter_margin=args.q_filter_margin,
    )
    write_json(output_dir / "config.json", {"args": vars(args), "td3_config": asdict(cfg)})

    metrics_path = output_dir / "metrics.csv"
    rows: list[dict[str, Any]] = []
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "update",
                "critic_loss",
                "actor_loss",
                "actor_q_mean",
                "actor_bc_loss",
                "actor_l2_loss",
                "q_mean",
                "q_target_mean",
                "reward_mean",
                "td_error_mean",
                "action_norm_mean",
                "action_norm_max",
            ],
        )
        writer.writeheader()
        for update in range(1, int(args.updates) + 1):
            batch = dataset.sample(args.batch_size, rng=rng, device=device)
            row = td3_update(
                actor=actor,
                actor_target=actor_target,
                critic1=critic1,
                critic2=critic2,
                critic1_target=critic1_target,
                critic2_target=critic2_target,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                batch=batch,
                cfg=cfg,
                update_step=update,
            )
            row["update"] = update
            check_metrics(row, update)
            rows.append(row)
            writer.writerow(row)
            if update % int(args.log_interval) == 0 or update == 1:
                print(json.dumps(row), flush=True)

    final_metrics = rows[-1] if rows else {}
    checkpoint_paths = save_td3_checkpoints(
        output_dir,
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        cfg=cfg,
        metrics={"final": final_metrics, "num_updates": int(args.updates)},
    )
    actor_smoke = actor_inference_smoke(actor, dataset, device=device, delta_max=args.actor_delta_max)
    summary = {
        "config": vars(args),
        "td3_config": asdict(cfg),
        "num_transitions": len(dataset),
        "updates": int(args.updates),
        "final_metrics": final_metrics,
        "actor_smoke": actor_smoke,
        "checkpoint_paths": checkpoint_paths,
        "metrics_path": str(metrics_path),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def check_metrics(row: dict[str, Any], update: int) -> None:
    delayed_actor_keys = {"actor_loss", "actor_q_mean", "actor_bc_loss", "actor_l2_loss"}
    for key, value in row.items():
        if key in delayed_actor_keys and isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"{key} is not finite at update {update}: {value}")
    if abs(float(row["q_mean"])) > 1e6 or abs(float(row["q_target_mean"])) > 1e6:
        raise FloatingPointError(f"Q value exploded at update {update}: {row}")


def actor_inference_smoke(
    actor: torch.nn.Module,
    dataset: NpzReplayDataset,
    *,
    device: torch.device,
    delta_max: float,
) -> dict[str, Any]:
    count = len(dataset)
    obs = torch.as_tensor(dataset.arrays["obs"][:count], dtype=torch.float32, device=device)
    behavior = torch.as_tensor(dataset.arrays["action"][:count], dtype=torch.float32, device=device)
    with torch.no_grad():
        action = actor_action(actor, obs)
    finite = bool(torch.isfinite(action).all().detach().cpu())
    norm = torch.linalg.norm(action, dim=1)
    behavior_norm = torch.linalg.norm(behavior, dim=1)
    clamp_bound_usage = (torch.abs(action) >= float(delta_max) - 1e-7).float().mean()
    saturation = (torch.abs(action) >= float(delta_max) - 1e-7).any(dim=1).float().mean()
    norm_cpu = norm.detach().cpu().numpy()
    behavior_norm_cpu = behavior_norm.detach().cpu().numpy()
    return {
        "finite": finite,
        "action_norm_mean": float(norm.mean().detach().cpu()) if count else 0.0,
        "action_norm_max": float(norm.max().detach().cpu()) if count else 0.0,
        "action_norm_p95": float(np.percentile(norm_cpu, 95)) if count else 0.0,
        "action_norm_p99": float(np.percentile(norm_cpu, 99)) if count else 0.0,
        "behavior_action_norm_mean": float(behavior_norm.mean().detach().cpu()) if count else 0.0,
        "behavior_action_norm_p95": float(np.percentile(behavior_norm_cpu, 95)) if count else 0.0,
        "behavior_action_norm_p99": float(np.percentile(behavior_norm_cpu, 99)) if count else 0.0,
        "clamp_bound_usage": float(clamp_bound_usage.detach().cpu()) if count else 0.0,
        "saturation_ratio": float(saturation.detach().cpu()) if count else 0.0,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
