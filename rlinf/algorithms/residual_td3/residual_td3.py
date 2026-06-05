# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Offline TD3 helpers for xyz residual actors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .residual_actor import ResidualActorConfig, ZeroInitResidualActorMLP


@dataclass(frozen=True)
class ResidualTD3Config:
    """Configuration for offline TD3 smoke updates."""

    obs_dim: int = 21
    action_dim: int = 3
    hidden_dim: int = 256
    delta_max: float = 0.05
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.001
    target_noise_clip: float = 0.003
    actor_bc_weight: float = 100.0
    actor_l2_weight: float = 0.0
    actor_bc_mode: str = "all"
    awbc_temperature: float = 1.0
    q_filter_margin: float = 0.0


class ResidualCritic(nn.Module):
    """Q-network for residual TD3."""

    def __init__(self, obs_dim: int = 21, action_dim: int = 3, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q values with shape ``[B, 1]``."""
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs must have shape [B, {self.obs_dim}], got {tuple(obs.shape)}.")
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action must have shape [B, {self.action_dim}], got {tuple(action.shape)}."
            )
        return self.net(torch.cat([obs.float(), action.float()], dim=-1))


class NpzReplayDataset:
    """In-memory numpy replay sampler."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "NpzReplayDataset":
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        return cls(arrays)

    def __len__(self) -> int:
        return int(self.arrays["obs"].shape[0])

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay.")
        indices = rng.integers(0, len(self), size=int(batch_size))
        return {
            "obs": tensor(self.arrays["obs"][indices], device),
            "action": tensor(self.arrays["action"][indices], device),
            "reward": tensor(self.arrays["reward"][indices], device).reshape(-1, 1),
            "next_obs": tensor(self.arrays["next_obs"][indices], device),
            "done": tensor(self.arrays["done"][indices].astype(np.float32), device).reshape(-1, 1),
        }

    def _validate(self) -> None:
        required = ("obs", "action", "reward", "next_obs", "done")
        missing = [key for key in required if key not in self.arrays]
        if missing:
            raise KeyError(f"Replay missing required fields: {missing}")
        if self.arrays["obs"].ndim != 2:
            raise ValueError("obs must have shape [N, obs_dim].")
        if self.arrays["action"].ndim != 2:
            raise ValueError("action must have shape [N, action_dim].")
        n = self.arrays["obs"].shape[0]
        for key in required:
            if self.arrays[key].shape[0] != n:
                raise ValueError(f"{key} first dimension does not match obs.")


def load_actor_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    delta_max: float | None = None,
) -> ZeroInitResidualActorMLP:
    """Load a residual actor checkpoint, optionally overriding delta bounds."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor_config = dict(checkpoint["actor_config"])
    if delta_max is not None:
        actor_config["delta_max"] = float(delta_max)
    actor = ZeroInitResidualActorMLP(ResidualActorConfig(**actor_config)).to(device)
    actor.load_state_dict(checkpoint["model_state_dict"])
    return actor


def actor_action(actor: ZeroInitResidualActorMLP, obs: torch.Tensor) -> torch.Tensor:
    """Return first-step xyz residual action from a chunk actor."""
    return actor(obs)[:, 0, :]


def td3_update(
    *,
    actor: ZeroInitResidualActorMLP,
    actor_target: ZeroInitResidualActorMLP,
    critic1: ResidualCritic,
    critic2: ResidualCritic,
    critic1_target: ResidualCritic,
    critic2_target: ResidualCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    cfg: ResidualTD3Config,
    update_step: int,
) -> dict[str, float]:
    """Run one TD3 update and return scalar metrics."""
    obs = batch["obs"]
    action = torch.clamp(batch["action"], -cfg.delta_max, cfg.delta_max)
    reward = batch["reward"]
    next_obs = batch["next_obs"]
    done = batch["done"]

    with torch.no_grad():
        noise = torch.randn_like(action) * cfg.target_noise
        noise = torch.clamp(noise, -cfg.target_noise_clip, cfg.target_noise_clip)
        next_action = torch.clamp(actor_action(actor_target, next_obs) + noise, -cfg.delta_max, cfg.delta_max)
        q1_target = critic1_target(next_obs, next_action)
        q2_target = critic2_target(next_obs, next_action)
        q_target = reward + cfg.gamma * (1.0 - done) * torch.minimum(q1_target, q2_target)

    q1 = critic1(obs, action)
    q2 = critic2(obs, action)
    critic_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss_value = float("nan")
    actor_q_mean_value = float("nan")
    actor_bc_loss_value = float("nan")
    actor_l2_loss_value = float("nan")
    if update_step % cfg.policy_delay == 0:
        pi_action = actor_action(actor, obs)
        q_pi = critic1(obs, pi_action)
        bc_loss = actor_bc_loss(
            critic=critic1,
            obs=obs,
            pi_action=pi_action,
            behavior_action=action,
            mode=cfg.actor_bc_mode,
            temperature=cfg.awbc_temperature,
            q_filter_margin=cfg.q_filter_margin,
        )
        l2_loss = pi_action.square().mean()
        actor_loss = -q_pi.mean() + cfg.actor_bc_weight * bc_loss + cfg.actor_l2_weight * l2_loss
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optimizer.step()
        soft_update(actor_target, actor, cfg.tau)
        soft_update(critic1_target, critic1, cfg.tau)
        soft_update(critic2_target, critic2, cfg.tau)
        actor_loss_value = float(actor_loss.detach().cpu())
        actor_q_mean_value = float(q_pi.detach().mean().cpu())
        actor_bc_loss_value = float(bc_loss.detach().cpu())
        actor_l2_loss_value = float(l2_loss.detach().cpu())

    td_error = (q1.detach() - q_target.detach()).abs()
    current_action = actor_action(actor, obs).detach()
    return {
        "critic_loss": float(critic_loss.detach().cpu()),
        "actor_loss": actor_loss_value,
        "actor_q_mean": actor_q_mean_value,
        "actor_bc_loss": actor_bc_loss_value,
        "actor_l2_loss": actor_l2_loss_value,
        "q_mean": float(q1.detach().mean().cpu()),
        "q_target_mean": float(q_target.detach().mean().cpu()),
        "reward_mean": float(reward.detach().mean().cpu()),
        "td_error_mean": float(td_error.mean().cpu()),
        "action_norm_mean": float(torch.linalg.norm(current_action, dim=1).mean().cpu()),
        "action_norm_max": float(torch.linalg.norm(current_action, dim=1).max().cpu()),
    }


def actor_bc_loss(
    *,
    critic: ResidualCritic,
    obs: torch.Tensor,
    pi_action: torch.Tensor,
    behavior_action: torch.Tensor,
    mode: str,
    temperature: float,
    q_filter_margin: float,
) -> torch.Tensor:
    """Return behavior regularization loss for actor update."""
    per_sample = (pi_action - behavior_action).square().mean(dim=1, keepdim=True)
    if mode == "none":
        return torch.zeros((), dtype=per_sample.dtype, device=per_sample.device)
    if mode == "all":
        return per_sample.mean()
    with torch.no_grad():
        q_behavior = critic(obs, behavior_action)
        q_pi = critic(obs, pi_action)
        advantage = q_behavior - q_pi
    if mode == "q_filter":
        mask = (advantage > float(q_filter_margin)).to(dtype=per_sample.dtype)
        if mask.sum() <= 0:
            return torch.zeros((), dtype=per_sample.dtype, device=per_sample.device)
        return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
    if mode == "advantage_weighted":
        if temperature <= 0.0:
            raise ValueError("awbc_temperature must be positive.")
        weights = torch.exp(torch.clamp(advantage / float(temperature), min=-20.0, max=20.0))
        weights = weights / weights.mean().clamp_min(1e-6)
        return (per_sample * weights).mean()
    raise ValueError(f"Unknown actor_bc_mode: {mode}")


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak-average target parameters toward source parameters."""
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def clone_module(module: nn.Module) -> nn.Module:
    """Deep-copy a torch module through state_dict."""
    import copy

    clone = copy.deepcopy(module)
    clone.load_state_dict(module.state_dict())
    return clone


def save_td3_checkpoints(
    output_dir: str | Path,
    *,
    actor: ZeroInitResidualActorMLP,
    critic1: ResidualCritic,
    critic2: ResidualCritic,
    cfg: ResidualTD3Config,
    metrics: dict[str, Any],
) -> dict[str, str]:
    """Save actor and critics in eval-compatible checkpoint files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    actor_path = output / "actor_td3.pt"
    critic_path = output / "critic_td3.pt"
    torch.save(
        {
            "model_state_dict": actor.state_dict(),
            "actor_config": actor.cfg.__dict__,
            "algorithm": "offline_td3",
            "td3_config": asdict(cfg),
            "metrics": metrics,
        },
        actor_path,
    )
    torch.save(
        {
            "critic1_state_dict": critic1.state_dict(),
            "critic2_state_dict": critic2.state_dict(),
            "critic_config": {
                "obs_dim": critic1.obs_dim,
                "action_dim": critic1.action_dim,
                "hidden_dim": critic1.hidden_dim,
            },
            "algorithm": "offline_td3",
            "td3_config": asdict(cfg),
            "metrics": metrics,
        },
        critic_path,
    )
    return {"actor_path": str(actor_path), "critic_path": str(critic_path)}


def tensor(array: np.ndarray, device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)
