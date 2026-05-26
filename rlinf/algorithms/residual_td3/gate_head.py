# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime MLP gate head for frozen VLA features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


class GateHeadMLP(nn.Module):
    """Small binary classifier over frozen VLA features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[-1] != self.feature_dim:
            raise ValueError(f"z must have shape [B, {self.feature_dim}], got {tuple(z.shape)}.")
        return self.net(z)


class ActionChunkEncoder(nn.Module):
    """Encode a fixed-length qpos action chunk for chunk-level gating."""

    def __init__(
        self,
        action_dim: int = 14,
        chunk_len: int = 50,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        encoder_type: str = "flatten_mlp",
    ):
        super().__init__()
        if encoder_type != "flatten_mlp":
            raise ValueError(f"unsupported encoder_type: {encoder_type}")
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.encoder_type = encoder_type
        self.net = nn.Sequential(
            nn.Linear(self.chunk_len * self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        expected = (self.chunk_len, self.action_dim)
        if action_chunk.ndim != 3 or tuple(action_chunk.shape[1:]) != expected:
            raise ValueError(
                "action_chunk must have shape "
                f"[B, {self.chunk_len}, {self.action_dim}], got {tuple(action_chunk.shape)}."
            )
        return self.net(action_chunk.reshape(action_chunk.shape[0], -1))


class ChunkAwareGateHead(nn.Module):
    """Binary gate over VLA hidden state and a qpos action chunk."""

    def __init__(
        self,
        z_dim: int = 1024,
        action_dim: int = 14,
        chunk_len: int = 50,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.z_proj = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim), nn.ReLU())
        self.action_encoder = ActionChunkEncoder(
            action_dim=self.action_dim,
            chunk_len=self.chunk_len,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[-1] != self.z_dim:
            raise ValueError(f"z must have shape [B, {self.z_dim}], got {tuple(z.shape)}.")
        z_hidden = self.z_proj(z)
        action_hidden = self.action_encoder(action_chunk)
        return self.gate_mlp(torch.cat([z_hidden, action_hidden], dim=-1))


@dataclass
class GateHeadRuntimeConfig:
    checkpoint_path: str
    feature_dim: int
    hidden_dim: int
    dropout: float = 0.0
    threshold: float = 0.5
    device: str = "cuda"


class GateHeadRuntime:
    """Loads a trained gate head and predicts binary gates from features."""

    def __init__(self, cfg: GateHeadRuntimeConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = GateHeadMLP(
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        ).to(self.device)
        checkpoint = torch.load(cfg.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def logits_from_feature(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(z):
            z = torch.as_tensor(z)
        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, {self.cfg.feature_dim}], got {tuple(z.shape)}.")
        if z.shape[-1] != self.cfg.feature_dim:
            raise ValueError(f"z must have shape [B, {self.cfg.feature_dim}], got {tuple(z.shape)}.")
        z = z.to(self.device, dtype=torch.float32)
        return self.model(z)

    @torch.no_grad()
    def predict_from_feature(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits_from_feature(z)
        prob = torch.sigmoid(logits)
        gate_binary = (prob >= self.cfg.threshold).to(dtype=torch.float32)
        return prob, gate_binary

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cuda",
        threshold: float | None = None,
    ) -> "GateHeadRuntime":
        checkpoint_path = str(checkpoint_path)
        checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        runtime = checkpoint.get("runtime", {})
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        inferred_feature_dim = _infer_linear_in_features(state_dict)
        inferred_hidden_dim = _infer_linear_out_features(state_dict)
        resolved_threshold = (
            float(threshold)
            if threshold is not None
            else float(runtime.get("threshold", config.get("threshold", 0.5)))
        )
        cfg = GateHeadRuntimeConfig(
            checkpoint_path=checkpoint_path,
            feature_dim=int(config.get("feature_dim", inferred_feature_dim)),
            hidden_dim=int(config.get("hidden_dim", inferred_hidden_dim)),
            dropout=float(config.get("dropout", 0.0)),
            threshold=resolved_threshold,
            device=device,
        )
        return cls(cfg)


@dataclass
class ChunkAwareGateRuntimeConfig:
    checkpoint_path: str
    z_dim: int = 1024
    action_dim: int = 14
    chunk_len: int = 50
    hidden_dim: int = 256
    dropout: float = 0.0
    threshold: float = 0.5
    device: str = "cuda"


class ChunkAwareGateRuntime:
    """Loads a chunk-aware gate head and predicts chunk-level gates."""

    def __init__(self, cfg: ChunkAwareGateRuntimeConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = ChunkAwareGateHead(
            z_dim=cfg.z_dim,
            action_dim=cfg.action_dim,
            chunk_len=cfg.chunk_len,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        ).to(self.device)
        checkpoint = torch.load(cfg.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def logits_from_inputs(self, z: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        action_chunk = torch.as_tensor(action_chunk, device=self.device, dtype=torch.float32)
        return self.model(z, action_chunk)

    @torch.no_grad()
    def predict_from_inputs(self, z: torch.Tensor, action_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits_from_inputs(z, action_chunk)
        prob = torch.sigmoid(logits)
        gate_binary = (prob >= self.cfg.threshold).to(dtype=torch.float32)
        return prob, gate_binary

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cuda",
        threshold: float | None = None,
    ) -> "ChunkAwareGateRuntime":
        checkpoint_path = str(checkpoint_path)
        checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        runtime = checkpoint.get("runtime", {})
        resolved_threshold = (
            float(threshold)
            if threshold is not None
            else float(runtime.get("threshold", config.get("threshold", 0.5)))
        )
        cfg = ChunkAwareGateRuntimeConfig(
            checkpoint_path=checkpoint_path,
            z_dim=int(config.get("z_dim", config.get("feature_dim", 1024))),
            action_dim=int(config.get("action_dim", 14)),
            chunk_len=int(config.get("chunk_len", 50)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            dropout=float(config.get("dropout", 0.0)),
            threshold=resolved_threshold,
            device=device,
        )
        return cls(cfg)


def build_gate_head_checkpoint(
    *,
    model: GateHeadMLP,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    payload_config = {
        "feature_dim": model.feature_dim,
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
        **config,
    }
    return {
        "model_state_dict": model.state_dict(),
        "config": payload_config,
        "metrics": metrics or {},
        "runtime": {
            "threshold": float(threshold),
            "checkpoint_format": "rlinf.residual_td3.gate_head.v1",
        },
    }


def save_gate_head_checkpoint(
    path: str | Path,
    *,
    model: GateHeadMLP,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_gate_head_checkpoint(
            model=model,
            config=config,
            metrics=metrics,
            threshold=threshold,
        ),
        path,
    )


def build_chunk_aware_gate_checkpoint(
    *,
    model: ChunkAwareGateHead,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    payload_config = {
        "z_dim": model.z_dim,
        "action_dim": model.action_dim,
        "chunk_len": model.chunk_len,
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
        **config,
    }
    return {
        "model_state_dict": model.state_dict(),
        "config": payload_config,
        "metrics": metrics or {},
        "runtime": {
            "threshold": float(threshold),
            "checkpoint_format": "rlinf.residual_td3.chunk_aware_gate_head.v1",
        },
    }


def save_chunk_aware_gate_checkpoint(
    path: str | Path,
    *,
    model: ChunkAwareGateHead,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_chunk_aware_gate_checkpoint(
            model=model,
            config=config,
            metrics=metrics,
            threshold=threshold,
        ),
        path,
    )


def runtime_config_to_dict(cfg: GateHeadRuntimeConfig) -> dict[str, Any]:
    return asdict(cfg)


def gate_seq_to_scalar(gate_seq: torch.Tensor, min_positive_frames: int = 1) -> torch.Tensor:
    """Convert per-frame chunk labels to a scalar chunk label."""
    if gate_seq.ndim != 3 or gate_seq.shape[-1] != 1:
        raise ValueError(f"gate_seq must have shape [B, C, 1], got {tuple(gate_seq.shape)}.")
    if min_positive_frames <= 0:
        raise ValueError("min_positive_frames must be positive.")
    positives = gate_seq.to(dtype=torch.float32).sum(dim=1)
    return (positives >= float(min_positive_frames)).to(dtype=torch.float32)


def _infer_linear_in_features(state_dict: dict[str, torch.Tensor]) -> int:
    for key in ("net.0.weight", "module.net.0.weight"):
        weight = state_dict.get(key)
        if torch.is_tensor(weight) and weight.ndim == 2:
            return int(weight.shape[1])
    raise KeyError("checkpoint config must contain feature_dim or net.0.weight.")


def _infer_linear_out_features(state_dict: dict[str, torch.Tensor]) -> int:
    for key in ("net.0.weight", "module.net.0.weight"):
        weight = state_dict.get(key)
        if torch.is_tensor(weight) and weight.ndim == 2:
            return int(weight.shape[0])
    raise KeyError("checkpoint config must contain hidden_dim or net.0.weight.")
