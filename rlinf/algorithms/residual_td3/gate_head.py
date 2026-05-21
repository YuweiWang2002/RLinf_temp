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
    def predict_from_feature(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = z.to(self.device, dtype=torch.float32)
        logits = self.model(z)
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
        resolved_threshold = (
            float(threshold)
            if threshold is not None
            else float(runtime.get("threshold", config.get("threshold", 0.5)))
        )
        cfg = GateHeadRuntimeConfig(
            checkpoint_path=checkpoint_path,
            feature_dim=int(config["feature_dim"]),
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


def runtime_config_to_dict(cfg: GateHeadRuntimeConfig) -> dict[str, Any]:
    return asdict(cfg)
