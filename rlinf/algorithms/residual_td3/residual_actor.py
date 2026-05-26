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
"""Residual actor networks for right-xyz EE residual inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ResidualActorConfig:
    """Configuration for the small residual actor MLP."""

    obs_dim: int = 21
    hidden_dim: int = 128
    chunk_len: int = 5
    residual_dim: int = 3
    delta_max: float = 0.02
    zero_init_output: bool = True


class ZeroInitResidualActorMLP(nn.Module):
    """MLP actor with zero-initialized output layer.

    The actor predicts local-frame right-xyz residuals shaped ``[B, C, 3]``.
    With ``zero_init_output=True`` the initial inference path is exactly zero.
    """

    def __init__(self, cfg: ResidualActorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ResidualActorConfig()
        if self.cfg.obs_dim <= 0:
            raise ValueError("obs_dim must be positive.")
        if self.cfg.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.cfg.chunk_len <= 0:
            raise ValueError("chunk_len must be positive.")
        if self.cfg.residual_dim <= 0:
            raise ValueError("residual_dim must be positive.")
        if self.cfg.delta_max < 0:
            raise ValueError("delta_max must be non-negative.")
        self.net = nn.Sequential(
            nn.Linear(self.cfg.obs_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.chunk_len * self.cfg.residual_dim),
        )
        if self.cfg.zero_init_output:
            self.zero_init_output_layer()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return bounded residuals with shape ``[B, C, residual_dim]``."""
        if obs.ndim != 2 or obs.shape[-1] != self.cfg.obs_dim:
            raise ValueError(f"obs must have shape [B, {self.cfg.obs_dim}], got {tuple(obs.shape)}.")
        out = self.net(obs.to(dtype=torch.float32))
        out = out.reshape(obs.shape[0], self.cfg.chunk_len, self.cfg.residual_dim)
        return torch.tanh(out) * self.cfg.delta_max

    def zero_init_output_layer(self) -> None:
        """Set the final linear layer to produce exactly zero."""
        final = self.net[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Residual actor final layer must be Linear.")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
