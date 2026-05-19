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
"""Config helpers for residual TD3 schema validation."""

from __future__ import annotations

from typing import Any, Mapping

from .action_adapter import ResidualActionAdapter, ResidualActionSpec


def validate_residual_td3_cfg(cfg: Mapping[str, Any]) -> None:
    """Validate residual TD3 config by building and checking an action spec."""

    spec = build_residual_action_spec(cfg)
    ResidualActionAdapter(spec).validate()


def build_residual_action_spec(cfg: Mapping[str, Any]) -> ResidualActionSpec:
    """Build :class:`ResidualActionSpec` from a dict-like config."""

    residual_cfg = _get(cfg, "residual_td3", cfg)
    if residual_cfg is None:
        raise ValueError("Missing residual_td3 config.")
    if not _get(residual_cfg, "enabled", True):
        raise ValueError("residual_td3.enabled must be true to build a residual spec.")

    rel_state_cfg = _get(residual_cfg, "rel_state", {}) or {}
    safety_cfg = _get(residual_cfg, "safety", {}) or {}

    spec = ResidualActionSpec(
        base_action_space=_require(residual_cfg, "base_action_space"),
        residual_mode=_require(residual_cfg, "residual_mode"),
        residual_frame=_get(residual_cfg, "residual_frame", "action"),
        base_action_dim=int(_require(residual_cfg, "base_action_dim")),
        residual_action_indices=_get(residual_cfg, "residual_action_indices", None),
        right_xyz_indices=list(_get(residual_cfg, "right_xyz_indices", [8, 9, 10])),
        residual_chunk_len=int(_get(residual_cfg, "residual_chunk_len", 1)),
        env_action_chunk_len=_optional_int(_get(residual_cfg, "env_action_chunk_len", None)),
        base_action_source=_get(residual_cfg, "base_action_source", "env_action"),
        fail_on_incompatible_action_space=bool(
            _get(safety_cfg, "fail_on_incompatible_action_space", True)
        ),
        delta_max=_get(safety_cfg, "delta_max", None),
        clamp_residual=bool(_get(safety_cfg, "clamp_residual", True)),
        left_ee_rotation_key=_get(rel_state_cfg, "left_ee_rotation_key", "left_ee_rot"),
    )
    ResidualActionAdapter(spec).validate()
    return spec


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _require(cfg: Any, key: str) -> Any:
    value = _get(cfg, key, None)
    if value is None:
        raise ValueError(f"Missing residual_td3.{key}.")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
