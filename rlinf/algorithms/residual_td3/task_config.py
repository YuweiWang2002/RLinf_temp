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
"""Task config loading for residual TD3 prototype scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

RUNTIME_REQUIRED_FIELDS = (
    "task.robotwin_task_name",
    "pi05.config",
    "pi05.checkpoint",
    "pi05.norm_stats_path",
    "gates.chunk_aware.checkpoint",
)


@dataclass(frozen=True)
class ResidualTaskConfig:
    """Loaded residual task config plus runtime completeness metadata."""

    path: Path
    data: dict[str, Any]
    missing_runtime_fields: tuple[str, ...]

    @property
    def is_runtime_ready(self) -> bool:
        """Whether the config contains the fields needed for rollout runtime."""

        return not self.missing_runtime_fields


def load_residual_task_config(path: str | Path) -> ResidualTaskConfig:
    """Load a residual TD3 task YAML file."""

    config_path = Path(path)
    cfg = _load_yaml_mapping(config_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Task config must be a mapping: {config_path}")
    missing = tuple(field for field in RUNTIME_REQUIRED_FIELDS if _is_missing(_get_path(cfg, field)))
    return ResidualTaskConfig(
        path=config_path,
        data=cfg,
        missing_runtime_fields=missing,
    )


def task_config_to_rollout_defaults(task_cfg: ResidualTaskConfig) -> dict[str, Any]:
    """Map task config fields to the existing rollout CLI argument names."""

    cfg = task_cfg.data
    defaults = {
        "task_config": str(task_cfg.path),
        "task_config_missing_runtime_fields": list(task_cfg.missing_runtime_fields),
    }
    _set_if_present(defaults, "config", _get_path(cfg, "pi05.config"))
    _set_if_present(defaults, "checkpoint", _get_path(cfg, "pi05.checkpoint"))
    _set_if_present(defaults, "norm_stats_path", _get_path(cfg, "pi05.norm_stats_path"))
    _set_if_present(defaults, "env_config", _get_path(cfg, "robotwin.env_config"))
    _set_if_present(defaults, "task_name", _get_path(cfg, "task.robotwin_task_name"))
    _set_if_present(
        defaults,
        "chunk_aware_gate_checkpoint",
        _get_path(cfg, "gates.chunk_aware.checkpoint"),
    )
    _set_if_present(defaults, "gate_threshold", _get_path(cfg, "gates.chunk_aware.threshold"))
    _set_if_present(defaults, "gate_chunk_len", _get_path(cfg, "gates.chunk_aware.chunk_len"))
    _set_if_present(defaults, "execution_mode", _get_path(cfg, "rollout.execution_mode"))
    _set_if_present(defaults, "save_dir", _get_path(cfg, "rollout.save_dir"))
    _set_if_present(defaults, "num_episodes", _get_path(cfg, "eval.normal.num_episodes"))
    _set_if_present(defaults, "max_steps", _get_path(cfg, "replay.max_steps"))
    _set_if_present(defaults, "seed", _get_path(cfg, "replay.seed"))
    _set_if_present(
        defaults,
        "enable_pregrasp_intervention",
        _get_path(cfg, "stages.pregrasp.intervention_enabled"),
    )
    _set_if_present(defaults, "residual_actor", _get_path(cfg, "intervention.residual.actor_type"))
    _set_if_present(
        defaults,
        "residual_actor_checkpoint",
        _get_path(cfg, "intervention.residual.actor_checkpoint"),
    )
    _set_if_present(defaults, "residual_scale", _get_path(cfg, "intervention.residual.scale"))
    _set_if_present(defaults, "residual_horizon_k", _get_path(cfg, "intervention.horizon_k"))
    _set_if_present(defaults, "residual_max_delta_local_xyz", _get_path(cfg, "intervention.delta_max"))
    _set_if_present(defaults, "ee16_execution_strategy", _get_path(cfg, "intervention.execution_strategy"))
    return defaults


def _get_path(cfg: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = cfg
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _set_if_present(defaults: dict[str, Any], key: str, value: Any) -> None:
    if not _is_missing(value):
        defaults[key] = value


def _is_missing(value: Any) -> bool:
    return value is None or value == "TODO"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except ModuleNotFoundError:
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(data, dict):
            return data
        return data
