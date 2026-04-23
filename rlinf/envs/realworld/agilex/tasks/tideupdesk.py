from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlinf.envs.realworld.agilex.agilex_env import AgilexEnv, AgilexRobotConfig


@dataclass
class TideupdeskEnvConfig(AgilexRobotConfig):
    task_description: str = "Tidy up the desk"


class TideupdeskEnv(AgilexEnv):
    """Agilex task env for tide-up-desk behavior."""

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        config = TideupdeskEnvConfig(**override_cfg)
        super().__init__(
            config=config,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
        )

    @property
    def task_description(self) -> str:
        return self.config.task_description
