# Copyright 2026 The RLinf Authors.
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

from dataclasses import dataclass
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class AgilexHWInfo(HardwareInfo):
    """Hardware information for an Agilex robot-side bridge endpoint."""

    config: "AgilexConfig"


@Hardware.register()
class AgilexRobot(Hardware):
    """Hardware policy for Agilex realworld bridge endpoints."""

    HW_TYPE = "Agilex"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["AgilexConfig"]] = None
    ) -> Optional[HardwareResource]:
        """Enumerate Agilex bridge resources on the given node rank."""
        assert configs is not None, "Agilex hardware requires explicit configurations."
        agilex_configs: list[AgilexConfig] = []
        for config in configs:
            if isinstance(config, AgilexConfig) and config.node_rank == node_rank:
                agilex_configs.append(config)

        if not agilex_configs:
            return None

        agilex_infos: list[AgilexHWInfo] = []
        for config in agilex_configs:
            agilex_infos.append(
                AgilexHWInfo(
                    type=cls.HW_TYPE,
                    model=cls.HW_TYPE,
                    config=config,
                )
            )
        return HardwareResource(type=cls.HW_TYPE, infos=agilex_infos)


@NodeHardwareConfig.register_hardware_config(AgilexRobot.HW_TYPE)
@dataclass
class AgilexConfig(HardwareConfig):
    """Configuration of an Agilex robot-side bridge endpoint."""

    robot_host: str
    """Hostname or IP of the robot-side bridge server."""

    robot_port: int
    """Port of the robot-side bridge server."""

    def __post_init__(self) -> None:
        """Validate configuration schema and value ranges."""
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in Agilex config must be an integer. "
            f"But got {type(self.node_rank)}."
        )
        assert isinstance(self.robot_host, str) and self.robot_host.strip(), (
            "robot_host in Agilex config must be a non-empty string."
        )
        assert isinstance(self.robot_port, int), (
            f"'robot_port' in Agilex config must be an integer. "
            f"But got {type(self.robot_port)}."
        )
        if self.robot_port <= 0 or self.robot_port > 65535:
            raise ValueError(
                f"'robot_port' in Agilex config must be in [1, 65535], "
                f"but got {self.robot_port}."
            )
