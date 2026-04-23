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

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.agilex.agilex_env import AgilexEnv, AgilexRobotConfig
from rlinf.envs.realworld.agilex.tasks.tideupdesk import TideupdeskEnv as TideupdeskEnv


def _apply_agilex_prechecks(env: gym.Env, env_cfg: Mapping[str, Any]) -> gym.Env:
    """Validate Agilex-specific constraints before rollout starts."""
    if env_cfg.get("no_gripper", False):
        raise NotImplementedError(
            "Agilex realworld does not support no_gripper=True: "
            "the policy action space is fixed to 14D dual-arm joint commands."
        )
    if env_cfg.get("use_relative_frame", False):
        raise NotImplementedError(
            "Agilex realworld does not support use_relative_frame=True: "
            "observations do not expose tcp_pose for frame transforms."
        )
    if env_cfg.get("use_quat2euler", False):
        raise NotImplementedError(
            "Agilex realworld does not support use_quat2euler=True: "
            "observations are joint-space without quaternion tcp_pose."
        )
    main_image_key = env_cfg.get("main_image_key", "cam_high")
    frame_spaces = env.observation_space["frames"].spaces
    if main_image_key not in frame_spaces:
        raise KeyError(
            f"main_image_key {main_image_key!r} is not in Agilex frame keys "
            f"{list(frame_spaces.keys())}."
        )
    return env


def create_agilex_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = AgilexEnv(
        config=AgilexRobotConfig(**override_cfg),
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return _apply_agilex_prechecks(env, env_cfg)


def create_tideupdesk_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = TideupdeskEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return _apply_agilex_prechecks(env, env_cfg)


register(
    id="AgilexRealWorldEnv-v1",
    entry_point="rlinf.envs.realworld.agilex.tasks:create_agilex_env",
)

register(
    id="AgilexTidyUpTheDeskEnv-v1",
    entry_point="rlinf.envs.realworld.agilex.tasks:create_tideupdesk_env",
)

register(
    id="AgilexTideupdeskEnv-v1",
    entry_point="rlinf.envs.realworld.agilex.tasks:create_tideupdesk_env",
)
