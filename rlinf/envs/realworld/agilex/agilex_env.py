from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.agilex.agilex_robot_state import AgilexRobotState
from rlinf.envs.realworld.agilex.policy_side_bridge import AgilexPolicySideBridge
from rlinf.scheduler import AgilexHWInfo, WorkerInfo


AGILEX_ACTION_DIM = 14


@dataclass
class AgilexRobotConfig:
    is_dummy: bool = False
    robot_host: Optional[str] = None
    robot_port: Optional[int] = None
    require_manual_reset: bool = False
    task_description: str = "Organize the shoes"
    use_dense_reward: bool = False
    max_num_steps: int = 100
    image_height: int = 480
    image_width: int = 640
    startup_check_timeout_s: float | None = 120.0
    startup_check_poll_interval_s: float = 1.0
    reset_timeout_s: float | None = None
    start_timeout_s: float | None = None
    observation_timeout_s: float = 10.0
    observation_poll_interval_s: float = 0.1


class AgilexEnv(gym.Env):
    """Joint-space realworld env backed by the Agilex Socket.IO bridge."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: AgilexRobotConfig,
        worker_info: WorkerInfo | None = None,
        hardware_info: AgilexHWInfo | None = None,
        env_idx: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.worker_info = worker_info
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._agilex_state = AgilexRobotState()
        if not self.config.is_dummy:
            self._setup_hardware()
            self._client: AgilexPolicySideBridge | None = AgilexPolicySideBridge(
                host=self.config.robot_host,
                port=self.config.robot_port,
            )
        else:
            self._client = None
        self._dummy_joint_state = np.zeros((AGILEX_ACTION_DIM,), dtype=np.float32)
        self._num_steps = 0
        self._operator_confirmed = False
        self._episode_started = False
        self._connection_generation = (
            0 if self._client is None else self._client.connection_generation
        )

        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(AGILEX_ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_state": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(AGILEX_ACTION_DIM,),
                            dtype=np.float32,
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        "cam_high": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=(config.image_height, config.image_width, 3),
                            dtype=np.uint8,
                        ),
                        "cam_left_wrist": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=(config.image_height, config.image_width, 3),
                            dtype=np.uint8,
                        ),
                        "cam_right_wrist": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=(config.image_height, config.image_width, 3),
                            dtype=np.uint8,
                        ),
                    }
                ),
            }
        )

    @property
    def task_description(self) -> str:
        return self.config.task_description

    def _setup_hardware(self) -> None:
        """Resolve host/port with priority: override_cfg > hardware_info > error."""
        assert self.env_idx >= 0, "env_idx must be set for AgilexEnv."
        assert isinstance(self.hardware_info, AgilexHWInfo), (
            f"hardware_info must be AgilexHWInfo, but got {type(self.hardware_info)}."
        )
        if self.config.robot_host is None or self.config.robot_host == "":
            self.config.robot_host = self.hardware_info.config.robot_host
        if self.config.robot_port is None:
            self.config.robot_port = self.hardware_info.config.robot_port
        if self.config.robot_host is None or self.config.robot_port is None:
            raise ValueError(
                "AgilexEnv requires robot_host and robot_port when is_dummy=False."
            )
        if self.config.robot_port <= 0:
            raise ValueError(
                f"robot_port must be a positive integer, but got {self.config.robot_port}."
            )

    def _run_startup_check_once(self) -> None:
        assert self._client is not None, "Robot bridge client must be initialized."
        response = self._client.wait_startup_check(
            timeout=self.config.startup_check_timeout_s,
            poll_interval=self.config.startup_check_poll_interval_s,
        )
        status = str(response.get("status", ""))
        if status != "success":
            raise RuntimeError(
                "Robot bridge startup check failed: "
                f"status={status!r}, response={response!r}."
            )

    def _sync_bridge_session_state(self) -> None:
        assert self._client is not None, "Robot bridge client must be initialized."
        generation = self._client.connection_generation
        if generation == self._connection_generation:
            return

        self._connection_generation = generation
        self._operator_confirmed = False
        self._episode_started = False

    def _ensure_startup_confirmed(self) -> None:
        assert self._client is not None, "Robot bridge client must be initialized."
        self._sync_bridge_session_state()
        if self._operator_confirmed:
            return

        self._run_startup_check_once()
        self._operator_confirmed = True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        super().close()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._num_steps = 0
        self._episode_started = False
        if self.config.is_dummy:
            self._dummy_joint_state.fill(0.0)
            self._agilex_state.joint_state = self._dummy_joint_state.copy()
            self._agilex_state.success = False
            return self._get_dummy_observation(), {"reset_success": True}

        assert self._client is not None, "Robot bridge client must be initialized."
        self._ensure_startup_confirmed()
        response = self._client.reset(
            manual_reset=self.config.require_manual_reset,
            timeout=self.config.reset_timeout_s,
        )
        if response is None:
            raise RuntimeError("Robot bridge reset failed or returned no response.")
        status = str(response.get("status", ""))
        if status != "success":
            state = response.get("state", "unknown")
            reason = response.get("reason", "unknown")
            raise RuntimeError(
                "Robot bridge reset failed: "
                f"status={status!r}, state={state!r}, reason={reason!r}."
            )
        observation = self._wait_for_observation()
        return observation, {"reset_success": True}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (AGILEX_ACTION_DIM,):
            raise ValueError(
                f"Agilex action must have shape ({AGILEX_ACTION_DIM},), "
                f"but got {action.shape}."
            )
        if self.config.is_dummy:
            self._num_steps += 1
            self._dummy_joint_state = action.copy()
            self._agilex_state.joint_state = self._dummy_joint_state.copy()
            self._agilex_state.success = False
            truncated = self._num_steps >= self.config.max_num_steps
            return self._get_dummy_observation(), 0.0, False, truncated, {
                "task_success": False
            }

        assert self._client is not None, "Robot bridge client must be initialized."
        self._ensure_startup_confirmed()
        self._start_episode_if_needed()
        self._client.publish_joint_commands(
            left_joints=action[:7].tolist(),
            right_joints=action[7:14].tolist(),
        )

        self._num_steps += 1
        raw_observation = self._wait_for_observation(return_raw=True)
        observation = self._convert_observation(raw_observation)
        reward, terminated = self._compute_reward(raw_observation)
        truncated = self._num_steps >= self.config.max_num_steps
        info = {
            "task_success": bool(raw_observation.get("success", False)),
        }
        return observation, reward, terminated, truncated, info

    def _start_episode_if_needed(self) -> None:
        if self._episode_started:
            return

        assert self._client is not None, "Robot bridge client must be initialized."
        response = self._client.start_episode(
            timeout=self.config.start_timeout_s,
        )
        if response is None:
            raise RuntimeError("Robot bridge start failed or returned no response.")

        status = str(response.get("status", ""))
        if status != "success":
            state = response.get("state", "unknown")
            reason = response.get("reason", "unknown")
            raise RuntimeError(
                "Robot bridge start failed: "
                f"status={status!r}, state={state!r}, reason={reason!r}."
            )
        self._episode_started = True

    def _wait_for_observation(
        self, *, return_raw: bool = False
    ) -> dict[str, dict[str, np.ndarray]]:
        assert self._client is not None, "Robot bridge client must be initialized."
        deadline = time.time() + self.config.observation_timeout_s
        while time.time() < deadline:
            observation = self._client.get_observation(self.config.task_description)
            if observation is not None:
                return observation if return_raw else self._convert_observation(observation)
            time.sleep(self.config.observation_poll_interval_s)
        raise TimeoutError("Timed out waiting for an observation from the robot bridge.")

    def _get_dummy_observation(self) -> dict[str, dict[str, np.ndarray]]:
        zeros_image = np.zeros(
            (self.config.image_height, self.config.image_width, 3), dtype=np.uint8
        )
        return {
            "state": collections.OrderedDict(
                {"joint_state": self._dummy_joint_state.astype(np.float32)}
            ),
            "frames": collections.OrderedDict(
                {
                    "cam_high": zeros_image.copy(),
                    "cam_left_wrist": zeros_image.copy(),
                    "cam_right_wrist": zeros_image.copy(),
                }
            ),
        }

    def _convert_observation(
        self, raw_observation: collections.OrderedDict
    ) -> dict[str, dict[str, np.ndarray]]:
        joint_state = np.asarray(raw_observation["state"], dtype=np.float32)
        self._agilex_state.joint_state = joint_state.copy()
        self._agilex_state.success = bool(raw_observation.get("success", False))
        return {
            "state": collections.OrderedDict(
                {
                    "joint_state": joint_state
                }
            ),
            "frames": collections.OrderedDict(
                (
                    camera_name,
                    np.asarray(image, dtype=np.uint8),
                )
                for camera_name, image in raw_observation["images"].items()
            ),
        }

    def _compute_reward(self, raw_observation: collections.OrderedDict) -> tuple[float, bool]:
        success = bool(raw_observation.get("success", False))

        if self.config.use_dense_reward:
            raise NotImplementedError(
                "Agilex realworld currently only supports sparse reward. "
                "The robot-side bridge returns only a success flag and does not provide "
                "a dense shaping signal."
            )

        return (1.0 if success else 0.0), success
