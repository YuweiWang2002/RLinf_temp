import numpy as np
import pytest
import torch

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv


class _Cfg:
    seed = 0
    auto_reset = False
    use_rel_reward = False
    ignore_terminations = False
    group_size = 1
    use_fixed_reset_state_ids = False
    use_custom_reward = False
    max_episode_steps = 100
    is_eval = False
    video_cfg = None

    def __init__(self, robotwin_action_mode: str = "qpos14") -> None:
        self.robotwin_action_mode = robotwin_action_mode

    def get(self, key, default=None):
        return getattr(self, key, default)


class _TaskConfig:
    task_name = "handover_block"


_Cfg.task_config = _TaskConfig()


class _FakeTask:
    step_lim = 10

    def __init__(self) -> None:
        self.eval_success = False
        self.take_action_cnt = 0
        self.calls = []

    def take_action(self, action, action_type="qpos"):
        self.calls.append((np.asarray(action).copy(), action_type))
        self.take_action_cnt += 1


class _FakeSubEnv:
    def __init__(self, task=None) -> None:
        if task is not None:
            self.task = task


class _FakeVenv:
    def __init__(self, num_envs: int, expose_tasks: bool = True) -> None:
        self.step_calls = []
        self.tasks = [_FakeTask() for _ in range(num_envs)]
        self.envs = [
            _FakeSubEnv(task if expose_tasks else None) for task in self.tasks
        ]

    def step(self, actions):
        self.step_calls.append(np.asarray(actions).copy())
        return (
            [{} for _ in self.tasks],
            [0.0 for _ in self.tasks],
            [False for _ in self.tasks],
            [False for _ in self.tasks],
            [{} for _ in self.tasks],
        )

    def get_obs(self):
        return [{} for _ in self.tasks]


def _env(robotwin_action_mode: str = "qpos14", expose_tasks: bool = True):
    env = object.__new__(RoboTwinEnv)
    env.num_envs = 2
    env.robotwin_action_mode = robotwin_action_mode
    env.venv = _FakeVenv(env.num_envs, expose_tasks=expose_tasks)
    env.cfg = _Cfg(robotwin_action_mode)
    env.auto_reset = False
    env.use_rel_reward = False
    env.ignore_terminations = False
    env.use_custom_reward = False
    env.record_metrics = True
    device = env.device
    env.prev_step_reward = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env.success_once = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env.fail_once = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env.returns = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._elapsed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._extract_obs_image = lambda raw_obs: {"raw_obs": raw_obs}
    return env


def test_qpos14_default_path_uses_venv_step_and_not_ee16_dispatch(monkeypatch):
    env = _env("qpos14")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_step_ee16_action should not be called")

    monkeypatch.setattr(env, "_step_ee16_action", fail_if_called)

    env.step(np.zeros((2, 14), dtype=np.float32))

    assert len(env.venv.step_calls) == 1
    assert env.venv.tasks[0].calls == []


def test_ee16_step_shape_validation_accepts_16_and_rejects_14():
    env = _env("ee16")

    env._validate_ee16_action(np.zeros((2, 16), dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        env._validate_ee16_action(np.zeros((2, 14), dtype=np.float32))


def test_ee16_chunk_shape_validation_accepts_16_and_rejects_14():
    env = _env("ee16")

    env._validate_ee16_chunk_actions(np.zeros((2, 5, 16), dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        env._validate_ee16_chunk_actions(np.zeros((2, 5, 14), dtype=np.float32))


def test_ee16_path_calls_task_take_action_with_ee_action_type():
    env = _env("ee16")
    actions = np.arange(2 * 16, dtype=np.float32).reshape(2, 16)
    actions[:, 3:7] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    actions[:, 11:15] = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    env.step(actions)

    assert len(env.venv.step_calls) == 0
    for env_id, task in enumerate(env.venv.tasks):
        assert len(task.calls) == 1
        action, action_type = task.calls[0]
        assert action.shape == (16,)
        assert action_type == "ee"
        expected = actions[env_id].copy()
        expected[3:7] = actions[env_id, [6, 3, 4, 5]]
        expected[11:15] = actions[env_id, [14, 11, 12, 13]]
        np.testing.assert_allclose(action, expected)


def test_ee16_mode_requires_task_access_and_never_silent_falls_back_to_qpos():
    env = _env("ee16", expose_tasks=False)

    with pytest.raises(NotImplementedError, match="task.take_action"):
        env.step(np.zeros((2, 16), dtype=np.float32))

    assert env.venv.step_calls == []
