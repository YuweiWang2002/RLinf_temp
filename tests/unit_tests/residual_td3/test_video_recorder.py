import json
from argparse import Namespace

import numpy as np

from rlinf.algorithms.residual_td3 import video_recorder as recorder_module
from rlinf.algorithms.residual_td3.video_recorder import ResidualVideoRecorder


def test_residual_video_recorder_writes_mp4_and_metadata(tmp_path, monkeypatch):
    def fake_append_frame(frames, env, obs, video_source):
        frames.append(np.full((8, 8, 3), int(obs), dtype=np.uint8))

    monkeypatch.setattr(recorder_module, "append_frame", fake_append_frame)
    args = Namespace(
        save_video=True,
        save_dir=str(tmp_path),
        video_base_dir=None,
        num_save_videos=None,
        video_temp_subsample=2,
        video_fps=5,
        video_source="third_view",
    )
    recorder = ResidualVideoRecorder.from_args(args)
    frames = []
    frame_to_env_step = []

    recorder.append_initial(frames, frame_to_env_step, env=object(), obs=0)
    recorder.append_chunk(
        frames,
        frame_to_env_step,
        env=object(),
        obs_list=[1, 2, 3, 4, 5],
        env_step_start=100,
        video_frame_mode="replan",
    )
    video_path = recorder.save_episode(
        frames,
        frame_to_env_step,
        episode_id=0,
        seed=100100000,
        success=False,
        failure_reason="timeout",
        num_steps=105,
        rows=[
            {
                "env_step_start": 100,
                "execution_mode": "ee16_zero_residual",
                "intervention_stage": "handover",
            },
            {
                "env_step_start": 105,
                "execution_mode": "qpos14",
                "intervention_stage": "none",
            },
        ],
    )

    assert video_path is not None
    assert video_path == tmp_path / "video" / "eval" / "episode_0000_seed_100100000.mp4"
    assert video_path.exists()
    meta = json.loads(video_path.with_suffix(".video_meta.json").read_text())
    assert meta["seed"] == 100100000
    assert meta["success"] is False
    assert meta["frame_stride"] == 2
    assert meta["frame_to_env_step"] == [0, 101, 103, 105]
    assert meta["trigger_steps"] == [100]
    assert meta["intervention_steps"] == [100]
    assert meta["action_modes"] == {"ee16_zero_residual": 1, "qpos14": 1}


def test_residual_video_recorder_respects_num_save_videos(tmp_path):
    args = Namespace(
        save_video=True,
        save_dir=str(tmp_path),
        video_base_dir=None,
        num_save_videos=0,
        video_temp_subsample=None,
        video_fps=5,
        video_source="third_view",
    )
    recorder = ResidualVideoRecorder.from_args(args)

    assert recorder.save_episode(
        [np.zeros((8, 8, 3), dtype=np.uint8)],
        [0],
        episode_id=0,
        seed=1,
        success=True,
        failure_reason=None,
        num_steps=1,
        rows=[],
    ) is None
