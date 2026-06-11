"""Video recording helpers for residual TD3 rollout diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.algorithms.residual_td3.episode_logger import write_json
from scripts.rollout_pi05_fk_ee16_zero_residual import append_frame


@dataclass
class ResidualVideoRecorder:
    """Small recorder that mirrors RLinf video_cfg paths for residual rollouts."""

    enabled: bool
    video_base_dir: Path
    num_save_videos: int | None = None
    frame_stride: int | None = None
    fps: int = 10
    video_source: str = "third_view"

    @classmethod
    def from_args(cls, args: Any) -> "ResidualVideoRecorder":
        """Build a recorder from rollout CLI args."""

        video_base_dir = getattr(args, "video_base_dir", None)
        if video_base_dir is None:
            video_base_dir = Path(args.save_dir) / "video" / "eval"
        frame_stride = getattr(args, "video_temp_subsample", None)
        if frame_stride is not None:
            frame_stride = max(1, int(frame_stride))
        num_save_videos = getattr(args, "num_save_videos", None)
        if num_save_videos is not None:
            num_save_videos = max(0, int(num_save_videos))
        return cls(
            enabled=bool(getattr(args, "save_video", False)),
            video_base_dir=Path(video_base_dir),
            num_save_videos=num_save_videos,
            frame_stride=frame_stride,
            fps=int(getattr(args, "video_fps", 10)),
            video_source=str(getattr(args, "video_source", "third_view")),
        )

    def should_save_episode(self, episode_id: int) -> bool:
        """Return whether an episode should produce a video artifact."""

        if not self.enabled:
            return False
        return self.num_save_videos is None or episode_id < self.num_save_videos

    def append_initial(
        self,
        frames: list[np.ndarray],
        frame_to_env_step: list[int],
        env: Any,
        obs: Any,
    ) -> None:
        """Append reset frame with env step zero."""

        if not self.enabled:
            return
        append_recorder_frame(frames, env, obs, self.video_source)
        frame_to_env_step.append(0)

    def append_chunk(
        self,
        frames: list[np.ndarray],
        frame_to_env_step: list[int],
        env: Any,
        obs_list: list[Any],
        *,
        env_step_start: int,
        video_frame_mode: str,
    ) -> None:
        """Append frames from one executed action chunk."""

        if not self.enabled or not obs_list:
            return
        if self.frame_stride is not None:
            stride = max(1, int(self.frame_stride))
            for offset, step_obs in enumerate(obs_list, start=1):
                if (offset - 1) % stride == 0 or offset == len(obs_list):
                    append_recorder_frame(frames, env, step_obs, self.video_source)
                    frame_to_env_step.append(int(env_step_start + offset))
            return
        if video_frame_mode == "executed_step":
            for offset, step_obs in enumerate(obs_list, start=1):
                append_recorder_frame(frames, env, step_obs, self.video_source)
                frame_to_env_step.append(int(env_step_start + offset))
            return
        append_recorder_frame(frames, env, obs_list[-1], self.video_source)
        frame_to_env_step.append(int(env_step_start + len(obs_list)))

    def save_episode(
        self,
        frames: list[np.ndarray],
        frame_to_env_step: list[int],
        *,
        episode_id: int,
        seed: int,
        success: bool,
        failure_reason: str | None,
        num_steps: int,
        rows: list[dict[str, Any]],
    ) -> Path | None:
        """Save an episode video and sidecar metadata."""

        if not self.should_save_episode(episode_id) or not frames:
            return None
        self.video_base_dir.mkdir(parents=True, exist_ok=True)
        video_path = self.video_base_dir / f"episode_{episode_id:04d}_seed_{seed}.mp4"
        self._write_mp4(frames, video_path)
        meta = {
            "seed": int(seed),
            "episode_id": int(episode_id),
            "success": bool(success),
            "failure_reason": failure_reason,
            "num_steps": int(num_steps),
            "fps": int(self.fps),
            "frame_stride": int(self.frame_stride or 1),
            "frame_to_env_step": [int(step) for step in frame_to_env_step],
            "trigger_steps": trigger_steps(rows),
            "intervention_steps": intervention_steps(rows),
            "action_modes": action_mode_summary(rows),
        }
        write_json(video_path.with_suffix(".video_meta.json"), meta)
        return video_path

    def _write_mp4(self, frames: list[np.ndarray], path: Path) -> None:
        import imageio

        writer = imageio.get_writer(str(path), fps=self.fps)
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()


def trigger_steps(rows: list[dict[str, Any]]) -> list[int]:
    """Return handover/pregrasp trigger env steps from hybrid rows."""

    return [
        int(row["env_step_start"])
        for row in rows
        if row.get("intervention_stage") in {"handover", "pregrasp"}
    ]


def intervention_steps(rows: list[dict[str, Any]]) -> list[int]:
    """Return env steps where EE intervention was executed."""

    return [
        int(row["env_step_start"])
        for row in rows
        if row.get("execution_mode") == "ee16_zero_residual"
    ]


def action_mode_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count rollout action modes in hybrid rows."""

    summary: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("execution_mode", "unknown"))
        summary[mode] = summary.get(mode, 0) + 1
    return summary


def append_recorder_frame(
    frames: list[np.ndarray],
    env: Any,
    obs: Any,
    video_source: str,
) -> None:
    """Append an obs-derived frame, falling back to the legacy capture helper."""

    frame = frame_from_obs(obs, video_source)
    if frame is not None:
        frames.append(frame)
        return
    append_frame(frames, env, obs, video_source)


def frame_from_obs(obs: Any, video_source: str) -> np.ndarray | None:
    """Extract a frame from an observation using RLinf RecordVideo-style keys."""

    if not isinstance(obs, dict):
        return None
    keys = ("main_images", "images", "rgb", "full_image", "main_image")
    if video_source == "main":
        keys = ("main_images", "images", "rgb", "full_image", "main_image")
    for key in keys:
        value = obs.get(key)
        if value is None:
            continue
        arr = as_numpy(value)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            return np.asarray(arr, dtype=np.uint8).copy()
    return None


def as_numpy(value: Any) -> np.ndarray:
    """Convert arrays/tensors to numpy without importing torch at module load."""

    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().numpy()
    return np.asarray(value)
