"""Validate and merge residual replay npz files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_FIELDS = (
    "obs",
    "action",
    "reward",
    "next_obs",
    "done",
    "action_source",
    "seed",
    "episode_id",
    "success",
    "failure_reason",
)
OPTIONAL_FIELDS = (
    "done_episode",
    "done_intervention",
    "env_step",
    "intervention_id",
    "intervention_step_i",
    "gate_score",
    "residual_scale",
    "noise_std",
    "pred_delta_local_xyz",
    "applied_delta_local_xyz",
    "applied_delta_world_xyz",
    "base_ee16",
    "exec_ee16",
    "saturation",
    "has_nan_or_inf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="append", default=[], help="replay.npz path or containing directory.")
    parser.add_argument("--input-dir", default="logs/residual_replay")
    parser.add_argument("--output-dir", default="logs/residual_replay/mainline_4i_merged")
    parser.add_argument("--delta-max", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    replay_paths = resolve_replay_paths(args)
    if not replay_paths:
        raise FileNotFoundError("No replay.npz files found. Pass --replay or check --input-dir.")
    loaded = [load_replay(path, delta_max=args.delta_max) for path in replay_paths]
    merged = merge_replays([item["arrays"] for item in loaded])
    merged_summary = replay_summary(merged, source_summaries=[item["summary"] for item in loaded], delta_max=args.delta_max)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "replay.npz", **merged)
    write_json(output_dir / "summary.json", merged_summary)
    print(f"MERGED REPLAY: {output_dir / 'replay.npz'}")
    print(f"SUMMARY: {output_dir / 'summary.json'}")
    print(json.dumps(merged_summary["merged"], indent=2, sort_keys=True))
    return 0


def resolve_replay_paths(args: argparse.Namespace) -> list[Path]:
    if args.replay:
        return [resolve_replay_path(Path(path)) for path in args.replay]
    root = Path(args.input_dir)
    candidates = []
    for path in sorted(root.rglob("replay.npz")):
        if "mainline_4i_merged" in path.parts:
            continue
        candidates.append(path)
    return candidates


def resolve_replay_path(path: Path) -> Path:
    return path if path.name == "replay.npz" else path / "replay.npz"


def load_replay(path: Path, *, delta_max: float) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    validate_replay(arrays, path=path)
    return {"path": str(path), "arrays": arrays, "summary": replay_file_summary(arrays, path, delta_max=delta_max)}


def validate_replay(arrays: dict[str, np.ndarray], *, path: Path) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in arrays]
    if missing:
        raise KeyError(f"{path} missing replay fields: {missing}")
    obs = arrays["obs"]
    action = arrays["action"]
    reward = arrays["reward"]
    next_obs = arrays["next_obs"]
    done = arrays["done"]
    if obs.ndim != 2 or obs.shape[1] != 21:
        raise ValueError(f"{path}: obs must have shape [N,21], got {obs.shape}")
    if action.ndim != 2 or action.shape[1] != 3:
        raise ValueError(f"{path}: action must have shape [N,3], got {action.shape}")
    if next_obs.shape != obs.shape:
        raise ValueError(f"{path}: next_obs shape {next_obs.shape} does not match obs {obs.shape}")
    n = obs.shape[0]
    for field, array in arrays.items():
        if array.shape and array.shape[0] != n:
            raise ValueError(f"{path}: field {field} first dim {array.shape[0]} != {n}")
    if reward.shape != (n,) or done.shape != (n,):
        raise ValueError(f"{path}: reward/done must have shape [N]")


def merge_replays(replays: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = sorted(set().union(*(replay.keys() for replay in replays)))
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        arrays = [ensure_field(replay, key) for replay in replays]
        merged[key] = np.concatenate(arrays, axis=0)
    return merged


def ensure_field(replay: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key in replay:
        return replay[key]
    n = int(replay["obs"].shape[0])
    if key in OPTIONAL_FIELDS:
        return default_optional_field(key, n)
    raise KeyError(f"Cannot merge replay missing required field {key}.")


def default_optional_field(key: str, n: int) -> np.ndarray:
    shapes: dict[str, tuple[type, tuple[int, ...]]] = {
        "done_episode": (np.bool_, ()),
        "done_intervention": (np.bool_, ()),
        "env_step": (np.int64, ()),
        "intervention_id": (np.int64, ()),
        "intervention_step_i": (np.int64, ()),
        "gate_score": (np.float32, ()),
        "residual_scale": (np.float32, ()),
        "noise_std": (np.float32, ()),
        "pred_delta_local_xyz": (np.float32, (3,)),
        "applied_delta_local_xyz": (np.float32, (3,)),
        "applied_delta_world_xyz": (np.float32, (3,)),
        "base_ee16": (np.float32, (16,)),
        "exec_ee16": (np.float32, (16,)),
        "saturation": (np.int64, ()),
        "has_nan_or_inf": (np.int64, ()),
    }
    dtype, shape = shapes[key]
    return np.zeros((n, *shape), dtype=dtype)


def replay_file_summary(arrays: dict[str, np.ndarray], path: Path, *, delta_max: float) -> dict[str, Any]:
    summary = replay_stats(arrays, delta_max=delta_max)
    summary["path"] = str(path)
    summary["action_source"] = sorted(str(value) for value in np.unique(arrays["action_source"]))
    return summary


def replay_summary(
    arrays: dict[str, np.ndarray],
    *,
    source_summaries: list[dict[str, Any]],
    delta_max: float,
) -> dict[str, Any]:
    return {
        "sources": source_summaries,
        "merged": replay_stats(arrays, delta_max=delta_max),
        "fields": sorted(arrays),
    }


def replay_stats(arrays: dict[str, np.ndarray], *, delta_max: float) -> dict[str, Any]:
    reward = arrays["reward"].astype(np.float32)
    action = arrays["action"].astype(np.float32)
    action_norm = np.linalg.norm(action, axis=1) if action.size else np.asarray([], dtype=np.float32)
    finite_arrays = [arrays["obs"], arrays["action"], arrays["reward"], arrays["next_obs"]]
    nan_inf_count = int(sum(np.size(arr) - np.isfinite(arr).sum() for arr in finite_arrays))
    if "has_nan_or_inf" in arrays:
        nan_inf_count += int(np.asarray(arrays["has_nan_or_inf"], dtype=np.int64).sum())
    saturation = np.asarray(arrays.get("saturation", np.zeros(action.shape[0])), dtype=np.float32)
    saturation_ratio = float(saturation.mean()) if saturation.size else 0.0
    if action.size:
        saturation_ratio = max(saturation_ratio, float((np.abs(action) >= delta_max - 1e-7).any(axis=1).mean()))
    episodes = unique_episodes(arrays)
    success_by_episode = episode_success_rate(arrays)
    return {
        "transitions": int(arrays["obs"].shape[0]),
        "episodes": int(episodes),
        "obs_shape": list(arrays["obs"].shape),
        "action_shape": list(arrays["action"].shape),
        "reward_mean": float(reward.mean()) if reward.size else 0.0,
        "reward_std": float(reward.std()) if reward.size else 0.0,
        "reward_min": float(reward.min()) if reward.size else 0.0,
        "reward_max": float(reward.max()) if reward.size else 0.0,
        "action_norm_mean": float(action_norm.mean()) if action_norm.size else 0.0,
        "action_norm_std": float(action_norm.std()) if action_norm.size else 0.0,
        "action_norm_max": float(action_norm.max()) if action_norm.size else 0.0,
        "action_norm_p95": float(np.percentile(action_norm, 95)) if action_norm.size else 0.0,
        "done_ratio": float(arrays["done"].mean()) if arrays["done"].size else 0.0,
        "success_rate": success_by_episode,
        "nan_inf_count": nan_inf_count,
        "saturation_ratio": saturation_ratio,
    }


def unique_episodes(arrays: dict[str, np.ndarray]) -> int:
    return len(
        {
            (str(source), int(seed), int(episode_id))
            for source, seed, episode_id in zip(
                arrays["action_source"],
                arrays["seed"],
                arrays["episode_id"],
                strict=True,
            )
        }
    )


def episode_success_rate(arrays: dict[str, np.ndarray]) -> float:
    if arrays["success"].size == 0:
        return 0.0
    by_episode: dict[tuple[str, int, int], bool] = {}
    for source, seed, episode_id, success in zip(
        arrays["action_source"],
        arrays["seed"],
        arrays["episode_id"],
        arrays["success"],
        strict=True,
    ):
        by_episode[(str(source), int(seed), int(episode_id))] = bool(success)
    return float(np.mean(list(by_episode.values()))) if by_episode else 0.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
