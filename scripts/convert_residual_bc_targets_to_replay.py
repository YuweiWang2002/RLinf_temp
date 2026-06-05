"""Convert residual BC target npz files into TD3-compatible replay npz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.algorithms.residual_td3.residual_bc_training import (
    FEATURE_DIMS,
    FEATURE_NAMES,
)
from scripts.merge_residual_replay import (
    load_replay,
    replay_file_summary,
    validate_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hard-seeds-json", default=None)
    parser.add_argument("--action-source", default="hard_expert")
    parser.add_argument("--reward-value", type=float, default=1.0)
    parser.add_argument("--delta-max", type=float, default=0.05)
    parser.add_argument("--target-horizon-k", type=int, default=1)
    parser.add_argument("--max-delta-local-xyz", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays = convert_bc_targets_to_replay(
        bc_targets=Path(args.bc_targets),
        action_source=args.action_source,
        reward_value=args.reward_value,
        target_horizon_k=args.target_horizon_k,
        max_delta_local_xyz=args.max_delta_local_xyz,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "replay.npz"
    np.savez_compressed(replay_path, **arrays)
    validate_replay(arrays, path=replay_path)
    hard_seed_info = hard_seed_membership(arrays["seed"], Path(args.hard_seeds_json) if args.hard_seeds_json else None)
    summary = {
        "source_bc_targets": str(Path(args.bc_targets)),
        "replay_path": str(replay_path),
        "conversion": {
            "action_source": args.action_source,
            "reward_mode": "success_constant",
            "reward_value": float(args.reward_value),
            "next_obs_mode": "next_sample_within_episode",
            "target_horizon_k": int(args.target_horizon_k),
            "max_delta_local_xyz": float(args.max_delta_local_xyz),
        },
        "quality": replay_file_summary(arrays, replay_path, delta_max=args.delta_max),
        "hard_seed_membership": hard_seed_info,
        "all_successful_expert_replay": bool(np.asarray(arrays["success"], dtype=bool).all()),
        "fields": sorted(arrays),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def convert_bc_targets_to_replay(
    *,
    bc_targets: Path,
    action_source: str,
    reward_value: float,
    target_horizon_k: int,
    max_delta_local_xyz: float,
) -> dict[str, np.ndarray]:
    del target_horizon_k
    arrays = load_bc_target_arrays(bc_targets)
    obs = build_obs_vectors(arrays).astype(np.float32)
    action = np.clip(
        np.asarray(arrays["target_delta_local_xyz"][:, 0, :], dtype=np.float32),
        -float(max_delta_local_xyz),
        float(max_delta_local_xyz),
    )
    episode_id = np.asarray(arrays["episode_index"], dtype=np.int64)
    frame_index = np.asarray(arrays["frame_index"], dtype=np.int64)
    seed = np.asarray(arrays.get("seed", np.full(obs.shape[0], -1)), dtype=np.int64)
    next_obs, done = build_next_obs_and_done(obs, episode_id=episode_id, frame_index=frame_index)
    n = int(obs.shape[0])
    base_ee16 = np.asarray(arrays["base_endpose16"][:, 0, :], dtype=np.float32)
    exec_ee16 = np.asarray(arrays["expert_endpose16"][:, 0, :], dtype=np.float32)
    return {
        "obs": obs,
        "action": action,
        "reward": np.full(n, float(reward_value), dtype=np.float32),
        "next_obs": next_obs.astype(np.float32),
        "done": done.astype(np.bool_),
        "done_episode": done.astype(np.bool_),
        "done_intervention": done.astype(np.bool_),
        "action_source": np.asarray([action_source] * n, dtype="U32"),
        "seed": seed,
        "episode_id": episode_id,
        "success": np.ones(n, dtype=np.bool_),
        "failure_reason": np.asarray([""] * n, dtype="U96"),
        "env_step": frame_index,
        "intervention_id": episode_id,
        "intervention_step_i": frame_index,
        "gate_score": np.asarray(arrays["obs_gate_label"][:, 0], dtype=np.float32).reshape(-1),
        "residual_scale": np.ones(n, dtype=np.float32),
        "noise_std": np.zeros(n, dtype=np.float32),
        "pred_delta_local_xyz": action.copy(),
        "applied_delta_local_xyz": action.copy(),
        "applied_delta_world_xyz": np.asarray(arrays["target_delta_world_xyz"][:, 0, :], dtype=np.float32),
        "base_ee16": base_ee16,
        "exec_ee16": exec_ee16,
        "saturation": np.zeros(n, dtype=np.int64),
        "has_nan_or_inf": has_nan_or_inf(obs, action, next_obs),
    }


def load_bc_target_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    required = (
        "target_delta_local_xyz",
        "target_delta_world_xyz",
        "base_endpose16",
        "expert_endpose16",
        "episode_index",
        "frame_index",
        "gate_seq",
        "obs_base_left_xyz",
        "obs_base_left_quat_wxyz",
        "obs_base_right_xyz",
        "obs_base_right_quat_wxyz",
        "obs_relative_right_xyz_in_left_frame",
        "obs_left_gripper",
        "obs_right_gripper",
        "obs_gate_label",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"{path} missing BC target arrays: {missing}")
    return arrays


def build_obs_vectors(arrays: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for name in FEATURE_NAMES:
        if name == "gate_positive_fraction":
            part = (np.asarray(arrays["gate_seq"]) > 0.0).mean(axis=1, keepdims=True).astype(np.float32)
        else:
            part = first_offset(
                arrays[f"obs_{name}"],
                scalar=name in {"left_gripper", "right_gripper", "gate_label", "horizon_offset"},
            )
        expected = FEATURE_DIMS[name]
        if part.shape[1] != expected:
            raise ValueError(f"obs_{name} must have {expected} columns after packing, got {part.shape}.")
        parts.append(part.astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def first_offset(value: np.ndarray, *, scalar: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        if scalar:
            return array[:, :1]
        return array[:, :1] if array.shape[1] == 1 else array
    return array[:, 0].reshape(array.shape[0], -1)


def build_next_obs_and_done(
    obs: np.ndarray,
    *,
    episode_id: np.ndarray,
    frame_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(obs.shape[0])
    next_obs = obs.copy()
    done = np.ones(n, dtype=np.bool_)
    for idx in range(n - 1):
        if int(episode_id[idx + 1]) == int(episode_id[idx]) and int(frame_index[idx + 1]) == int(frame_index[idx]) + 1:
            next_obs[idx] = obs[idx + 1]
            done[idx] = False
    return next_obs, done


def has_nan_or_inf(*arrays: np.ndarray) -> np.ndarray:
    flags = np.zeros(arrays[0].shape[0], dtype=np.int64)
    for array in arrays:
        flat = np.asarray(array).reshape(array.shape[0], -1)
        flags |= (~np.isfinite(flat).all(axis=1)).astype(np.int64)
    return flags


def hard_seed_membership(seeds: np.ndarray, path: Path | None) -> dict[str, Any]:
    unique = sorted(int(seed) for seed in np.unique(seeds) if int(seed) >= 0)
    if path is None or not path.exists():
        return {"unique_seeds": unique, "hard_seeds_json": str(path) if path else None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_fail = {int(seed) for seed in payload.get("all_fail", [])}
    candidates = {int(seed) for seed in payload.get("candidate_hard_eval_seeds", [])}
    for row in payload.get("rows", []):
        seed = int(row["seed"])
        source_group = str(row.get("source_group", ""))
        if source_group == "all_fail":
            all_fail.add(seed)
        if source_group in {"all_fail", "candidate_hard_eval_seeds", "candidate_hard"}:
            candidates.add(seed)
    for seed in payload.get("selected", []):
        candidates.add(int(seed))
    present = set(unique)
    return {
        "hard_seeds_json": str(path),
        "unique_seeds": unique,
        "num_unique_seeds": len(unique),
        "num_all_fail_present": len(present & all_fail),
        "num_candidate_hard_present": len(present & candidates),
        "all_seeds_are_all_fail": bool(present) and present <= all_fail,
    }


def load_converted_replay(path: Path, *, delta_max: float) -> dict[str, Any]:
    return load_replay(path, delta_max=delta_max)


if __name__ == "__main__":
    raise SystemExit(main())
