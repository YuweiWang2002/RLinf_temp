"""Convert successful RoboTwin expert HDF5 demos to residual BC targets.

The input is the successful expert dataset produced by
``collect_hard_seed_expert_demos.py``.  This script runs the frozen pi05 policy
on each expert observation, compares pi05 base action chunks with the expert
qpos14 chunks, and saves the existing residual BC target format.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from rlinf.algorithms.residual_td3.fk_bridge import AlohaFKBridge, AlohaFKBridgeConfig
from rlinf.algorithms.residual_td3.residual_bc_dataset import (
    ResidualBCTargetBuilder,
    ResidualBCTargetConfig,
)
from scripts.build_residual_bc_targets import (
    axis_stats,
    parse_offsets,
    save_histogram_csv,
    save_summary_parquet,
    scalar_stats,
    stack_samples,
)
from scripts.extract_pi05_action_cache import (
    concat_or_empty,
    episode_cache_path,
    validate_cache_schema,
    write_episode_cache_atomic,
)
from scripts.extract_pi05_features_for_gate import (
    DEFAULT_PROMPT,
    decode_image,
    load_pi05_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-demo-dir", required=True)
    parser.add_argument("--expert-summary", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pi05-config", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--pi05-checkpoint", required=True)
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--residual-horizon-k", type=int, default=50)
    parser.add_argument("--target-horizon-offsets", default="0,1,2")
    parser.add_argument("--max-delta-local-xyz", type=float, default=0.01)
    parser.add_argument("--gate-mode", choices=("all_one", "all_zero"), default="all_one")
    parser.add_argument("--gate-positive-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-positive-frames", type=int, default=1)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--save-parquet", action="store_true")
    parser.add_argument("--save-pi05-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every-episodes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards).")
    if args.chunk_len <= 0:
        raise ValueError("--chunk-len must be positive.")

    demo_data_dir = resolve_demo_data_dir(Path(args.expert_demo_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = infer_expert_summary_path(demo_data_dir, args.expert_summary)
    metadata_by_episode = load_success_metadata(summary_path)

    episode_paths = select_episode_paths(
        sorted(demo_data_dir.glob("episode*.hdf5"), key=episode_sort_key),
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        episode_limit=args.episode_limit,
    )
    if not episode_paths:
        raise FileNotFoundError(f"No selected episode*.hdf5 files found under {demo_data_dir}.")

    model = load_pi05_model(
        checkpoint=args.pi05_checkpoint,
        config_name=args.pi05_config,
        device=args.device,
        norm_stats_path=args.norm_stats_path,
        model_num_action_chunks=args.chunk_len,
    )
    cfg = ResidualBCTargetConfig(
        chunk_len=args.chunk_len,
        residual_horizon_k=args.residual_horizon_k,
        target_horizon_offsets=parse_offsets(args.target_horizon_offsets),
        max_delta_local_xyz=args.max_delta_local_xyz,
        gate_positive_only=args.gate_positive_only,
        min_positive_frames=args.min_positive_frames,
        base_action_source="pi05_cached_base",
    )
    builder = ResidualBCTargetBuilder(cfg, AlohaFKBridge(AlohaFKBridgeConfig(device=args.device)))

    samples: list[dict[str, object]] = []
    per_episode: list[dict[str, Any]] = []
    counters = {
        "total_candidate_frames": 0,
        "gate_positive_candidate_count": 0,
        "total_kept_samples": 0,
    }
    cache_dir = out_dir / "pi05_action_cache"
    failed: list[dict[str, Any]] = []
    for episode_count, path in enumerate(episode_paths, start=1):
        episode_id = episode_id_from_path(path)
        meta = metadata_by_episode.get(episode_id, {})
        seed = optional_int(meta.get("seed"))
        try:
            episode = read_expert_hdf5_episode(
                path,
                episode_id=episode_id,
                seed=seed,
                default_prompt=args.default_prompt,
            )
            payload = extract_episode_cache(
                model=model,
                episode=episode,
                chunk_size=args.chunk_len,
                batch_size=args.batch_size,
                device=args.device,
                gate_mode=args.gate_mode,
            )
            validate_cache_schema(payload, chunk_size=args.chunk_len)
            if args.save_pi05_cache:
                cache_path = episode_cache_path(cache_dir, episode_id)
                write_episode_cache_atomic(cache_path, payload)
            episode_samples, episode_counters = build_episode_samples(
                builder=builder,
                episode=episode,
                cache=payload,
            )
            for sample in episode_samples:
                sample["metadata"]["seed"] = seed
                sample["metadata"]["source_hdf5"] = str(path)
            samples.extend(episode_samples)
            merge_counters(counters, episode_counters)
            per_episode.append(
                {
                    "episode_index": episode_id,
                    "seed": seed,
                    "source_hdf5": str(path),
                    "num_frames": int(episode["action"].shape[0]),
                    "num_valid_chunks": int(payload["base_action_chunk"].shape[0]),
                    "num_samples": len(episode_samples),
                    "pi05_cache_path": str(episode_cache_path(cache_dir, episode_id))
                    if args.save_pi05_cache
                    else None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"episode_index": episode_id, "path": str(path), "error": repr(exc)})
            print(f"EPISODE FAIL episode={episode_id}: {type(exc).__name__}: {exc}", flush=True)
        if args.progress_every_episodes > 0 and episode_count % args.progress_every_episodes == 0:
            print(
                "progress",
                {
                    "episodes_done": episode_count,
                    "episodes_total": len(episode_paths),
                    "kept_samples": len(samples),
                    "failed": len(failed),
                },
                flush=True,
            )

    arrays = stack_samples(samples, cfg)
    arrays["seed"] = np.asarray([optional_int(sample["metadata"].get("seed"), default=-1) for sample in samples])
    source_hdf5 = np.asarray([str(sample["metadata"].get("source_hdf5", "")) for sample in samples], dtype=object)

    summary = build_summary(
        args=args,
        cfg=cfg,
        demo_data_dir=demo_data_dir,
        summary_path=summary_path,
        counters=counters,
        arrays=arrays,
        per_episode=per_episode,
        failed=failed,
        source_hdf5=source_hdf5,
    )
    if args.save_npz:
        npz_path = out_dir / "residual_bc_targets.npz"
        np.savez_compressed(npz_path, **arrays, source_hdf5=source_hdf5)
        summary["npz_path"] = str(npz_path)
    if args.save_parquet:
        parquet_path = save_summary_parquet(out_dir / "residual_bc_targets_summary.parquet", samples)
        summary["parquet_path"] = str(parquet_path)
    csv_path = save_histogram_csv(out_dir / "target_delta_rows.csv", arrays, cfg)
    summary["histogram_csv_path"] = str(csv_path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


def resolve_demo_data_dir(path: Path) -> Path:
    """Return the directory containing episode*.hdf5 files."""
    if (path / "data").is_dir():
        path = path / "data"
    if not path.is_dir():
        raise FileNotFoundError(f"Expert demo directory does not exist: {path}")
    return path


def infer_expert_summary_path(demo_data_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    candidates = [
        demo_data_dir.parent.parent / "expert_success_summary.csv",
        demo_data_dir.parent / "expert_success_summary.csv",
        demo_data_dir / "expert_success_summary.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def load_success_metadata(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row_is_success(row):
                continue
            episode_id = first_present_int(row, ("episode_id", "episode_index", "demo_episode_id"))
            if episode_id is None:
                hdf5_path = row.get("hdf5_path") or row.get("saved_hdf5_path") or row.get("path")
                episode_id = episode_id_from_path(Path(hdf5_path)) if hdf5_path else None
            if episode_id is not None:
                rows[episode_id] = dict(row)
    return rows


def row_is_success(row: dict[str, Any]) -> bool:
    for key in ("success", "expert_success", "check_success", "plan_success"):
        value = str(row.get(key, "")).strip().lower()
        if value in ("true", "1", "yes"):
            return True
    return False


def select_episode_paths(
    episode_paths: list[Path],
    *,
    num_shards: int,
    shard_id: int,
    episode_limit: int | None,
) -> list[Path]:
    selected = [path for idx, path in enumerate(episode_paths) if idx % num_shards == shard_id]
    if episode_limit is not None:
        selected = selected[:episode_limit]
    return selected


def read_expert_hdf5_episode(
    path: Path,
    *,
    episode_id: int,
    seed: int | None,
    default_prompt: str,
) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as root:
        if "/observations/qpos" in root and "/action" in root:
            state = np.asarray(root["/observations/qpos"][:], dtype=np.float32)
            action = np.asarray(root["/action"][:], dtype=np.float32)
            images = {
                "cam_high": np.stack([decode_image(value) for value in root["/observations/images/cam_high"][:]]),
                "cam_left_wrist": np.stack(
                    [decode_image(value) for value in root["/observations/images/cam_left_wrist"][:]]
                ),
                "cam_right_wrist": np.stack(
                    [decode_image(value) for value in root["/observations/images/cam_right_wrist"][:]]
                ),
            }
        else:
            qpos = load_robotwin_qpos14(root)
            state, action = make_state_action_pairs(qpos)
            images = load_robotwin_images(root, length=state.shape[0])
    validate_episode_arrays(path, state, action, images)
    return {
        "episode_index": np.full((state.shape[0],), episode_id, dtype=np.int64),
        "frame_index": np.arange(state.shape[0], dtype=np.int64),
        "seed": np.full((state.shape[0],), -1 if seed is None else seed, dtype=np.int64),
        "state": state,
        "action": action,
        "cam_high": images["cam_high"],
        "cam_left_wrist": images["cam_left_wrist"],
        "cam_right_wrist": images["cam_right_wrist"],
        "prompt": np.asarray([default_prompt] * state.shape[0], dtype=object),
    }


def load_robotwin_qpos14(root: h5py.File) -> np.ndarray:
    if "/joint_action/vector" in root:
        return np.asarray(root["/joint_action/vector"][:], dtype=np.float32)
    required = (
        "/joint_action/left_arm",
        "/joint_action/left_gripper",
        "/joint_action/right_arm",
        "/joint_action/right_gripper",
    )
    missing = [key for key in required if key not in root]
    if missing:
        raise KeyError(f"Expert HDF5 missing qpos fields: {missing}")
    left_arm = np.asarray(root["/joint_action/left_arm"][:], dtype=np.float32)
    left_gripper = np.asarray(root["/joint_action/left_gripper"][:], dtype=np.float32).reshape(-1, 1)
    right_arm = np.asarray(root["/joint_action/right_arm"][:], dtype=np.float32)
    right_gripper = np.asarray(root["/joint_action/right_gripper"][:], dtype=np.float32).reshape(-1, 1)
    return np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1).astype(np.float32)


def make_state_action_pairs(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"qpos must have shape [T,14], got {qpos.shape}.")
    if qpos.shape[0] < 2:
        raise ValueError("qpos must contain at least two frames.")
    return qpos[:-1].astype(np.float32), qpos[1:].astype(np.float32)


def load_robotwin_images(root: h5py.File, *, length: int) -> dict[str, np.ndarray]:
    camera_map = {
        "cam_high": "/observation/head_camera/rgb",
        "cam_left_wrist": "/observation/left_camera/rgb",
        "cam_right_wrist": "/observation/right_camera/rgb",
    }
    missing = [key for key in camera_map.values() if key not in root]
    if missing:
        raise KeyError(f"Expert HDF5 missing camera fields: {missing}")
    return {
        out_key: np.stack([decode_robotwin_image(value) for value in root[h5_key][:length]])
        for out_key, h5_key in camera_map.items()
    }


def decode_robotwin_image(value: Any) -> np.ndarray:
    import cv2

    if isinstance(value, np.ndarray):
        if value.ndim == 3:
            image = value
        else:
            raw = value.tobytes()
            image = cv2.imdecode(np.frombuffer(raw.rstrip(b"\0"), np.uint8), cv2.IMREAD_COLOR)
    else:
        raw = bytes(value).rstrip(b"\0")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode RoboTwin camera image.")
    if image.shape[:2] != (480, 640):
        image = cv2.resize(image, (640, 480))
    return np.asarray(image, dtype=np.uint8)


def validate_episode_arrays(
    path: Path,
    state: np.ndarray,
    action: np.ndarray,
    images: dict[str, np.ndarray],
) -> None:
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
        raise ValueError(f"{path}: state/action must both have shape [T,14], got {state.shape} and {action.shape}.")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: state/action contains NaN or Inf.")
    for key, image in images.items():
        if image.shape[0] != state.shape[0] or image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"{path}: {key} must have shape [T,H,W,3], got {image.shape}.")


@torch.no_grad()
def extract_episode_cache(
    *,
    model: Any,
    episode: dict[str, np.ndarray],
    chunk_size: int,
    batch_size: int,
    device: str,
    gate_mode: str,
) -> dict[str, np.ndarray]:
    episode_len = episode["action"].shape[0]
    valid_count = max(0, episode_len - chunk_size + 1)
    starts = np.arange(valid_count, dtype=np.int64)
    action_chunks = []
    for start in range(0, valid_count, batch_size):
        batch_starts = starts[start : start + batch_size]
        batch = {
            "main_images": torch.as_tensor(episode["cam_high"][batch_starts], device=device),
            "wrist_images": torch.as_tensor(
                np.stack(
                    [episode["cam_left_wrist"][batch_starts], episode["cam_right_wrist"][batch_starts]],
                    axis=1,
                ),
                device=device,
            ),
            "extra_view_images": None,
            "states": torch.as_tensor(episode["state"][batch_starts], device=device),
            "task_descriptions": episode["prompt"][batch_starts].tolist(),
        }
        actions, _ = model.predict_action_batch(batch, mode="eval", compute_values=False)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        action_chunks.append(actions[:, :chunk_size, :].detach().to(dtype=torch.float32).cpu().numpy())

    gate_value = 1 if gate_mode == "all_one" else 0
    payload = {
        "base_action_chunk": concat_or_empty(action_chunks, (0, chunk_size, 14), np.float32),
        "expert_action_chunk": np.stack(
            [episode["action"][idx : idx + chunk_size] for idx in starts],
            axis=0,
        ).astype(np.float32)
        if valid_count
        else np.empty((0, chunk_size, 14), dtype=np.float32),
        "residual_gate_seq": np.full((valid_count, chunk_size), gate_value, dtype=np.uint8),
        "gate_label_chunk": np.full((valid_count,), gate_value, dtype=np.uint8),
        "frame_index": episode["frame_index"][:valid_count].astype(np.int64),
        "episode_index": episode["episode_index"][:valid_count].astype(np.int64),
        "seed": episode["seed"][:valid_count].astype(np.int64),
    }
    return payload


def build_episode_samples(
    *,
    builder: ResidualBCTargetBuilder,
    episode: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    samples = []
    counters = {
        "total_candidate_frames": int(cache["base_action_chunk"].shape[0]),
        "gate_positive_candidate_count": 0,
        "total_kept_samples": 0,
    }
    for row in range(cache["base_action_chunk"].shape[0]):
        gate_seq = cache["residual_gate_seq"][row]
        if int((gate_seq > 0).sum()) >= builder.cfg.min_positive_frames:
            counters["gate_positive_candidate_count"] += 1
        frame_index = int(cache["frame_index"][row])
        sample = builder.build_sample(
            base_qpos_chunk=cache["base_action_chunk"][row],
            expert_qpos_chunk=cache["expert_action_chunk"][row],
            gate_seq=gate_seq,
            current_state=episode["state"][frame_index],
            episode_index=int(cache["episode_index"][row]),
            frame_index=frame_index,
        )
        if sample is None:
            continue
        samples.append(sample)
        counters["total_kept_samples"] += 1
    return samples, counters


def build_summary(
    *,
    args: argparse.Namespace,
    cfg: ResidualBCTargetConfig,
    demo_data_dir: Path,
    summary_path: Path | None,
    counters: dict[str, int],
    arrays: dict[str, np.ndarray],
    per_episode: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    source_hdf5: np.ndarray,
) -> dict[str, Any]:
    kept = counters["total_kept_samples"]
    local = arrays["target_delta_local_xyz"]
    world = arrays["target_delta_world_xyz"]
    norms = arrays["delta_local_norm_clipped"]
    clip_mask = arrays["clip_mask"]
    summary: dict[str, Any] = {
        **counters,
        "positive_ratio": counters["gate_positive_candidate_count"] / counters["total_candidate_frames"]
        if counters["total_candidate_frames"]
        else 0.0,
        "expert_demo_dir": str(demo_data_dir),
        "expert_summary": str(summary_path) if summary_path else None,
        "out_dir": str(Path(args.out_dir)),
        "num_input_episodes": len(per_episode) + len(failed),
        "num_converted_episodes": len(per_episode),
        "failed_episodes": failed,
        "per_episode": per_episode,
        "unique_seeds": sorted({row["seed"] for row in per_episode if row["seed"] is not None}),
        "unique_source_hdf5": sorted(set(source_hdf5.astype(str).tolist())),
        "config": {
            **cfg.__dict__,
            "gate_mode": args.gate_mode,
            "batch_size": args.batch_size,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "episode_limit": args.episode_limit,
            "save_pi05_cache": args.save_pi05_cache,
        },
        "model": {
            "pi05_config": args.pi05_config,
            "pi05_checkpoint": args.pi05_checkpoint,
            "norm_stats_path": args.norm_stats_path,
        },
        "save_npz": args.save_npz,
        "save_parquet": args.save_parquet,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "privileged_note": (
            "RoboTwin expert demos may use privileged simulator state during offline scripted "
            "generation. Only pi05 observations/state and expert qpos targets are converted here; "
            "privileged object pose is not an online residual actor input."
        ),
    }
    summary["per_offset"] = {}
    for offset_i, offset in enumerate(cfg.target_horizon_offsets):
        summary["per_offset"][str(offset)] = {
            "delta_local": axis_stats(local[:, offset_i, :]),
            "delta_world": axis_stats(world[:, offset_i, :]),
            "delta_norm": scalar_stats(norms[:, offset_i]),
            "clip_ratio": float(clip_mask[:, offset_i, :].any(axis=-1).mean()) if kept else 0.0,
            "percentage_near_zero": float((norms[:, offset_i] < 1e-6).mean()) if kept else 0.0,
        }
    return summary


def merge_counters(total: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        total[key] += int(value)


def optional_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def first_present_int(row: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = optional_int(row.get(key))
        if value is not None:
            return value
    return None


def episode_sort_key(path: Path) -> tuple[int, str]:
    return episode_id_from_path(path), path.name


def episode_id_from_path(path: Path) -> int:
    digits = "".join(char for char in path.stem if char.isdigit())
    return int(digits) if digits else 0


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    raise SystemExit(main())
