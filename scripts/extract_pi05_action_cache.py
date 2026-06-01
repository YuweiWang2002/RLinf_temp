"""Extract frozen pi05 qpos14 action chunks for LeRobot observations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from scripts.extract_pi05_features_for_gate import (
    DEFAULT_DATASET_DIR,
    DEFAULT_PROMPT,
    decode_image,
    load_pi05_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05-config", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--pi05-checkpoint", required=True)
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", "--chunk-len", dest="chunk_size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-action-head-hidden", action="store_true")
    parser.add_argument("--save-expert-action-chunk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-residual-gate-seq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every-episodes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_tmp_files(output_dir)

    all_paths = sorted(Path(args.dataset_dir).glob("data/chunk-*/episode_*.parquet"), key=episode_sort_key)
    selected_paths = select_episode_paths(
        all_paths,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        max_episodes=args.max_episodes,
    )
    if not selected_paths:
        raise FileNotFoundError(f"No selected episode parquet files under {args.dataset_dir}.")

    model = load_pi05_model(
        checkpoint=args.pi05_checkpoint,
        config_name=args.pi05_config,
        device=args.device,
        norm_stats_path=args.norm_stats_path,
        model_num_action_chunks=args.chunk_size,
    )
    completed = []
    skipped = []
    failed = []
    per_episode = []
    for episode_count, path in enumerate(selected_paths, start=1):
        episode_id = episode_id_from_path(path)
        out_path = episode_cache_path(output_dir, episode_id)
        if should_skip_existing(out_path, skip_existing=args.skip_existing, overwrite=args.overwrite):
            skipped.append(episode_id)
            try:
                with np.load(out_path) as cache:
                    per_episode.append(
                        {
                            "episode_index": episode_id,
                            "path": str(out_path),
                            "num_frames_valid": int(cache["base_action_chunk"].shape[0]),
                            "skipped_existing": True,
                        }
                    )
            except Exception:  # noqa: BLE001
                per_episode.append(
                    {
                        "episode_index": episode_id,
                        "path": str(out_path),
                        "num_frames_valid": None,
                        "skipped_existing": True,
                    }
                )
            continue
        try:
            episode = read_episode(path, args.default_prompt)
            payload = extract_episode_cache(
                model=model,
                episode=episode,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
                device=args.device,
                save_action_head_hidden=args.save_action_head_hidden,
                save_expert_action_chunk=args.save_expert_action_chunk,
                save_residual_gate_seq=args.save_residual_gate_seq,
            )
            validate_cache_schema(payload, chunk_size=args.chunk_size)
            write_episode_cache_atomic(out_path, payload)
            smoke_check_episode_cache(out_path, episode_len=episode["action"].shape[0], chunk_size=args.chunk_size)
            completed.append(episode_id)
            per_episode.append(
                {
                    "episode_index": episode_id,
                    "path": str(out_path),
                    "num_frames_total": int(episode["action"].shape[0]),
                    "num_frames_valid": int(payload["base_action_chunk"].shape[0]),
                    "skipped_existing": False,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"episode_index": episode_id, "path": str(path), "error": repr(exc)})
            print(f"EPISODE FAIL episode={episode_id}: {type(exc).__name__}: {exc}", flush=True)
            if args.fail_fast:
                raise
        if args.progress_every_episodes > 0 and episode_count % args.progress_every_episodes == 0:
            manifest = build_manifest(args, selected_paths, completed, skipped, failed, per_episode)
            write_manifest(output_dir / "manifest.json", manifest)
            print(
                "progress",
                {
                    "episodes_done": episode_count,
                    "episodes_total": len(selected_paths),
                    "completed": len(completed),
                    "skipped": len(skipped),
                    "failed": len(failed),
                },
                flush=True,
            )

    manifest = build_manifest(args, selected_paths, completed, skipped, failed, per_episode)
    write_manifest(output_dir / "manifest.json", manifest)
    write_manifest(output_dir / "shard_info.json", manifest["shard"])
    if args.max_episodes is not None:
        smoke_summary = build_smoke_summary(output_dir, per_episode, args.chunk_size)
        write_manifest(output_dir / "smoke_summary.json", smoke_summary)
    print(json.dumps(manifest, indent=2))
    return 1 if failed and args.fail_fast else 0


def extract_episode_cache(
    *,
    model: Any,
    episode: dict[str, np.ndarray],
    chunk_size: int,
    batch_size: int,
    device: str,
    save_action_head_hidden: bool,
    save_expert_action_chunk: bool,
    save_residual_gate_seq: bool,
) -> dict[str, np.ndarray]:
    episode_len = episode["action"].shape[0]
    valid_count = max(0, episode_len - chunk_size + 1)
    starts = np.arange(valid_count, dtype=np.int64)
    action_chunks = []
    hidden_chunks = []
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
        with torch.no_grad():
            actions, info = model.predict_action_batch(
                batch,
                mode="eval",
                compute_values=False,
                return_action_head_hidden=save_action_head_hidden,
                action_head_hidden_chunk_len=chunk_size if save_action_head_hidden else None,
            )
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        actions = actions[:, :chunk_size, :].detach().to(dtype=torch.float32).cpu().numpy()
        action_chunks.append(actions)
        if save_action_head_hidden:
            hidden = info["action_head_hidden"].detach().to(dtype=torch.float32).cpu().numpy()
            hidden_chunks.append(hidden)

    payload = {
        "base_action_chunk": concat_or_empty(action_chunks, (0, chunk_size, 14), np.float32),
        "frame_index": episode["frame_index"][:valid_count].astype(np.int64),
        "episode_index": episode["episode_index"][:valid_count].astype(np.int64),
    }
    if save_action_head_hidden:
        payload["action_head_hidden"] = concat_or_empty(hidden_chunks, (0, 0), np.float32)
    if save_residual_gate_seq:
        payload["residual_gate_seq"] = np.stack(
            [episode["gate"][idx : idx + chunk_size] for idx in starts],
            axis=0,
        ).astype(np.uint8)
        payload["gate_label_chunk"] = (payload["residual_gate_seq"].sum(axis=1) > 0).astype(np.uint8)
    if save_expert_action_chunk:
        payload["expert_action_chunk"] = np.stack(
            [episode["action"][idx : idx + chunk_size] for idx in starts],
            axis=0,
        ).astype(np.float32)
    return payload


def read_episode(path: Path, default_prompt: str) -> dict[str, np.ndarray]:
    columns = [
        "episode_index",
        "frame_index",
        "observation.state",
        "observation.residual_gate",
        "action",
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    table = pq.read_table(path, columns=columns)
    data = table.to_pydict()
    return {
        "episode_index": np.asarray(data["episode_index"], dtype=np.int64),
        "frame_index": np.asarray(data["frame_index"], dtype=np.int64),
        "state": np.asarray(data["observation.state"], dtype=np.float32),
        "gate": np.asarray(data["observation.residual_gate"], dtype=np.float32).reshape(-1),
        "action": np.asarray(data["action"], dtype=np.float32),
        "cam_high": np.stack([decode_image(value) for value in data["observation.images.cam_high"]]),
        "cam_left_wrist": np.stack([decode_image(value) for value in data["observation.images.cam_left_wrist"]]),
        "cam_right_wrist": np.stack([decode_image(value) for value in data["observation.images.cam_right_wrist"]]),
        "prompt": np.asarray([default_prompt] * table.num_rows, dtype=object),
    }


def select_episode_paths(
    episode_paths: list[Path],
    *,
    num_shards: int,
    shard_id: int,
    episode_start: int | None,
    episode_end: int | None,
    max_episodes: int | None,
) -> list[Path]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must be in [0, num_shards).")
    selected = [
        path
        for path in episode_paths
        if (episode_start is None or episode_id_from_path(path) >= episode_start)
        and (episode_end is None or episode_id_from_path(path) < episode_end)
    ]
    selected = [path for path in selected if episode_id_from_path(path) % num_shards == shard_id]
    if max_episodes is not None:
        selected = selected[:max_episodes]
    return selected


def should_skip_existing(path: Path, *, skip_existing: bool, overwrite: bool) -> bool:
    return path.exists() and skip_existing and not overwrite


def validate_cache_schema(payload: dict[str, np.ndarray], *, chunk_size: int) -> None:
    if "base_action_chunk" not in payload:
        raise ValueError("cache payload missing base_action_chunk.")
    base = np.asarray(payload["base_action_chunk"])
    if base.ndim != 3 or base.shape[1:] != (chunk_size, 14) or base.dtype != np.float32:
        raise ValueError(
            f"base_action_chunk must have shape [T,{chunk_size},14] float32, "
            f"got {base.shape} {base.dtype}."
        )
    if not np.isfinite(base).all():
        raise ValueError("base_action_chunk contains NaN or Inf.")
    frame_index = np.asarray(payload.get("frame_index"))
    if frame_index.shape != (base.shape[0],):
        raise ValueError("frame_index must have shape [T_valid].")
    if frame_index.size and not np.array_equal(frame_index, np.arange(frame_index.size, dtype=frame_index.dtype)):
        raise ValueError("frame_index must be contiguous from 0 to T_valid-1 for the current LeRobot cache.")
    episode_index = np.asarray(payload.get("episode_index"))
    if episode_index.shape not in ((), (base.shape[0],)):
        raise ValueError("episode_index must be scalar or shape [T_valid].")
    if "expert_action_chunk" in payload:
        expert = np.asarray(payload["expert_action_chunk"])
        if expert.shape != base.shape or expert.dtype != np.float32:
            raise ValueError("expert_action_chunk must match base_action_chunk shape and dtype float32.")
        if not np.isfinite(expert).all():
            raise ValueError("expert_action_chunk contains NaN or Inf.")
    if "residual_gate_seq" in payload:
        gate = np.asarray(payload["residual_gate_seq"])
        if gate.shape != (base.shape[0], chunk_size):
            raise ValueError(f"residual_gate_seq must have shape [T,{chunk_size}].")
    if "gate_label_chunk" in payload and np.asarray(payload["gate_label_chunk"]).shape != (base.shape[0],):
        raise ValueError("gate_label_chunk must have shape [T_valid].")
    if "action_head_hidden" in payload and np.asarray(payload["action_head_hidden"]).shape[0] != base.shape[0]:
        raise ValueError("action_head_hidden first dimension must match T_valid.")


def smoke_check_episode_cache(path: Path, *, episode_len: int, chunk_size: int) -> dict[str, Any]:
    with np.load(path) as cache:
        payload = {key: cache[key] for key in cache.files}
    validate_cache_schema(payload, chunk_size=chunk_size)
    expected_valid = max(0, episode_len - chunk_size + 1)
    actual_valid = int(payload["base_action_chunk"].shape[0])
    if actual_valid != expected_valid:
        raise ValueError(f"T_valid mismatch for {path}: expected {expected_valid}, got {actual_valid}.")
    return {"path": str(path), "num_frames_valid": actual_valid, "expected_valid": expected_valid}


def write_episode_cache_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    with np.load(tmp_path) as cache:
        reloaded = {key: cache[key] for key in cache.files}
    chunk_size = int(payload["base_action_chunk"].shape[1])
    validate_cache_schema(reloaded, chunk_size=chunk_size)
    os.replace(tmp_path, path)


def build_manifest(
    args: argparse.Namespace,
    selected_paths: list[Path],
    completed: list[int],
    skipped: list[int],
    failed: list[dict[str, Any]],
    per_episode: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset_path": str(Path(args.dataset_dir)),
        "output_dir": str(Path(args.output_dir)),
        "chunk_size": int(args.chunk_size),
        "action_dim": 14,
        "total_episodes": len(selected_paths),
        "completed_episodes": completed,
        "skipped_existing_episodes": skipped,
        "failed_episodes": failed,
        "per_episode": per_episode,
        "model": {
            "pi05_config": args.pi05_config,
            "pi05_checkpoint": args.pi05_checkpoint,
            "norm_stats_path": args.norm_stats_path,
        },
        "shard": {
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "device": args.device,
            "episode_start": args.episode_start,
            "episode_end": args.episode_end,
            "max_episodes": args.max_episodes,
        },
        "save_expert_action_chunk": bool(args.save_expert_action_chunk),
        "save_residual_gate_seq": bool(args.save_residual_gate_seq),
        "save_action_head_hidden": bool(args.save_action_head_hidden),
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
    }


def build_smoke_summary(output_dir: Path, per_episode: list[dict[str, Any]], chunk_size: int) -> dict[str, Any]:
    checks = []
    for row in per_episode:
        if row.get("skipped_existing"):
            continue
        path = Path(str(row["path"]))
        checks.append(
            smoke_check_episode_cache(
                path,
                episode_len=int(row["num_frames_total"]),
                chunk_size=chunk_size,
            )
        )
    return {
        "output_dir": str(output_dir),
        "chunk_size": chunk_size,
        "checked_episodes": checks,
        "pass": True,
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def remove_stale_tmp_files(output_dir: Path) -> None:
    for path in output_dir.glob("episode_*.npz.tmp"):
        path.unlink()


def episode_cache_path(output_dir: Path, episode_id: int) -> Path:
    return output_dir / f"episode_{episode_id:06d}.npz"


def concat_or_empty(values: list[np.ndarray], shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if not values:
        return np.empty(shape, dtype=dtype)
    return np.concatenate(values, axis=0).astype(dtype)


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
