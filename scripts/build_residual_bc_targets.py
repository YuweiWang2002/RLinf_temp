"""Build residual BC target datasets from gated LeRobot handover episodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from rlinf.algorithms.residual_td3.fk_bridge import AlohaFKBridge, AlohaFKBridgeConfig
from rlinf.algorithms.residual_td3.residual_bc_dataset import (
    ResidualBCTargetBuilder,
    ResidualBCTargetConfig,
)

DEFAULT_LEROBOT_DIR = "/nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert_with_gate/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-dir", default=DEFAULT_LEROBOT_DIR)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--base-action-source",
        choices=("expert_as_base", "pi05_cached_base"),
        default="expert_as_base",
    )
    parser.add_argument("--pi05-action-cache-dir", default=None)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--residual-horizon-k", type=int, default=50)
    parser.add_argument("--target-horizon-offsets", default="0,1,2")
    parser.add_argument("--max-delta-local-xyz", type=float, default=0.05)
    parser.add_argument("--gate-positive-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-positive-frames", type=int, default=1)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--save-parquet", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every-episodes", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ResidualBCTargetConfig(
        chunk_len=args.chunk_len,
        residual_horizon_k=args.residual_horizon_k,
        target_horizon_offsets=parse_offsets(args.target_horizon_offsets),
        max_delta_local_xyz=args.max_delta_local_xyz,
        gate_positive_only=args.gate_positive_only,
        min_positive_frames=args.min_positive_frames,
        base_action_source=args.base_action_source,
    )
    if args.base_action_source == "pi05_cached_base" and not args.pi05_action_cache_dir:
        raise FileNotFoundError(
            "--pi05-action-cache-dir is required for base_action_source=pi05_cached_base. "
            "Expected per-episode npz files with base_action_chunk/action_chunk shaped [T,C,14]."
        )

    bridge = AlohaFKBridge(AlohaFKBridgeConfig(device=args.device))
    builder = ResidualBCTargetBuilder(cfg, bridge)
    episode_paths = sorted(Path(args.lerobot_dir).glob("data/chunk-*/episode_*.parquet"), key=episode_sort_key)
    if args.episode_limit is not None:
        episode_paths = episode_paths[: args.episode_limit]
    if not episode_paths:
        raise FileNotFoundError(f"No LeRobot episode parquet files found under {args.lerobot_dir}.")

    samples, counters = build_dataset(
        builder=builder,
        episode_paths=episode_paths,
        pi05_action_cache_dir=Path(args.pi05_action_cache_dir) if args.pi05_action_cache_dir else None,
        progress_every_episodes=args.progress_every_episodes,
    )
    arrays = stack_samples(samples, cfg)
    summary = build_summary(
        cfg=cfg,
        counters=counters,
        arrays=arrays,
        out_dir=out_dir,
        lerobot_dir=Path(args.lerobot_dir),
        pi05_action_cache_dir=args.pi05_action_cache_dir,
        save_npz=args.save_npz,
        save_parquet=args.save_parquet,
    )
    if args.save_npz:
        npz_path = out_dir / "residual_bc_targets.npz"
        np.savez_compressed(npz_path, **arrays)
        summary["npz_path"] = str(npz_path)
    if args.save_parquet:
        parquet_path = save_summary_parquet(out_dir / "residual_bc_targets_summary.parquet", samples)
        summary["parquet_path"] = str(parquet_path)
    csv_path = save_histogram_csv(out_dir / "target_delta_rows.csv", arrays, cfg)
    summary["histogram_csv_path"] = str(csv_path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def build_dataset(
    *,
    builder: ResidualBCTargetBuilder,
    episode_paths: list[Path],
    pi05_action_cache_dir: Path | None,
    progress_every_episodes: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    samples: list[dict[str, object]] = []
    counters = {
        "total_candidate_frames": 0,
        "gate_positive_candidate_count": 0,
        "total_kept_samples": 0,
    }
    for episode_count, path in enumerate(episode_paths, start=1):
        episode = read_episode(path)
        starts = range(max(episode["action"].shape[0] - builder.cfg.chunk_len + 1, 0))
        episode_id = int(episode["episode_index"][0]) if episode["episode_index"].size else episode_id_from_path(path)
        pi05_cache = (
            load_episode_cache(pi05_action_cache_dir, episode_id)
            if builder.cfg.base_action_source == "pi05_cached_base"
            else None
        )
        episode_endpose = None
        if builder.cfg.base_action_source == "expert_as_base":
            with torch.no_grad():
                episode_endpose = (
                    builder.fk_bridge.qpos14_to_endpose16(
                        torch.as_tensor(episode["action"][None], dtype=torch.float32)
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
        for start in starts:
            counters["total_candidate_frames"] += 1
            expert_chunk = episode["action"][start : start + builder.cfg.chunk_len]
            gate_seq = episode["gate"][start : start + builder.cfg.chunk_len]
            frame_index = int(episode["frame_index"][start])
            if pi05_cache is not None:
                cache_row = cache_row_for_frame(pi05_cache, frame_index)
                if "expert_action_chunk" in pi05_cache:
                    expert_chunk = np.asarray(pi05_cache["expert_action_chunk"][cache_row], dtype=np.float32)
                if "residual_gate_seq" in pi05_cache:
                    gate_seq = np.asarray(pi05_cache["residual_gate_seq"][cache_row], dtype=np.float32)
            if int((gate_seq > 0.0).sum()) >= builder.cfg.min_positive_frames:
                counters["gate_positive_candidate_count"] += 1
            if episode_endpose is not None:
                endpose_chunk = episode_endpose[start : start + builder.cfg.chunk_len]
                sample = builder.build_sample_from_endpose(
                    base_endpose16=endpose_chunk,
                    expert_endpose16=endpose_chunk,
                    gate_seq=gate_seq,
                    current_state=episode["state"][start],
                    episode_index=episode_id,
                    frame_index=int(episode["frame_index"][start]),
                )
            else:
                base_chunk = select_base_chunk(
                    builder.cfg.base_action_source,
                    expert_chunk,
                    pi05_cache,
                    frame_index,
                )
                sample = builder.build_sample(
                    base_qpos_chunk=base_chunk,
                    expert_qpos_chunk=expert_chunk,
                    gate_seq=gate_seq,
                    current_state=episode["state"][start],
                    episode_index=episode_id,
                    frame_index=frame_index,
                )
            if sample is None:
                continue
            samples.append(sample)
            counters["total_kept_samples"] += 1
        if progress_every_episodes > 0 and episode_count % progress_every_episodes == 0:
            print(
                "progress",
                {
                    "episode_path": str(path),
                    "episodes_done": episode_count,
                    "episodes_total": len(episode_paths),
                    "kept_samples": counters["total_kept_samples"],
                },
                flush=True,
            )
    return samples, counters


def read_episode(path: Path) -> dict[str, np.ndarray]:
    columns = [
        "episode_index",
        "frame_index",
        "observation.state",
        "observation.residual_gate",
        "action",
    ]
    table = pq.read_table(path, columns=columns)
    data = table.to_pydict()
    return {
        "episode_index": np.asarray(data["episode_index"], dtype=np.int64),
        "frame_index": np.asarray(data["frame_index"], dtype=np.int64),
        "state": np.asarray(data["observation.state"], dtype=np.float32),
        "gate": np.asarray(data["observation.residual_gate"], dtype=np.float32).reshape(-1),
        "action": np.asarray(data["action"], dtype=np.float32),
    }


def select_base_chunk(
    source: str,
    expert_chunk: np.ndarray,
    cache: dict[str, np.ndarray] | None,
    frame_index: int,
) -> np.ndarray:
    if source == "expert_as_base":
        return expert_chunk
    if cache is None:
        raise FileNotFoundError("pi05_cached_base requires a loaded pi05 action cache.")
    key = "base_action_chunk" if "base_action_chunk" in cache else "action_chunk"
    if key not in cache:
        raise KeyError("pi05 action cache must contain base_action_chunk or action_chunk.")
    return np.asarray(cache[key][cache_row_for_frame(cache, frame_index)], dtype=np.float32)


def load_episode_cache(cache_dir: Path | None, episode_id: int) -> dict[str, np.ndarray]:
    if cache_dir is None:
        raise FileNotFoundError("pi05_cached_base requires --pi05-action-cache-dir.")
    path = find_cache_file(cache_dir, episode_id)
    if path is None:
        raise FileNotFoundError(
            f"No pi05 action cache found for episode {episode_id}. "
            "Expected episode_<id>.npz with base_action_chunk/action_chunk shaped [T,C,14]."
        )
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def cache_row_for_frame(cache: dict[str, np.ndarray], frame_index: int) -> int:
    frame_indices = cache.get("frame_index")
    if frame_indices is None:
        return frame_index
    matches = np.where(frame_indices == frame_index)[0]
    if matches.size == 0:
        raise IndexError(f"frame_index {frame_index} not found in pi05 action cache.")
    return int(matches[0])


def find_cache_file(cache_dir: Path, episode_id: int) -> Path | None:
    candidates = [
        cache_dir / f"episode_{episode_id:06d}.npz",
        cache_dir / f"episode_{episode_id:04d}.npz",
        cache_dir / f"episode_{episode_id}.npz",
    ]
    return next((path for path in candidates if path.exists()), None)


def stack_samples(samples: list[dict[str, object]], cfg: ResidualBCTargetConfig) -> dict[str, np.ndarray]:
    if not samples:
        offsets = len(cfg.target_horizon_offsets)
        return {
            "target_delta_local_xyz": np.empty((0, offsets, 3), dtype=np.float32),
            "target_delta_local_xyz_unclipped": np.empty((0, offsets, 3), dtype=np.float32),
            "target_delta_world_xyz": np.empty((0, offsets, 3), dtype=np.float32),
            "clip_mask": np.empty((0, offsets, 3), dtype=bool),
            "delta_local_norm_unclipped": np.empty((0, offsets), dtype=np.float32),
            "delta_local_norm_clipped": np.empty((0, offsets), dtype=np.float32),
            "base_endpose16": np.empty((0, cfg.chunk_len, 16), dtype=np.float32),
            "expert_endpose16": np.empty((0, cfg.chunk_len, 16), dtype=np.float32),
            "gate_seq": np.empty((0, cfg.chunk_len), dtype=np.float32),
            "episode_index": np.empty((0,), dtype=np.int64),
            "frame_index": np.empty((0,), dtype=np.int64),
        }
    arrays = {
        key: np.stack([np.asarray(sample[key]) for sample in samples]).astype(np.float32)
        for key in (
            "target_delta_local_xyz",
            "target_delta_local_xyz_unclipped",
            "target_delta_world_xyz",
            "delta_local_norm_unclipped",
            "delta_local_norm_clipped",
            "base_endpose16",
            "expert_endpose16",
            "gate_seq",
        )
    }
    arrays["clip_mask"] = np.stack([np.asarray(sample["clip_mask"]) for sample in samples]).astype(bool)
    arrays["episode_index"] = np.asarray(
        [sample["metadata"]["episode_index"] for sample in samples],
        dtype=np.int64,
    )
    arrays["frame_index"] = np.asarray(
        [sample["metadata"]["frame_index"] for sample in samples],
        dtype=np.int64,
    )
    obs_keys = samples[0]["obs_features"].keys()
    for key in obs_keys:
        arrays[f"obs_{key}"] = np.stack(
            [np.asarray(sample["obs_features"][key]) for sample in samples]
        ).astype(np.float32)
    return arrays


def build_summary(
    *,
    cfg: ResidualBCTargetConfig,
    counters: dict[str, int],
    arrays: dict[str, np.ndarray],
    out_dir: Path,
    lerobot_dir: Path,
    pi05_action_cache_dir: str | None,
    save_npz: bool,
    save_parquet: bool,
) -> dict[str, Any]:
    kept = counters["total_kept_samples"]
    summary: dict[str, Any] = {
        **counters,
        "positive_ratio": counters["gate_positive_candidate_count"] / counters["total_candidate_frames"]
        if counters["total_candidate_frames"]
        else 0.0,
        "lerobot_dir": str(lerobot_dir),
        "out_dir": str(out_dir),
        "pi05_action_cache_dir": pi05_action_cache_dir,
        "config": cfg.__dict__,
        "save_npz": save_npz,
        "save_parquet": save_parquet,
    }
    local = arrays["target_delta_local_xyz"]
    world = arrays["target_delta_world_xyz"]
    norms = arrays["delta_local_norm_clipped"]
    clip_mask = arrays["clip_mask"]
    summary["per_offset"] = {}
    for offset_i, offset in enumerate(cfg.target_horizon_offsets):
        summary["per_offset"][str(offset)] = {
            "delta_local": axis_stats(local[:, offset_i, :]),
            "delta_world": axis_stats(world[:, offset_i, :]),
            "delta_norm": scalar_stats(norms[:, offset_i]),
            "clip_ratio": float(clip_mask[:, offset_i, :].any(axis=-1).mean()) if kept else 0.0,
            "percentage_near_zero": float((norms[:, offset_i] < 1e-6).mean()) if kept else 0.0,
        }
    max_abs = float(np.abs(local).max()) if local.size else 0.0
    summary["expert_as_base_sanity"] = {
        "max_abs_target": max_abs,
        "pass": bool(max_abs < 1e-6) if cfg.base_action_source == "expert_as_base" else None,
    }
    return summary


def axis_stats(values: np.ndarray) -> dict[str, list[float]]:
    if values.size == 0:
        return {"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0], "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
    }


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "max": float(values.max()),
    }


def save_histogram_csv(path: Path, arrays: dict[str, np.ndarray], cfg: ResidualBCTargetConfig) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    local = arrays["target_delta_local_xyz"]
    world = arrays["target_delta_world_xyz"]
    norms = arrays["delta_local_norm_clipped"]
    clip = arrays["clip_mask"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "episode_index",
                "frame_index",
                "horizon_offset",
                "delta_local_x",
                "delta_local_y",
                "delta_local_z",
                "delta_world_x",
                "delta_world_y",
                "delta_world_z",
                "delta_norm",
                "is_clipped",
            ],
        )
        writer.writeheader()
        for sample_i in range(local.shape[0]):
            for offset_i, offset in enumerate(cfg.target_horizon_offsets):
                writer.writerow(
                    {
                        "sample_index": sample_i,
                        "episode_index": int(arrays["episode_index"][sample_i]),
                        "frame_index": int(arrays["frame_index"][sample_i]),
                        "horizon_offset": int(offset),
                        "delta_local_x": float(local[sample_i, offset_i, 0]),
                        "delta_local_y": float(local[sample_i, offset_i, 1]),
                        "delta_local_z": float(local[sample_i, offset_i, 2]),
                        "delta_world_x": float(world[sample_i, offset_i, 0]),
                        "delta_world_y": float(world[sample_i, offset_i, 1]),
                        "delta_world_z": float(world[sample_i, offset_i, 2]),
                        "delta_norm": float(norms[sample_i, offset_i]),
                        "is_clipped": int(clip[sample_i, offset_i, :].any()),
                    }
                )
    return path


def save_summary_parquet(path: Path, samples: list[dict[str, object]]) -> Path:
    import pyarrow as pa

    rows = [
        {
            "episode_index": sample["metadata"]["episode_index"],
            "frame_index": sample["metadata"]["frame_index"],
            "target_delta_local_xyz": np.asarray(sample["target_delta_local_xyz"]).reshape(-1).tolist(),
            "target_delta_world_xyz": np.asarray(sample["target_delta_world_xyz"]).reshape(-1).tolist(),
            "clip_mask": np.asarray(sample["clip_mask"]).reshape(-1).astype(np.int8).tolist(),
        }
        for sample in samples
    ]
    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)
    return path


def parse_offsets(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def episode_sort_key(path: Path) -> tuple[int, str]:
    return episode_id_from_path(path), path.name


def episode_id_from_path(path: Path) -> int:
    digits = "".join(char for char in path.stem if char.isdigit())
    return int(digits) if digits else 0


if __name__ == "__main__":
    raise SystemExit(main())
