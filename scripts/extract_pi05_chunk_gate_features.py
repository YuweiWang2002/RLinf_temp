"""Extract pi05 features plus expert action chunks for chunk-aware gate training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from scripts.extract_pi05_features_for_gate import (
    DEFAULT_DATASET_DIR,
    DEFAULT_PROMPT,
    decode_image,
    extract_features,
    load_pi05_model,
    select_episode_shard,
    split_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05-config", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--pi05-checkpoint", required=True)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--min-positive-frames", type=int, default=1)
    parser.add_argument("--feature-source", choices=("action_head_hidden", "prefix_mean"), default="action_head_hidden")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--default-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--episode-shard-index", type=int, default=None)
    parser.add_argument("--episode-num-shards", type=int, default=None)
    parser.add_argument("--progress-every-episodes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_len <= 0:
        raise ValueError("--chunk-len must be positive.")
    if args.min_positive_frames <= 0:
        raise ValueError("--min-positive-frames must be positive.")
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"), key=_episode_sort_key)
    episode_paths = select_episode_shard(
        episode_paths,
        shard_index=args.episode_shard_index,
        num_shards=args.episode_num_shards,
    )
    if args.limit_episodes is not None:
        episode_paths = episode_paths[: args.limit_episodes]
    if not episode_paths:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_dir}.")

    train_eps, val_eps = split_episodes([_episode_id(path) for path in episode_paths], args.train_ratio, args.seed)
    model = load_pi05_model(
        checkpoint=args.pi05_checkpoint,
        config_name=args.pi05_config,
        device=args.device,
        norm_stats_path=args.norm_stats_path,
        model_num_action_chunks=args.chunk_len,
    )

    train = extract_split(
        model=model,
        paths=[path for path in episode_paths if _episode_id(path) in train_eps],
        batch_size=args.batch_size,
        device=args.device,
        feature_source=args.feature_source,
        default_prompt=args.default_prompt,
        chunk_len=args.chunk_len,
        min_positive_frames=args.min_positive_frames,
        progress_every_episodes=args.progress_every_episodes,
    )
    val = extract_split(
        model=model,
        paths=[path for path in episode_paths if _episode_id(path) in val_eps],
        batch_size=args.batch_size,
        device=args.device,
        feature_source=args.feature_source,
        default_prompt=args.default_prompt,
        chunk_len=args.chunk_len,
        min_positive_frames=args.min_positive_frames,
        progress_every_episodes=args.progress_every_episodes,
    )

    np.savez_compressed(feature_dir / "features_train.npz", **train)
    np.savez_compressed(feature_dir / "features_val.npz", **val)
    z_dim = int(train["z"].shape[1] if train["z"].size else val["z"].shape[1])
    config = {
        "pi05_config": args.pi05_config,
        "pi05_checkpoint": args.pi05_checkpoint,
        "norm_stats_path": args.norm_stats_path,
        "dataset_dir": str(dataset_dir),
        "chunk_len": args.chunk_len,
        "feature_source": args.feature_source,
        "z_dim": z_dim,
        "action_dim": 14,
        "model_action_chunk": int(model.config.action_chunk),
        "min_positive_frames": args.min_positive_frames,
        "train_episode_indices": sorted(train_eps),
        "val_episode_indices": sorted(val_eps),
        "episode_shard_index": args.episode_shard_index,
        "episode_num_shards": args.episode_num_shards,
        "train_gate_scalar_positive_ratio": positive_ratio(train["gate_scalar"]),
        "val_gate_scalar_positive_ratio": positive_ratio(val["gate_scalar"]),
        "train_gate_seq_positive_ratio": positive_ratio(train["gate_seq"]),
        "val_gate_seq_positive_ratio": positive_ratio(val["gate_seq"]),
        "default_prompt": args.default_prompt,
        "tail_policy": "drop",
    }
    (feature_dir / "feature_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    return 0


def extract_split(
    *,
    model: Any,
    paths: list[Path],
    batch_size: int,
    device: str,
    feature_source: str,
    default_prompt: str,
    chunk_len: int,
    min_positive_frames: int,
    progress_every_episodes: int,
) -> dict[str, np.ndarray]:
    features = []
    action_chunks = []
    gate_sequences = []
    gate_scalars = []
    episode_indices = []
    frame_indices = []
    total_chunks = 0
    for episode_count, path in enumerate(paths, start=1):
        episode = read_episode(path, default_prompt)
        starts = np.arange(max(episode["state"].shape[0] - chunk_len + 1, 0), dtype=np.int64)
        for start in range(0, starts.shape[0], batch_size):
            batch_starts = starts[start : start + batch_size]
            batch = {
                "cam_high": episode["cam_high"][batch_starts],
                "cam_left_wrist": episode["cam_left_wrist"][batch_starts],
                "cam_right_wrist": episode["cam_right_wrist"][batch_starts],
                "state": episode["state"][batch_starts],
                "prompt": episode["prompt"][batch_starts],
            }
            z = extract_features(model, batch, device=device, feature_source=feature_source)
            gates = np.stack([episode["gate"][idx : idx + chunk_len] for idx in batch_starts], axis=0)
            gates = gates.astype(np.float32)[:, :, None]
            scalar = (gates.sum(axis=1) >= float(min_positive_frames)).astype(np.float32)
            actions = np.stack([episode["action"][idx : idx + chunk_len] for idx in batch_starts], axis=0)
            features.append(z)
            action_chunks.append(actions.astype(np.float32))
            gate_sequences.append(gates)
            gate_scalars.append(scalar)
            episode_indices.append(episode["episode_index"][batch_starts].astype(np.int64))
            frame_indices.append(episode["frame_index"][batch_starts].astype(np.int64))
            total_chunks += int(batch_starts.shape[0])
        if progress_every_episodes > 0 and episode_count % progress_every_episodes == 0:
            print(
                "progress",
                {
                    "episode_path": str(path),
                    "episodes_done": episode_count,
                    "episodes_total": len(paths),
                    "chunks_seen": total_chunks,
                },
                flush=True,
            )
    return {
        "z": concat_or_empty(features, (0, 0), np.float32),
        "action_chunk": concat_or_empty(action_chunks, (0, chunk_len, 14), np.float32),
        "gate_seq": concat_or_empty(gate_sequences, (0, chunk_len, 1), np.float32),
        "gate_scalar": concat_or_empty(gate_scalars, (0, 1), np.float32),
        "episode_index": concat_or_empty(episode_indices, (0,), np.int64),
        "frame_index": concat_or_empty(frame_indices, (0,), np.int64),
    }


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
        "cam_left_wrist": np.stack(
            [decode_image(value) for value in data["observation.images.cam_left_wrist"]]
        ),
        "cam_right_wrist": np.stack(
            [decode_image(value) for value in data["observation.images.cam_right_wrist"]]
        ),
        "prompt": np.asarray([default_prompt] * table.num_rows, dtype=object),
    }


def concat_or_empty(values: list[np.ndarray], shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if not values:
        return np.empty(shape, dtype=dtype)
    return np.concatenate(values, axis=0).astype(dtype)


def positive_ratio(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _episode_sort_key(path: Path) -> tuple[int, str]:
    return (_episode_id(path), path.name)


def _episode_id(path: Path) -> int:
    digits = "".join(char for char in path.stem if char.isdigit())
    return int(digits) if digits else 0


if __name__ == "__main__":
    raise SystemExit(main())
