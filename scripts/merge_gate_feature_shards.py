"""Merge multi-GPU GateHead feature shards into one feature directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_feature_dirs = sorted(shards_dir.glob("shard_*/features"))
    if not shard_feature_dirs:
        raise FileNotFoundError(f"No shard feature dirs found under {shards_dir}.")

    train_parts = [load_npz(path / "features_train.npz") for path in shard_feature_dirs]
    val_parts = [load_npz(path / "features_val.npz") for path in shard_feature_dirs]
    train = concat_parts(train_parts)
    val = concat_parts(val_parts)
    np.savez_compressed(out_dir / "features_train.npz", **train)
    np.savez_compressed(out_dir / "features_val.npz", **val)

    configs = [load_json(path / "feature_config.json") for path in shard_feature_dirs]
    config = dict(configs[0])
    config["merged_from"] = [str(path) for path in shard_feature_dirs]
    config["train_episode_indices"] = sorted(
        episode for item in configs for episode in item.get("train_episode_indices", [])
    )
    config["val_episode_indices"] = sorted(
        episode for item in configs for episode in item.get("val_episode_indices", [])
    )
    config["train_positive_ratio"] = float(train["y"].mean()) if train["y"].size else 0.0
    config["val_positive_ratio"] = float(val["y"].mean()) if val["y"].size else 0.0
    config["num_feature_shards"] = len(shard_feature_dirs)
    config["num_train_frames"] = int(train["y"].shape[0])
    config["num_val_frames"] = int(val["y"].shape[0])
    config.pop("episode_shard_index", None)
    config.pop("episode_num_shards", None)
    (out_dir / "feature_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    return 0


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def concat_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
