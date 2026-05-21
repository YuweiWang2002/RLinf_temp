"""Extract frozen pi05 action-head hidden features for residual gate training."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

DEFAULT_DATASET_DIR = (
    "/nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert_with_gate/"
)
DEFAULT_OUTPUT_DIR = "logs/gate_head_vla_feature"
DEFAULT_PROMPT = (
    "Use the left arm to grasp the red block on the table, handover it to the right arm "
    "and place it on the blue pad."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05-config", default="pi05_aloha_robotwin_handover")
    parser.add_argument("--pi05-checkpoint", required=True)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-source", choices=("action_head_hidden", "prefix_mean"), default="action_head_hidden")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--episode-shard-index", type=int, default=None)
    parser.add_argument("--episode-num-shards", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--default-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--progress-every-episodes", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    del args.num_workers
    batch_size = args.feature_batch_size or args.batch_size
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
    )

    train = extract_split(
        model=model,
        paths=[path for path in episode_paths if _episode_id(path) in train_eps],
        batch_size=batch_size,
        device=args.device,
        feature_source=args.feature_source,
        default_prompt=args.default_prompt,
        max_frames=args.max_frames,
        progress_every_episodes=args.progress_every_episodes,
    )
    val = extract_split(
        model=model,
        paths=[path for path in episode_paths if _episode_id(path) in val_eps],
        batch_size=batch_size,
        device=args.device,
        feature_source=args.feature_source,
        default_prompt=args.default_prompt,
        max_frames=args.max_frames,
        progress_every_episodes=args.progress_every_episodes,
    )

    np.savez_compressed(feature_dir / "features_train.npz", **train)
    np.savez_compressed(feature_dir / "features_val.npz", **val)
    z_dim = int(train["z"].shape[1] if train["z"].size else val["z"].shape[1])
    config = {
        "pi05_config": args.pi05_config,
        "pi05_checkpoint": args.pi05_checkpoint,
        "feature_source": args.feature_source,
        "z_dim": z_dim,
        "dataset_dir": str(dataset_dir),
        "train_episode_indices": sorted(train_eps),
        "val_episode_indices": sorted(val_eps),
        "episode_shard_index": args.episode_shard_index,
        "episode_num_shards": args.episode_num_shards,
        "train_positive_ratio": float(train["y"].mean()) if train["y"].size else 0.0,
        "val_positive_ratio": float(val["y"].mean()) if val["y"].size else 0.0,
        "default_prompt": args.default_prompt,
    }
    (feature_dir / "feature_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))
    return 0


def load_pi05_model(
    *,
    checkpoint: str,
    config_name: str,
    device: str,
    norm_stats_path: str | None,
) -> Any:
    from omegaconf import OmegaConf

    from rlinf.models import get_model
    from scripts.probe_openpi_robotwin import install_norm_stats_fallback

    model_path = Path(checkpoint)
    if model_path.name.endswith(".safetensors"):
        model_path = model_path.parent
    cfg = OmegaConf.load("examples/embodiment/config/model/pi0_5.yaml")
    cfg.model_path = str(model_path)
    cfg.action_dim = 14
    cfg.openpi.config_name = config_name
    if norm_stats_path:
        cfg.norm_stats_path = norm_stats_path
    else:
        install_norm_stats_fallback(action_dim=14)
    root = OmegaConf.create({"actor": {"model": cfg}})
    OmegaConf.resolve(root)
    with force_model_factory_cpu(device):
        model = get_model(root.actor.model)
    model.eval()
    model.to(torch.device(device))
    model.requires_grad_(False)
    return model


@contextmanager
def force_model_factory_cpu(device: str):
    if torch.device(device).type != "cpu":
        yield
        return

    from rlinf.scheduler.worker import Worker

    old_platform = Worker.torch_platform
    old_device_type = Worker.torch_device_type
    Worker.torch_platform = None
    Worker.torch_device_type = "cpu"
    try:
        yield
    finally:
        Worker.torch_platform = old_platform
        Worker.torch_device_type = old_device_type


def extract_split(
    *,
    model: Any,
    paths: list[Path],
    batch_size: int,
    device: str,
    feature_source: str,
    default_prompt: str,
    max_frames: int | None,
    progress_every_episodes: int,
) -> dict[str, np.ndarray]:
    features = []
    labels = []
    episode_indices = []
    frame_indices = []
    states = []
    frames_seen = 0
    for episode_count, path in enumerate(paths, start=1):
        episode = read_episode(path, default_prompt)
        episode_frames = 0
        for start in range(0, episode["state"].shape[0], batch_size):
            if max_frames is not None and frames_seen >= max_frames:
                break
            stop = min(start + batch_size, episode["state"].shape[0])
            if max_frames is not None:
                stop = min(stop, start + max_frames - frames_seen)
            batch = {key: value[start:stop] for key, value in episode.items()}
            z = extract_features(model, batch, device=device, feature_source=feature_source)
            features.append(z)
            labels.append(batch["gate"].astype(np.float32)[:, None])
            episode_indices.append(batch["episode_index"].astype(np.int64))
            frame_indices.append(batch["frame_index"].astype(np.int64))
            states.append(batch["state"].astype(np.float32))
            frames_seen += stop - start
            episode_frames += stop - start
        if max_frames is not None and frames_seen >= max_frames:
            break
        if progress_every_episodes > 0 and episode_count % progress_every_episodes == 0:
            print(
                "progress",
                {
                    "episode_path": str(path),
                    "episodes_done": episode_count,
                    "episodes_total": len(paths),
                    "episode_frames": episode_frames,
                    "frames_seen": frames_seen,
                },
                flush=True,
            )
    return {
        "z": _concat_or_empty(features, (0, 0), np.float32),
        "y": _concat_or_empty(labels, (0, 1), np.float32),
        "episode_index": _concat_or_empty(episode_indices, (0,), np.int64),
        "frame_index": _concat_or_empty(frame_indices, (0,), np.int64),
        "state": _concat_or_empty(states, (0, 14), np.float32),
    }


def read_episode(path: Path, default_prompt: str) -> dict[str, np.ndarray]:
    columns = [
        "episode_index",
        "frame_index",
        "observation.state",
        "observation.residual_gate",
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
        "cam_high": np.stack([decode_image(value) for value in data["observation.images.cam_high"]]),
        "cam_left_wrist": np.stack(
            [decode_image(value) for value in data["observation.images.cam_left_wrist"]]
        ),
        "cam_right_wrist": np.stack(
            [decode_image(value) for value in data["observation.images.cam_right_wrist"]]
        ),
        "prompt": np.asarray([default_prompt] * table.num_rows, dtype=object),
    }


def decode_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("bytes", value)
    if isinstance(value, bytes):
        return np.asarray(Image.open(BytesIO(value)).convert("RGB"))
    return np.asarray(value, dtype=np.uint8)


@torch.no_grad()
def extract_features(model: Any, batch: dict[str, np.ndarray], *, device: str, feature_source: str) -> np.ndarray:
    env_obs = {
        "main_images": torch.as_tensor(batch["cam_high"], device=device),
        "wrist_images": torch.as_tensor(
            np.stack([batch["cam_left_wrist"], batch["cam_right_wrist"]], axis=1),
            device=device,
        ),
        "extra_view_images": None,
        "states": torch.as_tensor(batch["state"], device=device),
        "task_descriptions": batch["prompt"].tolist(),
    }
    to_process_obs = model.obs_processor(env_obs)
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)

    from openpi.models import model as _model

    observation = _model.Observation.from_dict(processed_obs)
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation,
        train=False,
    )
    images = [image.to(device) for image in images]
    img_masks = [mask.to(device) for mask in img_masks]
    state = state.to(device)
    prefix_output, prefix_pad_masks, past_key_values = model._build_prefix_cache(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
    )
    if feature_source == "prefix_mean":
        z = prefix_output.to(dtype=torch.float32).mean(dim=1)
        return z.detach().cpu().numpy().astype(np.float32)
    batch_size = state.shape[0]
    x_t = torch.zeros(
        batch_size,
        model.config.action_horizon,
        model.config.action_dim,
        device=device,
        dtype=model.action_in_proj.weight.dtype,
    )
    timestep = torch.full((batch_size,), 0.5, device=device, dtype=torch.float32)
    suffix_out = model.get_suffix_out(
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    )
    z = suffix_out[:, : model.config.action_chunk].mean(dim=1)
    return z.detach().cpu().numpy().astype(np.float32)


def split_episodes(episode_indices: list[int], train_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(episode_indices, dtype=np.int64)
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_ratio))
    train = set(shuffled[:train_count].astype(int).tolist())
    val = set(shuffled[train_count:].astype(int).tolist())
    return train, val


def select_episode_shard(
    episode_paths: list[Path],
    *,
    shard_index: int | None,
    num_shards: int | None,
) -> list[Path]:
    if shard_index is None and num_shards is None:
        return episode_paths
    if shard_index is None or num_shards is None:
        raise ValueError("--episode-shard-index and --episode-num-shards must be set together.")
    if num_shards <= 0:
        raise ValueError("--episode-num-shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--episode-shard-index must be in [0, episode_num_shards).")
    return episode_paths[shard_index::num_shards]


def _concat_or_empty(values: list[np.ndarray], shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if not values:
        return np.empty(shape, dtype=dtype)
    return np.concatenate(values, axis=0).astype(dtype)


def _episode_sort_key(path: Path) -> tuple[int, str]:
    return (_episode_id(path), path.name)


def _episode_id(path: Path) -> int:
    digits = "".join(char for char in path.stem if char.isdigit())
    return int(digits) if digits else 0


if __name__ == "__main__":
    raise SystemExit(main())
