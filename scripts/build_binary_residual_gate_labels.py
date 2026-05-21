"""Build binary residual gate sidecar labels from RoboTwin raw HDF5 episodes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from rlinf.algorithms.residual_td3.gate_labeler import (
    BinaryGateLabelerConfig,
    BinaryResidualGateLabeler,
)

DEFAULT_RAW_DIR = "/nfs/data3/rlinf_data/handover/raw_collect/handover_block/data/"
DEFAULT_OUT_DIR = "/nfs/data3/rlinf_data/handover/residual_gate_labels_binary/"
DEFAULT_LEROBOT_DIR = "/nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--lerobot-dir", default=DEFAULT_LEROBOT_DIR)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--plot-limit", type=int, default=0)
    parser.add_argument("--moving-eps", type=float, default=1e-3)
    parser.add_argument("--distance-small-threshold", type=float, default=None)
    parser.add_argument("--distance-stable-eps", type=float, default=1e-4)
    parser.add_argument("--gripper-closed-threshold", type=float, default=None)
    parser.add_argument("--gripper-closed-is-high", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-dilation", type=int, default=3)
    parser.add_argument("--min-positive-segment-len", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    episodes = sorted(raw_dir.glob("episode*.hdf5"), key=_episode_sort_key)
    if args.episode_limit is not None:
        episodes = episodes[: args.episode_limit]
    if not episodes:
        raise FileNotFoundError(f"No episode*.hdf5 files found under {raw_dir}.")

    if args.inspect_only:
        inspect_episode(episodes[0], Path(args.lerobot_dir))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    if args.plot_limit > 0:
        plots_dir.mkdir(parents=True, exist_ok=True)

    cfg = BinaryGateLabelerConfig(
        moving_eps=args.moving_eps,
        distance_small_threshold=args.distance_small_threshold,
        distance_stable_eps=args.distance_stable_eps,
        gripper_closed_threshold=args.gripper_closed_threshold,
        gripper_closed_is_high=args.gripper_closed_is_high,
        contact_dilation=args.contact_dilation,
        min_positive_segment_len=args.min_positive_segment_len,
    )
    labeler = BinaryResidualGateLabeler(cfg)

    summaries: list[dict[str, Any]] = []
    for plot_index, episode_path in enumerate(episodes):
        episode = load_episode_fields(episode_path)
        result = labeler.label_episode(
            left_ee_pos=episode["left_ee_pos"],
            right_ee_pos=episode["right_ee_pos"],
            left_gripper=episode["left_gripper"],
            right_gripper=episode["right_gripper"],
        )
        episode_id = _episode_id(episode_path)
        save_sidecar(out_dir / f"episode{episode_id:06d}_gate.npz", result, episode, cfg)
        summaries.append(summarize_episode(episode_path.name, result))
        if plot_index < args.plot_limit:
            maybe_plot_episode(plots_dir / f"episode{episode_id:06d}.png", result, episode)

    summary = summarize_all(summaries, cfg)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def inspect_episode(path: Path, lerobot_dir: Path | None = None) -> None:
    print(f"FILE {path}")
    with h5py.File(path, "r") as h5_file:
        h5_file.visititems(_print_hdf5_node)
    episode = load_episode_fields(path)
    print("\nKEY_FIELDS")
    for key in (
        "qpos14",
        "left_ee",
        "right_ee",
        "left_gripper",
        "right_gripper",
        "left_ee_pos",
        "right_ee_pos",
    ):
        value = episode[key]
        print(f"{key}: shape={value.shape} dtype={value.dtype}")
        print(f"  sample={np.array2string(value[:3], precision=5, threshold=40)}")
    print("ee16: not stored as one [T,16] dataset; reconstruct from left/right pose7 + grippers.")
    print("ee16_layout: left xyz + left quat4 + left gripper + right xyz + right quat4 + right gripper")
    print("quat_order_in_raw_pose7: inferred xyzw from RoboTwin pose convention; verify in RoboTwin converter before metadata merge.")
    print_lengths(episode)
    if lerobot_dir is not None:
        print_lerobot_length_compare(lerobot_dir, _episode_id(path), episode["qpos14"].shape[0])


def _print_hdf5_node(name: str, obj: Any) -> None:
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET {name} shape={obj.shape} dtype={obj.dtype}")
        if obj.dtype.kind in {"S", "O"} or _is_large_image_like(name, obj):
            print("  sample=<skipped bytes/image payload>")
            return
        try:
            sample = obj[()] if obj.shape == () else obj[: min(3, obj.shape[0])]
            print(f"  sample={np.array2string(np.asarray(sample), precision=5, threshold=60)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  sample_error={type(exc).__name__}: {exc}")
    else:
        print(f"GROUP {name}")


def _is_large_image_like(name: str, obj: h5py.Dataset) -> bool:
    return "rgb" in name.lower() or (obj.shape and obj.dtype.itemsize * max(1, obj.shape[0]) > 2_000_000)


def load_episode_fields(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5_file:
        left_ee = _read_first_dataset(h5_file, ["endpose/left_endpose"])
        right_ee = _read_first_dataset(h5_file, ["endpose/right_endpose"])
        left_gripper = _read_first_dataset(
            h5_file,
            ["endpose/left_gripper", "joint_action/left_gripper"],
        )
        right_gripper = _read_first_dataset(
            h5_file,
            ["endpose/right_gripper", "joint_action/right_gripper"],
        )
        qpos14 = _read_first_dataset(h5_file, ["joint_action/vector"])
    if left_ee.shape[-1] != 7 or right_ee.shape[-1] != 7:
        raise ValueError(f"{path}: expected left/right endpose shape [T, 7].")
    return {
        "left_ee": left_ee,
        "right_ee": right_ee,
        "left_ee_pos": left_ee[:, :3],
        "right_ee_pos": right_ee[:, :3],
        "left_gripper": left_gripper.reshape(-1),
        "right_gripper": right_gripper.reshape(-1),
        "qpos14": qpos14,
    }


def _read_first_dataset(h5_file: h5py.File, candidates: list[str]) -> np.ndarray:
    for candidate in candidates:
        if candidate in h5_file:
            return np.asarray(h5_file[candidate][...])
    raise KeyError(f"None of the candidate datasets were found: {candidates}")


def print_lengths(episode: dict[str, np.ndarray]) -> None:
    print("\nLENGTHS")
    for name, value in episode.items():
        if value.ndim > 0:
            print(f"{name}: T={value.shape[0]}")


def print_lerobot_length_compare(lerobot_dir: Path, episode_id: int, raw_len: int) -> None:
    length = find_lerobot_episode_length(lerobot_dir, episode_id)
    print("\nLEROBOT_COMPARE")
    if length is None:
        print(f"episode={episode_id} lerobot_length=<not found> raw_length={raw_len}")
    else:
        print(f"episode={episode_id} lerobot_length={length} raw_length={raw_len} equal={length == raw_len}")


def find_lerobot_episode_length(lerobot_dir: Path, episode_id: int) -> int | None:
    try:
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        print(f"pandas_unavailable_for_lerobot_compare={type(exc).__name__}: {exc}")
        return None
    for parquet_path in sorted(lerobot_dir.rglob("*.parquet")):
        try:
            frame = pd.read_parquet(parquet_path, columns=["episode_index"])
        except Exception:  # noqa: BLE001
            continue
        if "episode_index" not in frame:
            continue
        matches = frame["episode_index"] == episode_id
        if bool(matches.any()):
            return int(matches.sum())
    return None


def save_sidecar(
    path: Path,
    result: dict[str, Any],
    episode: dict[str, np.ndarray],
    cfg: BinaryGateLabelerConfig,
) -> None:
    np.savez_compressed(
        path,
        w_binary=result["w_binary"].astype(np.int64),
        w_raw=result["w_raw"].astype(np.int64),
        d_LR=result["d_LR"],
        v_L=result["v_L"],
        v_R=result["v_R"],
        delta_d=result["delta_d"],
        left_gripper=episode["left_gripper"],
        right_gripper=episode["right_gripper"],
        left_closed=result["left_closed"].astype(np.int64),
        right_closed=result["right_closed"].astype(np.int64),
        both_moving=result["both_moving"].astype(np.int64),
        thresholds=json.dumps(result["thresholds"]),
        config=json.dumps(asdict(cfg)),
    )


def summarize_episode(name: str, result: dict[str, Any]) -> dict[str, Any]:
    w_binary = result["w_binary"].astype(bool)
    lengths = _positive_segment_lengths(w_binary)
    return {
        "episode": name,
        "frames": int(w_binary.size),
        "positive_ratio": float(w_binary.mean()) if w_binary.size else 0.0,
        "segment_count": int(len(lengths)),
        "mean_positive_segment_len": float(np.mean(lengths)) if lengths else 0.0,
        "thresholds": result["thresholds"],
    }


def summarize_all(episodes: list[dict[str, Any]], cfg: BinaryGateLabelerConfig) -> dict[str, Any]:
    ratios = [episode["positive_ratio"] for episode in episodes]
    total_frames = sum(episode["frames"] for episode in episodes)
    threshold_keys = (
        "distance_small_threshold",
        "left_gripper_closed_threshold",
        "right_gripper_closed_threshold",
    )
    threshold_stats = {}
    for key in threshold_keys:
        values = [episode["thresholds"][key] for episode in episodes]
        threshold_stats[key] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    return {
        "config": asdict(cfg),
        "total_episodes": int(len(episodes)),
        "total_frames": int(total_frames),
        "mean_positive_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "positive_ratio_min": float(np.min(ratios)) if ratios else 0.0,
        "positive_ratio_max": float(np.max(ratios)) if ratios else 0.0,
        "per_episode_positive_ratio": ratios,
        "mean_segment_count": float(np.mean([episode["segment_count"] for episode in episodes]))
        if episodes
        else 0.0,
        "mean_positive_segment_len": float(
            np.mean([episode["mean_positive_segment_len"] for episode in episodes])
        )
        if episodes
        else 0.0,
        "threshold_statistics": threshold_stats,
        "episodes": episodes,
    }


def maybe_plot_episode(path: Path, result: dict[str, Any], episode: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable, skip plot {path}: {type(exc).__name__}: {exc}")
        return

    t = np.arange(result["w_binary"].shape[0])
    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, result["w_raw"], label="w_raw", alpha=0.7)
    axes[0].plot(t, result["w_binary"], label="w_binary", alpha=0.7)
    axes[0].legend(loc="upper right")
    axes[1].plot(t, result["d_LR"], label="d_LR")
    axes[1].axhline(result["thresholds"]["distance_small_threshold"], color="r", linestyle="--")
    axes[1].legend(loc="upper right")
    axes[2].plot(t, result["v_L"], label="v_L")
    axes[2].plot(t, result["v_R"], label="v_R")
    axes[2].legend(loc="upper right")
    axes[3].plot(t, episode["left_gripper"], label="left_gripper")
    axes[3].plot(t, episode["right_gripper"], label="right_gripper")
    axes[3].legend(loc="upper right")
    axes[4].plot(t, result["left_closed"].astype(np.int64), label="left_closed")
    axes[4].plot(t, result["right_closed"].astype(np.int64), label="right_closed")
    axes[4].legend(loc="upper right")
    axes[4].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _positive_segment_lengths(mask: np.ndarray) -> list[int]:
    lengths = []
    start = None
    for index, value in enumerate(np.r_[mask.astype(bool), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            lengths.append(index - start)
            start = None
    return lengths


def _episode_sort_key(path: Path) -> tuple[int, str]:
    return (_episode_id(path), path.name)


def _episode_id(path: Path) -> int:
    digits = "".join(char for char in path.stem if char.isdigit())
    return int(digits) if digits else 0


if __name__ == "__main__":
    raise SystemExit(main())
