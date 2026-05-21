"""Audit raw RoboTwin gate labels against LeRobot parquet frame alignment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq

DEFAULT_RAW_DIR = "/nfs/data3/rlinf_data/handover/raw_collect/handover_block/data/"
DEFAULT_GATE_DIR = "/nfs/data3/rlinf_data/handover/residual_gate_labels_binary/"
DEFAULT_LEROBOT_DIR = "/nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--gate-dir", default=DEFAULT_GATE_DIR)
    parser.add_argument("--lerobot-dir", default=DEFAULT_LEROBOT_DIR)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--dump-examples", action="store_true")
    parser.add_argument("--offset-min", type=int, default=-20)
    parser.add_argument("--offset-max", type=int, default=20)
    parser.add_argument(
        "--same-index-only",
        action="store_true",
        help="Only compare LeRobot episode i with raw episode i. By default the script searches all raw episodes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    gate_dir = Path(args.gate_dir)
    lerobot_dir = Path(args.lerobot_dir)
    episode_paths = sorted(lerobot_dir.glob("data/chunk-*/episode_*.parquet"), key=_episode_sort_key)
    if args.episode_limit is not None:
        episode_paths = episode_paths[: args.episode_limit]
    if not episode_paths:
        raise FileNotFoundError(f"No LeRobot episode parquet files found under {lerobot_dir}.")

    raw_episodes = load_raw_episodes(raw_dir)
    episodes = []
    for parquet_path in episode_paths:
        episode_index = _episode_id(parquet_path)
        raw_candidates = (
            [raw_episodes[episode_index]]
            if args.same_index_only
            else raw_episodes
        )
        episode = audit_episode(
            episode_index=episode_index,
            raw_candidates=raw_candidates,
            gate_dir=gate_dir,
            parquet_path=parquet_path,
            offset_min=args.offset_min,
            offset_max=args.offset_max,
            dump_examples=args.dump_examples,
        )
        episodes.append(episode)

    summary = summarize_alignment(episodes)
    output = {"summary": summary, "episodes": episodes}
    text = json.dumps(output, indent=2)
    print(text)
    out_json = Path(args.out_json) if args.out_json else gate_dir / "lerobot_alignment_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(text, encoding="utf-8")
    print(f"WROTE {out_json}")
    return 0


@dataclass(frozen=True)
class RawEpisode:
    episode_index: int
    path: Path
    joint_action: np.ndarray


def audit_episode(
    *,
    episode_index: int,
    raw_candidates: list[RawEpisode],
    gate_dir: Path,
    parquet_path: Path,
    offset_min: int,
    offset_max: int,
    dump_examples: bool,
) -> dict[str, Any]:
    lerobot = read_lerobot_episode(parquet_path)
    lr_action = lerobot["action"]
    lr_state = lerobot["observation.state"]
    source_raw, state_match = find_best_source_raw(
        lr_state,
        raw_candidates,
        offset_min,
        offset_max,
    )
    raw_action = source_raw.joint_action
    gate_path = gate_dir / f"episode{source_raw.episode_index:06d}_gate.npz"
    gate = read_gate(gate_path)

    action_match = find_best_offset(lr_action, raw_action, offset_min, offset_max)
    trim_start = state_match["offset"]
    trim_end = raw_action.shape[0] - trim_start - lr_state.shape[0]
    action_expected_from_state = state_match["offset"] + 1
    action_at_state_plus_one = score_offset(lr_action, raw_action, action_expected_from_state)

    result: dict[str, Any] = {
        "episode_index": int(episode_index),
        "source_raw_episode_index": int(source_raw.episode_index),
        "raw_path": str(source_raw.path),
        "gate_path": str(gate_path),
        "lerobot_path": str(parquet_path),
        "T_raw": int(raw_action.shape[0]),
        "T_gate": int(gate.shape[0]),
        "T_lr": int(lr_state.shape[0]),
        "best_action_offset": int(action_match["offset"]),
        "best_state_offset": int(state_match["offset"]),
        "mean_action_error": float(action_match["mean_error"]),
        "max_action_error": float(action_match["max_error"]),
        "mean_state_error": float(state_match["mean_error"]),
        "max_state_error": float(state_match["max_error"]),
        "action_state_plus_one_offset": int(action_expected_from_state),
        "action_state_plus_one_mean_error": float(action_at_state_plus_one["mean_error"]),
        "action_state_plus_one_max_error": float(action_at_state_plus_one["max_error"]),
        "inferred_trim_start": int(trim_start),
        "inferred_trim_end": int(trim_end),
        "action_compare_len": int(action_match["compare_len"]),
        "state_compare_len": int(state_match["compare_len"]),
        "lengths_ok_for_mapping": bool(gate.shape[0] == raw_action.shape[0] and trim_end >= 0),
        "frame_index_start": int(lerobot["frame_index"][0]) if lerobot["frame_index"].size else None,
        "frame_index_end": int(lerobot["frame_index"][-1]) if lerobot["frame_index"].size else None,
    }
    if dump_examples:
        result["examples"] = {
            "raw_first3": raw_action[:3].tolist(),
            "lr_state_first3": lr_state[:3].tolist(),
            "lr_action_first3": lr_action[:3].tolist(),
            "gate_first10": gate[:10].astype(int).tolist(),
        }
    return result


def load_raw_episodes(raw_dir: Path) -> list[RawEpisode]:
    episodes = []
    for path in sorted(raw_dir.glob("episode*.hdf5"), key=_episode_sort_key):
        episodes.append(
            RawEpisode(
                episode_index=_episode_id(path),
                path=path,
                joint_action=read_raw_joint_action(path),
            )
        )
    if not episodes:
        raise FileNotFoundError(f"No raw episode*.hdf5 files found under {raw_dir}.")
    return episodes


def read_raw_joint_action(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5_file:
        return np.asarray(h5_file["/joint_action/vector"][...], dtype=np.float64)


def read_gate(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["w_binary"], dtype=np.int64)


def read_lerobot_episode(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(
        path,
        columns=["episode_index", "frame_index", "observation.state", "action"],
    )
    data = table.to_pydict()
    return {
        "episode_index": np.asarray(data["episode_index"], dtype=np.int64),
        "frame_index": np.asarray(data["frame_index"], dtype=np.int64),
        "observation.state": np.asarray(data["observation.state"], dtype=np.float64),
        "action": np.asarray(data["action"], dtype=np.float64),
    }


def find_best_offset(
    lerobot_values: np.ndarray,
    raw_values: np.ndarray,
    offset_min: int,
    offset_max: int,
) -> dict[str, float | int]:
    scores = [score_offset(lerobot_values, raw_values, offset) for offset in range(offset_min, offset_max + 1)]
    valid_scores = [score for score in scores if score["compare_len"] > 0]
    if not valid_scores:
        raise ValueError("No valid offset comparisons.")
    return min(valid_scores, key=lambda item: (item["mean_error"], item["max_error"], -item["compare_len"]))


def find_best_source_raw(
    lr_state: np.ndarray,
    raw_candidates: list[RawEpisode],
    offset_min: int,
    offset_max: int,
) -> tuple[RawEpisode, dict[str, float | int]]:
    best_episode = None
    best_score = None
    for raw_episode in raw_candidates:
        score = find_best_offset(lr_state, raw_episode.joint_action, offset_min, offset_max)
        full_overlap = int(score["compare_len"]) == lr_state.shape[0]
        ranking = (
            0 if full_overlap else 1,
            float(score["mean_error"]),
            float(score["max_error"]),
            -int(score["compare_len"]),
        )
        if best_score is None or ranking < best_score:
            best_episode = raw_episode
            best_score = ranking
            best_match = score
    if best_episode is None:
        raise ValueError("No source raw episode candidates found.")
    return best_episode, best_match


def score_offset(lerobot_values: np.ndarray, raw_values: np.ndarray, offset: int) -> dict[str, float | int]:
    lr_start = max(0, -offset)
    raw_start = max(0, offset)
    compare_len = min(lerobot_values.shape[0] - lr_start, raw_values.shape[0] - raw_start)
    if compare_len <= 0:
        return {"offset": offset, "mean_error": float("inf"), "max_error": float("inf"), "compare_len": 0}
    lr_slice = lerobot_values[lr_start : lr_start + compare_len]
    raw_slice = raw_values[raw_start : raw_start + compare_len]
    errors = np.linalg.norm(lr_slice - raw_slice, axis=-1)
    return {
        "offset": int(offset),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "compare_len": int(compare_len),
    }


def summarize_alignment(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    state_offsets = [episode["best_state_offset"] for episode in episodes]
    action_offsets = [episode["best_action_offset"] for episode in episodes]
    source_raw_episode_indices = [episode["source_raw_episode_index"] for episode in episodes]
    trim_starts = [episode["inferred_trim_start"] for episode in episodes]
    trim_ends = [episode["inferred_trim_end"] for episode in episodes]
    state_errors = [episode["mean_state_error"] for episode in episodes]
    action_errors = [episode["mean_action_error"] for episode in episodes]
    return {
        "total_episodes": int(len(episodes)),
        "source_raw_episode_indices_are_unique": len(set(source_raw_episode_indices)) == len(source_raw_episode_indices),
        "state_offsets": _value_counts(state_offsets),
        "action_offsets": _value_counts(action_offsets),
        "trim_starts": _value_counts(trim_starts),
        "trim_ends": _value_counts(trim_ends),
        "mean_state_error_max": float(np.max(state_errors)) if state_errors else 0.0,
        "mean_action_error_max": float(np.max(action_errors)) if action_errors else 0.0,
        "all_lengths_ok_for_mapping": bool(all(episode["lengths_ok_for_mapping"] for episode in episodes)),
        "all_state_offsets_same": len(set(state_offsets)) == 1,
        "all_action_offsets_same": len(set(action_offsets)) == 1,
        "all_trim_same": len(set(zip(trim_starts, trim_ends, strict=False))) == 1,
    }


def _value_counts(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _episode_sort_key(path: Path) -> tuple[int, str]:
    return (_episode_id(path), path.name)


def _episode_id(path: Path) -> int:
    digits = "".join(char for char in path.stem if char.isdigit())
    return int(digits) if digits else 0


if __name__ == "__main__":
    raise SystemExit(main())
