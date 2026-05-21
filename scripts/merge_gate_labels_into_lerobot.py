"""Merge aligned residual gate labels into a LeRobot dataset copy."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_LEROBOT_DIR = "/nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert/"
DEFAULT_GATE_DIR = "/nfs/data3/rlinf_data/handover/residual_gate_labels_binary/"
DEFAULT_ALIGNMENT_JSON = "/nfs/data3/rlinf_data/handover/residual_gate_labels_binary/lerobot_alignment_audit.json"
DEFAULT_OUT_DIR = "/nfs/data3/rlinf_data/lerobot_cache/huggingface/lerobot/handover_expert_with_gate/"
GATE_FIELD = "observation.residual_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-dir", default=DEFAULT_LEROBOT_DIR)
    parser.add_argument("--gate-dir", default=DEFAULT_GATE_DIR)
    parser.add_argument("--alignment-json", default=DEFAULT_ALIGNMENT_JSON)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--plot-limit", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--state-error-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lerobot_dir = Path(args.lerobot_dir)
    gate_dir = Path(args.gate_dir)
    out_dir = Path(args.out_dir)
    alignment = load_alignment(Path(args.alignment_json))
    validate_alignment(alignment, args.state_error_tolerance)

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)
    shutil.copytree(lerobot_dir, out_dir)

    rows = []
    for episode in alignment["episodes"]:
        episode_index = int(episode["episode_index"])
        src_rel = Path(episode["lerobot_path"]).relative_to(lerobot_dir)
        dst_path = out_dir / src_rel
        gate = read_aligned_gate(gate_dir, episode)
        raw_index = np.arange(
            episode["inferred_trim_start"],
            episode["inferred_trim_start"] + episode["T_lr"],
            dtype=np.int64,
        )
        table = pq.read_table(dst_path)
        table = append_gate_column(table, gate)
        pq.write_table(table, dst_path)
        frame_index = table.column("frame_index").to_numpy().astype(np.int64)
        rows.append(
            pa.table(
                {
                    "episode_index": np.full(gate.shape[0], episode_index, dtype=np.int64),
                    "source_raw_episode_index": np.full(
                        gate.shape[0],
                        int(episode["source_raw_episode_index"]),
                        dtype=np.int64,
                    ),
                    "frame_index": frame_index,
                    "raw_index": raw_index,
                    "residual_gate": gate.astype(np.float32),
                }
            )
        )

    update_info_json(out_dir / "meta" / "info.json")
    sidecar = pa.concat_tables(rows) if rows else pa.table({})
    pq.write_table(sidecar, out_dir / "meta" / "residual_gate.parquet")
    maybe_plot(out_dir, sidecar, args.plot_limit)
    summary = verify_output(out_dir)
    print(json.dumps(summary, indent=2))
    return 0


def load_alignment(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_alignment(alignment: dict[str, Any], state_error_tolerance: float) -> None:
    for episode in alignment["episodes"]:
        if not episode["lengths_ok_for_mapping"]:
            raise ValueError(f"Episode {episode['episode_index']} is not length-safe for mapping.")
        if episode["mean_state_error"] > state_error_tolerance:
            raise ValueError(
                f"Episode {episode['episode_index']} mean_state_error="
                f"{episode['mean_state_error']} exceeds {state_error_tolerance}."
            )


def read_aligned_gate(gate_dir: Path, episode: dict[str, Any]) -> np.ndarray:
    episode_index = int(episode["episode_index"])
    source_raw_episode_index = int(episode["source_raw_episode_index"])
    gate_path = gate_dir / f"episode{source_raw_episode_index:06d}_gate.npz"
    with np.load(gate_path) as data:
        gate = np.asarray(data["w_binary"], dtype=np.float32)
    start = int(episode["inferred_trim_start"])
    stop = start + int(episode["T_lr"])
    aligned = gate[start:stop]
    if aligned.shape[0] != int(episode["T_lr"]):
        raise ValueError(
            f"Episode {episode_index}: aligned gate length {aligned.shape[0]} "
            f"!= T_lr {episode['T_lr']}."
        )
    return aligned


def append_gate_column(table: pa.Table, gate: np.ndarray) -> pa.Table:
    if GATE_FIELD in table.column_names:
        index = table.column_names.index(GATE_FIELD)
        table = table.remove_column(index)
    values = pa.array(gate.astype(np.float32), type=pa.float32())
    gate_column = pa.FixedSizeListArray.from_arrays(values, 1)
    return table.append_column(GATE_FIELD, gate_column)


def update_info_json(path: Path) -> None:
    info = json.loads(path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    features[GATE_FIELD] = {
        "dtype": "float32",
        "shape": [1],
        "names": [["residual_gate"]],
    }
    path.write_text(json.dumps(info, indent=4), encoding="utf-8")


def maybe_plot(out_dir: Path, sidecar: pa.Table, plot_limit: int) -> None:
    if plot_limit <= 0 or sidecar.num_rows == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable, skip gate alignment plots: {type(exc).__name__}: {exc}")
        return

    plots_dir = out_dir / "gate_alignment_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data = sidecar.to_pydict()
    episode_indices = np.asarray(data["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(data["frame_index"], dtype=np.int64)
    raw_indices = np.asarray(data["raw_index"], dtype=np.int64)
    gates = np.asarray(data["residual_gate"], dtype=np.float32)
    for episode_index in sorted(set(episode_indices.tolist()))[:plot_limit]:
        mask = episode_indices == episode_index
        parquet_path = out_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        episode_table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        episode_data = episode_table.to_pydict()
        state = np.asarray(episode_data["observation.state"], dtype=np.float32)
        action = np.asarray(episode_data["action"], dtype=np.float32)

        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(frame_indices[mask], gates[mask], label="residual_gate")
        axes[0].legend(loc="upper right")
        axes[1].plot(frame_indices[mask], action[:, 6], label="action left_gripper")
        axes[1].plot(frame_indices[mask], action[:, 13], label="action right_gripper")
        axes[1].legend(loc="upper right")
        axes[2].plot(frame_indices[mask], state[:, 6], label="state left_gripper")
        axes[2].plot(frame_indices[mask], state[:, 13], label="state right_gripper")
        axes[2].legend(loc="upper right")
        axes[3].plot(frame_indices[mask], raw_indices[mask], label="raw_index")
        axes[3].legend(loc="upper right")
        axes[3].set_xlabel("LeRobot frame_index")
        fig.tight_layout()
        fig.savefig(plots_dir / f"episode_{episode_index:06d}.png")
        plt.close(fig)


def verify_output(out_dir: Path) -> dict[str, Any]:
    info = json.loads((out_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    sidecar = pq.read_table(out_dir / "meta" / "residual_gate.parquet")
    gate_values = sidecar.column("residual_gate").to_numpy()
    parquet_paths = sorted(out_dir.glob("data/chunk-*/episode_*.parquet"))
    has_gate = []
    for path in parquet_paths:
        schema = pq.ParquetFile(path).schema_arrow
        has_gate.append(GATE_FIELD in schema.names)
    return {
        "out_dir": str(out_dir),
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "sidecar_rows": int(sidecar.num_rows),
        "parquet_count": int(len(parquet_paths)),
        "all_parquets_have_gate": bool(all(has_gate)),
        "unique_gate": sorted(np.unique(gate_values).astype(float).tolist()),
        "positive_ratio": float(np.mean(gate_values)) if gate_values.size else 0.0,
        "feature": info["features"][GATE_FIELD],
    }


if __name__ == "__main__":
    raise SystemExit(main())
