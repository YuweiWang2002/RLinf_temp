"""Episode logging helpers for residual rollout scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def residual_prediction_stats(records: list[dict[str, object]]) -> dict[str, float | int]:
    """Aggregate predicted/applied residual norms from intervention records."""

    pred_norm = np.asarray([row.get("pred_delta_norm", 0.0) for row in records], dtype=np.float64)
    applied_norm = np.asarray([row.get("applied_delta_norm", 0.0) for row in records], dtype=np.float64)
    saturation = np.asarray([row.get("saturation", 0) for row in records], dtype=np.float64)
    nan_inf = np.asarray([row.get("has_nan_or_inf", 0) for row in records], dtype=np.int64)
    if pred_norm.size == 0:
        return {
            "pred_norm_mean": 0.0,
            "pred_norm_std": 0.0,
            "pred_norm_max": 0.0,
            "pred_norm_p50": 0.0,
            "pred_norm_p90": 0.0,
            "pred_norm_p95": 0.0,
            "pred_norm_p99": 0.0,
            "applied_norm_mean": 0.0,
            "applied_norm_std": 0.0,
            "applied_norm_max": 0.0,
            "applied_norm_p50": 0.0,
            "applied_norm_p90": 0.0,
            "applied_norm_p95": 0.0,
            "applied_norm_p99": 0.0,
            "saturation_ratio": 0.0,
            "nan_inf_count": 0,
        }
    return {
        "pred_norm_mean": float(pred_norm.mean()),
        "pred_norm_std": float(pred_norm.std()),
        "pred_norm_max": float(pred_norm.max()),
        "pred_norm_p50": float(np.percentile(pred_norm, 50)),
        "pred_norm_p90": float(np.percentile(pred_norm, 90)),
        "pred_norm_p95": float(np.percentile(pred_norm, 95)),
        "pred_norm_p99": float(np.percentile(pred_norm, 99)),
        "applied_norm_mean": float(applied_norm.mean()) if applied_norm.size else 0.0,
        "applied_norm_std": float(applied_norm.std()) if applied_norm.size else 0.0,
        "applied_norm_max": float(applied_norm.max()) if applied_norm.size else 0.0,
        "applied_norm_p50": float(np.percentile(applied_norm, 50)) if applied_norm.size else 0.0,
        "applied_norm_p90": float(np.percentile(applied_norm, 90)) if applied_norm.size else 0.0,
        "applied_norm_p95": float(np.percentile(applied_norm, 95)) if applied_norm.size else 0.0,
        "applied_norm_p99": float(np.percentile(applied_norm, 99)) if applied_norm.size else 0.0,
        "saturation_ratio": float(saturation.mean()) if saturation.size else 0.0,
        "nan_inf_count": int(nan_inf.sum()) if nan_inf.size else 0,
    }


def save_hybrid_log(save_dir: Path, episode_id: int, rows: list[dict[str, Any]]) -> Path:
    """Write one episode hybrid decision log in parquet plus CSV form."""

    path = save_dir / f"hybrid_log_episode_{episode_id:04d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}), path)
    csv_path = path.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return path


def save_intervention_records(save_dir: Path, episode_id: int, records: list[dict[str, object]]) -> Path:
    """Write one episode intervention records parquet."""

    path = save_dir / f"intervention_records_episode_{episode_id:04d}.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(records) if records else pa.table({}), path)
    return path


def save_hybrid_plot(save_dir: Path, episode_id: int, rows: list[dict[str, Any]], threshold: float) -> Path | None:
    """Write the optional gate/mode diagnostic plot."""

    if not rows:
        return None
    import matplotlib.pyplot as plt

    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"hybrid_plot_episode_{episode_id:04d}.png"
    x = np.asarray([row["env_step"] for row in rows], dtype=np.int64)
    gate_prob = np.asarray([row["gate_prob"] for row in rows], dtype=np.float32)
    gate = np.asarray([row["gate_binary"] for row in rows], dtype=np.float32)
    mode = np.asarray([1.0 if row["execution_mode"] == "ee16_zero_residual" else 0.0 for row in rows])
    left_gripper = np.asarray([row["obs_state_06"] for row in rows], dtype=np.float32)
    right_gripper = np.asarray([row["obs_state_13"] for row in rows], dtype=np.float32)
    dist = np.asarray([np.nan if row["d_LR"] is None else row["d_LR"] for row in rows], dtype=np.float32)
    success = np.asarray([row["success"] for row in rows], dtype=np.float32)
    done = np.asarray([row["done"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, gate_prob, label="gate_prob")
    axes[0].axhline(threshold, color="tab:red", linestyle="--", label=f"threshold={threshold:g}")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="upper right")
    axes[1].step(x, gate, where="post", label="gate_binary")
    axes[1].step(x, mode, where="post", label="ee16_zero_residual")
    axes[1].legend(loc="upper right")
    axes[2].plot(x, left_gripper, label="left_gripper")
    axes[2].plot(x, right_gripper, label="right_gripper")
    axes[2].legend(loc="upper right")
    axes[3].plot(x, dist, label="d_LR")
    axes[3].scatter(x[done > 0], done[done > 0], marker="x", label="done")
    axes[3].scatter(x[success > 0], success[success > 0], marker="o", label="success")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("env_step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON object, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_summary(metrics: dict[str, Any]) -> None:
    """Print the compact aggregate rollout summary."""

    print("HYBRID SUMMARY")
    print("--------------")
    for key in ("execution_mode", "ee16_execution_strategy", "success_rate", "mean_return", "save_dir"):
        print(f"{key}={metrics[key]}")
