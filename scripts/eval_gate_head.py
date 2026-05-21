"""Evaluate and plot a trained residual GateHead on frozen VLA features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.residual_td3.gate_head import GateHeadRuntime
from scripts.train_gate_head_from_features import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--plot-limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val = load_features(Path(args.feature_dir) / "features_val.npz")
    runtime = GateHeadRuntime.load_from_checkpoint(
        args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )
    prob, gate = runtime.predict_from_feature(torch.from_numpy(val["z"]))
    prob_np = prob.detach().cpu().numpy().reshape(-1)
    gate_np = gate.detach().cpu().numpy().reshape(-1)
    metrics = classification_metrics(val["y"].reshape(-1), prob_np, runtime.cfg.threshold)
    metrics["threshold"] = runtime.cfg.threshold
    per_episode = compute_per_episode_metrics(val, prob_np, runtime.cfg.threshold)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "per_episode_metrics.json", per_episode)
    write_predictions(output_dir / "predictions.csv", val, prob_np, gate_np)
    maybe_plot(output_dir / "plots", val, prob_np, runtime.cfg.threshold, args.plot_limit)
    print(json.dumps(metrics, indent=2))
    return 0


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compute_per_episode_metrics(
    val: dict[str, np.ndarray],
    prob: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for episode_index in sorted(set(val["episode_index"].astype(int).tolist())):
        mask = val["episode_index"] == episode_index
        rows.append(
            {
                "episode_index": int(episode_index),
                **classification_metrics(val["y"][mask].reshape(-1), prob[mask], threshold),
            }
        )
    return rows


def write_predictions(path: Path, val: dict[str, np.ndarray], prob: np.ndarray, gate: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_index", "frame_index", "label", "prob", "gate_pred"])
        for ep, frame, label, p, g in zip(
            val["episode_index"],
            val["frame_index"],
            val["y"].reshape(-1),
            prob,
            gate,
            strict=True,
        ):
            writer.writerow([int(ep), int(frame), float(label), float(p), float(g)])


def maybe_plot(
    plots_dir: Path,
    val: dict[str, np.ndarray],
    prob: np.ndarray,
    threshold: float,
    plot_limit: int,
) -> None:
    if plot_limit <= 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable, skip plots: {type(exc).__name__}: {exc}")
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    for episode_index in sorted(set(val["episode_index"].astype(int).tolist()))[:plot_limit]:
        mask = val["episode_index"] == episode_index
        frames = val["frame_index"][mask]
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(frames, val["y"][mask].reshape(-1), label="gt")
        axes[0].plot(frames, prob[mask], label="pred_prob")
        axes[0].axhline(threshold, color="r", linestyle="--", label="threshold")
        axes[0].legend(loc="upper right")
        if "state" in val:
            state = val["state"][mask]
            axes[1].plot(frames, state[:, 6], label="state left_gripper")
            axes[1].plot(frames, state[:, 13], label="state right_gripper")
            axes[1].legend(loc="upper right")
        axes[2].plot(frames, prob[mask] >= threshold, label="pred_gate")
        axes[2].legend(loc="upper right")
        axes[2].set_xlabel("frame_index")
        fig.tight_layout()
        fig.savefig(plots_dir / f"episode_{episode_index:06d}.png")
        plt.close(fig)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
