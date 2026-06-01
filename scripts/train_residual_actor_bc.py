"""Train an offline BC baseline for the residual right-xyz actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.residual_td3.residual_actor import (
    ResidualActorConfig,
    ZeroInitResidualActorMLP,
)
from rlinf.algorithms.residual_td3.residual_bc_training import (
    FEATURE_NAMES,
    ResidualBCDatasetConfig,
    ResidualBCNpzDataset,
    iter_numpy_batches,
    split_by_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--residual-horizon-k", type=int, default=10)
    parser.add_argument("--delta-max", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--include-current-qpos14", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--compute-zero-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = train_residual_actor_bc(args)
    print(json.dumps(metrics, indent=2))
    return 0


def train_residual_actor_bc(args: argparse.Namespace) -> dict[str, Any]:
    """Train the BC actor and save checkpoint, metrics, predictions, and plots."""

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ResidualBCNpzDataset(
        args.target_dataset,
        ResidualBCDatasetConfig(
            target_horizon_k=args.residual_horizon_k,
            max_delta_local_xyz=args.delta_max,
            include_current_qpos14=args.include_current_qpos14,
        ),
    )
    all_indices = np.arange(len(dataset), dtype=np.int64)
    if args.sample_limit is not None:
        all_indices = all_indices[: min(args.sample_limit, all_indices.shape[0])]
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    if getattr(args, "eval_only", False):
        if not getattr(args, "checkpoint", None):
            raise ValueError("--checkpoint is required with --eval-only.")
        return evaluate_residual_actor_bc(args, dataset, all_indices, device)

    train_indices, val_indices, split_info = resolve_train_val_split(
        dataset,
        all_indices,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    actor_cfg = ResidualActorConfig(
        obs_dim=dataset.obs_dim,
        hidden_dim=args.hidden_dim,
        chunk_len=dataset.target_horizon_k,
        residual_dim=3,
        delta_max=args.delta_max,
        zero_init_output=True,
    )
    actor = ZeroInitResidualActorMLP(actor_cfg).to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            actor,
            dataset,
            train_indices,
            batch_size=args.batch_size,
            device=device,
            optimizer=optimizer,
            shuffle=True,
            seed=args.seed + epoch,
        )
        val_metrics = run_epoch(
            actor,
            dataset,
            val_indices,
            batch_size=args.batch_size,
            device=device,
            optimizer=None,
            shuffle=False,
            seed=args.seed,
        )
        row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)

    val_predictions = predict(actor, dataset, val_indices, batch_size=args.batch_size, device=device)
    final_metrics = {
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "obs_dim": dataset.obs_dim,
        "target_shape": list(dataset.arrays["target_delta_local_xyz"].shape),
        "train_samples": int(train_indices.shape[0]),
        "val_samples": int(val_indices.shape[0]),
        "train_episodes": split_info["train_episodes"],
        "val_episodes": split_info["val_episodes"],
        "split_source": split_info["split_source"],
        "history": history,
        "final_train": history[-1]["train"] if history else {},
        "final_val": history[-1]["val"] if history else {},
        "val_prediction_metrics": prediction_metrics(
            val_predictions["pred_delta_local_xyz"],
            val_predictions["target_delta_local_xyz"],
            args.delta_max,
        ),
    }

    checkpoint_path = output_dir / "residual_actor_bc.pt"
    torch.save(
        {
            "model_state_dict": actor.state_dict(),
            "actor_config": actor_cfg.__dict__,
            "feature_names": FEATURE_NAMES,
            "metrics": final_metrics,
        },
        checkpoint_path,
    )
    final_metrics["checkpoint_path"] = str(checkpoint_path)
    if getattr(args, "compute_zero_baseline", False):
        comparison = baseline_comparison(
            val_predictions["pred_delta_local_xyz"],
            val_predictions["target_delta_local_xyz"],
            args.delta_max,
        )
        final_metrics["baseline_comparison"] = comparison
        (output_dir / "baseline_comparison.json").write_text(
            json.dumps(comparison, indent=2),
            encoding="utf-8",
        )
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "val_predictions.npz", **val_predictions)
    plot_dir = output_dir / "plots"
    save_plots(plot_dir, val_predictions)
    final_metrics["plots_path"] = str(plot_dir)
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    return final_metrics


def evaluate_residual_actor_bc(
    args: argparse.Namespace,
    dataset: ResidualBCNpzDataset,
    all_indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a saved BC actor on the same episode-level validation split."""

    output_dir = Path(args.output_dir)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_indices, val_indices, split_info = resolve_train_val_split(
        dataset,
        all_indices,
        val_ratio=args.val_ratio,
        seed=args.seed,
        checkpoint=checkpoint,
    )
    actor_cfg = ResidualActorConfig(**checkpoint["actor_config"])
    if actor_cfg.obs_dim != dataset.obs_dim:
        raise ValueError(f"Checkpoint obs_dim={actor_cfg.obs_dim} does not match dataset obs_dim={dataset.obs_dim}.")
    if actor_cfg.chunk_len != dataset.target_horizon_k:
        raise ValueError(
            f"Checkpoint chunk_len={actor_cfg.chunk_len} does not match dataset horizon "
            f"{dataset.target_horizon_k}."
        )
    actor = ZeroInitResidualActorMLP(actor_cfg).to(device)
    actor.load_state_dict(checkpoint["model_state_dict"])
    val_predictions = predict(actor, dataset, val_indices, batch_size=args.batch_size, device=device)
    actor_metrics = prediction_metrics(
        val_predictions["pred_delta_local_xyz"],
        val_predictions["target_delta_local_xyz"],
        args.delta_max,
    )
    comparison = baseline_comparison(
        val_predictions["pred_delta_local_xyz"],
        val_predictions["target_delta_local_xyz"],
        args.delta_max,
    )
    metrics = {
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "obs_dim": dataset.obs_dim,
        "target_shape": list(dataset.arrays["target_delta_local_xyz"].shape),
        "train_samples": int(train_indices.shape[0]),
        "val_samples": int(val_indices.shape[0]),
        "train_episodes": split_info["train_episodes"],
        "val_episodes": split_info["val_episodes"],
        "checkpoint_path": str(args.checkpoint),
        "val_prediction_metrics": actor_metrics,
        "baseline_comparison": comparison,
        "split_source": split_info["split_source"],
    }
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "baseline_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "val_predictions_eval.npz", **val_predictions)
    return metrics


def resolve_train_val_split(
    dataset: ResidualBCNpzDataset,
    all_indices: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
    checkpoint: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Resolve train/val indices, preferring the checkpoint's saved episode split."""

    saved_split = None if checkpoint is None else checkpoint.get("metrics", {})
    if saved_split and saved_split.get("train_episodes") and saved_split.get("val_episodes"):
        train_episodes = {int(ep) for ep in saved_split["train_episodes"]}
        val_episodes = {int(ep) for ep in saved_split["val_episodes"]}
        episode_index = dataset.arrays["episode_index"][all_indices]
        train_mask = np.asarray([int(ep) in train_episodes for ep in episode_index], dtype=bool)
        val_mask = np.asarray([int(ep) in val_episodes for ep in episode_index], dtype=bool)
        train_indices = all_indices[train_mask]
        val_indices = all_indices[val_mask]
        if train_indices.size == 0 or val_indices.size == 0:
            raise ValueError("Checkpoint episode split does not overlap with the requested dataset/sample_limit.")
        return train_indices, val_indices, {
            "train_episodes": sorted(train_episodes),
            "val_episodes": sorted(val_episodes),
            "split_source": "checkpoint_episode_split",
        }

    train_rows, val_rows = split_by_episode(
        dataset.arrays["episode_index"][all_indices],
        val_ratio=val_ratio,
        seed=seed,
    )
    train_indices = all_indices[train_rows]
    val_indices = all_indices[val_rows]
    return train_indices, val_indices, {
        "train_episodes": sorted(np.unique(dataset.arrays["episode_index"][train_indices]).astype(int).tolist()),
        "val_episodes": sorted(np.unique(dataset.arrays["episode_index"][val_indices]).astype(int).tolist()),
        "split_source": "seeded_episode_split",
    }


def run_epoch(
    actor: ZeroInitResidualActorMLP,
    dataset: ResidualBCNpzDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    shuffle: bool,
    seed: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    actor.train(is_train)
    losses = []
    preds = []
    targets = []
    for obs, target in iter_numpy_batches(
        dataset,
        indices,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    ):
        obs = obs.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(is_train):
            pred = actor(obs)
            loss = torch.nn.functional.mse_loss(pred, target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0) if preds else np.empty((0, actor.cfg.chunk_len, 3), dtype=np.float32)
    target_np = (
        np.concatenate(targets, axis=0)
        if targets
        else np.empty((0, actor.cfg.chunk_len, 3), dtype=np.float32)
    )
    metrics = prediction_metrics(pred_np, target_np, actor.cfg.delta_max)
    metrics["mse"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def predict(
    actor: ZeroInitResidualActorMLP,
    dataset: ResidualBCNpzDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    actor.eval()
    preds = []
    targets = []
    obs = []
    with torch.no_grad():
        for obs_batch, target_batch in iter_numpy_batches(
            dataset,
            indices,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            pred = actor(obs_batch.to(device)).detach().cpu().numpy()
            preds.append(pred)
            targets.append(target_batch.numpy())
            obs.append(obs_batch.numpy())
    return {
        "obs_vector": np.concatenate(obs, axis=0),
        "pred_delta_local_xyz": np.concatenate(preds, axis=0),
        "target_delta_local_xyz": np.concatenate(targets, axis=0),
        "episode_index": dataset.arrays["episode_index"][indices],
        "frame_index": dataset.arrays["frame_index"][indices],
    }


def prediction_metrics(pred: np.ndarray, target: np.ndarray, delta_max: float) -> dict[str, float]:
    err = pred - target
    pred_norm = np.linalg.norm(pred, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    err_norm = np.linalg.norm(err, axis=-1)
    abs_pred = np.abs(pred)
    eps = 1e-12
    axis_mae = np.mean(np.abs(err.reshape(-1, 3)), axis=0) if err.size else np.zeros(3, dtype=np.float32)
    return {
        "mse": float(np.mean(err**2)) if err.size else 0.0,
        "pred_norm_mean": float(pred_norm.mean()) if pred_norm.size else 0.0,
        "pred_norm_std": float(pred_norm.std()) if pred_norm.size else 0.0,
        "pred_norm_max": float(pred_norm.max()) if pred_norm.size else 0.0,
        "target_norm_mean": float(target_norm.mean()) if target_norm.size else 0.0,
        "target_norm_std": float(target_norm.std()) if target_norm.size else 0.0,
        "target_norm_max": float(target_norm.max()) if target_norm.size else 0.0,
        "error_norm_mean": float(err_norm.mean()) if err_norm.size else 0.0,
        "error_norm_std": float(err_norm.std()) if err_norm.size else 0.0,
        "error_norm_max": float(err_norm.max()) if err_norm.size else 0.0,
        "error_lt_1mm": float((err_norm < 0.001).mean()) if err_norm.size else 0.0,
        "error_lt_2mm": float((err_norm < 0.002).mean()) if err_norm.size else 0.0,
        "error_lt_5mm": float((err_norm < 0.005).mean()) if err_norm.size else 0.0,
        "per_axis_mae": axis_mae.astype(float).tolist(),
        "clip_saturation_ratio": float((abs_pred >= (delta_max - eps)).any(axis=-1).mean())
        if pred.size and delta_max > 0
        else 0.0,
    }


def baseline_comparison(pred: np.ndarray, target: np.ndarray, delta_max: float) -> dict[str, Any]:
    """Compare trained BC predictions with the zero predictor on the same targets."""

    zero_pred = np.zeros_like(target, dtype=np.float32)
    actor_metrics = prediction_metrics(pred, target, delta_max)
    zero_metrics = prediction_metrics(zero_pred, target, delta_max)
    improvement = {
        "mse_improvement_pct": improvement_pct(zero_metrics["mse"], actor_metrics["mse"]),
        "mean_error_norm_improvement_pct": improvement_pct(
            zero_metrics["error_norm_mean"],
            actor_metrics["error_norm_mean"],
        ),
        "error_lt_1mm_improvement_pct": improvement_pct(
            zero_metrics["error_lt_1mm"],
            actor_metrics["error_lt_1mm"],
            higher_is_better=True,
        ),
        "error_lt_2mm_improvement_pct": improvement_pct(
            zero_metrics["error_lt_2mm"],
            actor_metrics["error_lt_2mm"],
            higher_is_better=True,
        ),
        "error_lt_5mm_improvement_pct": improvement_pct(
            zero_metrics["error_lt_5mm"],
            actor_metrics["error_lt_5mm"],
            higher_is_better=True,
        ),
        "error_lt_1mm_improvement_pp": 100.0 * (actor_metrics["error_lt_1mm"] - zero_metrics["error_lt_1mm"]),
        "error_lt_2mm_improvement_pp": 100.0 * (actor_metrics["error_lt_2mm"] - zero_metrics["error_lt_2mm"]),
        "error_lt_5mm_improvement_pp": 100.0 * (actor_metrics["error_lt_5mm"] - zero_metrics["error_lt_5mm"]),
    }
    return {
        "zero_predictor": zero_metrics,
        "trained_bc_actor": actor_metrics,
        "improvement": improvement,
    }


def improvement_pct(baseline: float, candidate: float, *, higher_is_better: bool = False) -> float | None:
    """Return percent improvement from baseline to candidate."""

    if abs(baseline) < 1e-12:
        return None
    if higher_is_better:
        return float((candidate - baseline) / baseline * 100.0)
    return float((baseline - candidate) / baseline * 100.0)


def save_plots(plot_dir: Path, predictions: dict[str, np.ndarray]) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        (plot_dir / "plots_skipped.txt").write_text("matplotlib is not installed\n", encoding="utf-8")
        return

    pred = predictions["pred_delta_local_xyz"]
    target = predictions["target_delta_local_xyz"]
    error_norm = np.linalg.norm(pred - target, axis=-1)
    target_flat = target.reshape(-1, 3)
    pred_flat = pred.reshape(-1, 3)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, name in enumerate(("x", "y", "z")):
        axes[axis].hist(target_flat[:, axis], bins=60)
        axes[axis].set_title(f"target {name}")
    fig.tight_layout()
    fig.savefig(plot_dir / "target_delta_local_xyz_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, name in enumerate(("x", "y", "z")):
        axes[axis].scatter(target_flat[:, axis], pred_flat[:, axis], s=4, alpha=0.35)
        axes[axis].set_xlabel("target")
        axes[axis].set_ylabel("pred")
        axes[axis].set_title(name)
    fig.tight_layout()
    fig.savefig(plot_dir / "pred_vs_target_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(error_norm.reshape(-1), bins=80)
    ax.set_xlabel("error norm (m)")
    fig.tight_layout()
    fig.savefig(plot_dir / "error_norm_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(error_norm.shape[1]), error_norm.mean(axis=0), marker="o")
    ax.set_xlabel("horizon step")
    ax.set_ylabel("mean error norm (m)")
    fig.tight_layout()
    fig.savefig(plot_dir / "error_by_horizon_step.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
