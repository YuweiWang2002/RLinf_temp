"""Train a lightweight residual GateHead MLP from frozen VLA features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rlinf.algorithms.residual_td3.gate_head import (
    GateHeadMLP,
    save_gate_head_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_features(feature_dir / "features_train.npz")
    val = load_features(feature_dir / "features_val.npz")
    feature_config = load_json(feature_dir / "feature_config.json")
    device = torch.device(args.device)
    model = GateHeadMLP(
        feature_dim=train["z"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    pos_weight = None
    if args.use_pos_weight:
        positives = float(train["y"].sum())
        negatives = float(train["y"].shape[0] - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train["z"]), torch.from_numpy(train["y"])),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    best_threshold = 0.5
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for z, y in loader:
            z = z.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(z), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate_arrays(model, val["z"], val["y"], device=device, threshold=0.5)
        metrics["train_loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        sweep = threshold_sweep(model, val["z"], val["y"], device=device)
        epoch_best = max(sweep, key=lambda item: item["f1"])
        if epoch_best["f1"] > best_f1:
            best_f1 = epoch_best["f1"]
            best_threshold = epoch_best["threshold"]
            best_metrics = {**metrics, "best_threshold": best_threshold, "best_threshold_f1": best_f1}
            save_gate_head_checkpoint(
                output_dir / "gate_head.pt",
                model=model,
                config={
                    "feature_dim": int(train["z"].shape[1]),
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "feature_config": feature_config,
                },
                metrics=best_metrics,
                threshold=best_threshold,
            )
        print(json.dumps(metrics))

    final_sweep = threshold_sweep(model, val["z"], val["y"], device=device)
    write_json(output_dir / "metrics.json", {"history": history, "best": best_metrics})
    write_json(output_dir / "threshold_sweep.json", final_sweep)
    write_json(
        output_dir / "gate_head_config.json",
        {
            "feature_dim": int(train["z"].shape[1]),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "use_pos_weight": args.use_pos_weight,
            "feature_config": feature_config,
            "best_threshold": best_threshold,
        },
    )
    save_predictions(output_dir / "val_predictions.csv", model, val, device=device)
    print(json.dumps({"best": best_metrics, "checkpoint": str(output_dir / "gate_head.pt")}, indent=2))
    return 0


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) if key in {"z", "y", "state"} else data[key] for key in data.files}


def evaluate_arrays(
    model: GateHeadMLP,
    z: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(z).to(device=device, dtype=torch.float32))
        prob = torch.sigmoid(logits).detach().cpu().numpy()
    return classification_metrics(y.reshape(-1), prob.reshape(-1), threshold)


def threshold_sweep(model: GateHeadMLP, z: np.ndarray, y: np.ndarray, *, device: torch.device) -> list[dict[str, float]]:
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(z).to(device=device, dtype=torch.float32))).cpu().numpy()
    return [
        {"threshold": float(threshold), **classification_metrics(y.reshape(-1), prob.reshape(-1), threshold)}
        for threshold in np.linspace(0.1, 0.9, 9)
    ]


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= threshold
    truth = y_true >= 0.5
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    tn = float(np.logical_and(~pred, ~truth).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    metrics = {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_ratio_gt": float(truth.mean()),
        "positive_ratio_pred": float(pred.mean()),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        metrics["pr_auc"] = float(average_precision_score(truth.astype(int), prob))
        metrics["roc_auc"] = float(roc_auc_score(truth.astype(int), prob))
    except Exception:  # noqa: BLE001
        pass
    return metrics


def save_predictions(path: Path, model: GateHeadMLP, val: dict[str, np.ndarray], *, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(val["z"]).to(device=device, dtype=torch.float32))).cpu().numpy()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_index", "frame_index", "label", "prob"])
        for ep, frame, label, pred in zip(
            val["episode_index"],
            val["frame_index"],
            val["y"].reshape(-1),
            prob.reshape(-1),
            strict=True,
        ):
            writer.writerow([int(ep), int(frame), float(label), float(pred)])


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
