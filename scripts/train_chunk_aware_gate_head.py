"""Train a chunk-aware GateHead over pi05 hidden features and action chunks."""

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
    ChunkAwareGateHead,
    save_chunk_aware_gate_checkpoint,
)
from scripts.train_gate_head_from_features import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-len", type=int, default=50)
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
    validate_feature_shapes(train, args.chunk_len)
    validate_feature_shapes(val, args.chunk_len)

    device = torch.device(args.device)
    model = ChunkAwareGateHead(
        z_dim=train["z"].shape[1],
        action_dim=train["action_chunk"].shape[-1],
        chunk_len=args.chunk_len,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    pos_weight = None
    if args.use_pos_weight:
        positives = float(train["gate_scalar"].sum())
        negatives = float(train["gate_scalar"].shape[0] - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train["z"]),
            torch.from_numpy(train["action_chunk"]),
            torch.from_numpy(train["gate_scalar"]),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_f1 = -1.0
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {}
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for z, action_chunk, label in loader:
            z = z.to(device=device, dtype=torch.float32)
            action_chunk = action_chunk.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(z, action_chunk), label)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate_arrays(model, val, device=device, threshold=0.5)
        metrics["train_loss"] = float(np.mean(losses)) if losses else 0.0
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        sweep = threshold_sweep(model, val, device=device)
        epoch_best = max(sweep, key=lambda item: item["f1"])
        if epoch_best["f1"] > best_f1:
            best_f1 = epoch_best["f1"]
            best_threshold = epoch_best["threshold"]
            best_metrics = {
                **metrics,
                "best_threshold": best_threshold,
                "best_threshold_f1": best_f1,
            }
            save_chunk_aware_gate_checkpoint(
                output_dir / "chunk_aware_gate_head.pt",
                model=model,
                config={
                    "z_dim": int(train["z"].shape[1]),
                    "action_dim": int(train["action_chunk"].shape[-1]),
                    "chunk_len": args.chunk_len,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "feature_config": feature_config,
                },
                metrics=best_metrics,
                threshold=best_threshold,
            )
        print(json.dumps(metrics))

    final_sweep = threshold_sweep(model, val, device=device)
    write_json(output_dir / "metrics.json", {"history": history, "best": best_metrics})
    write_json(output_dir / "threshold_sweep.json", final_sweep)
    write_json(
        output_dir / "config.json",
        {
            "z_dim": int(train["z"].shape[1]),
            "action_dim": int(train["action_chunk"].shape[-1]),
            "chunk_len": args.chunk_len,
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
    print(
        json.dumps(
            {
                "best": best_metrics,
                "checkpoint": str(output_dir / "chunk_aware_gate_head.pt"),
            },
            indent=2,
        )
    )
    return 0


def load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            key: data[key].astype(np.float32)
            if key in {"z", "action_chunk", "gate_seq", "gate_scalar"}
            else data[key]
            for key in data.files
        }


def validate_feature_shapes(data: dict[str, np.ndarray], chunk_len: int) -> None:
    if data["z"].ndim != 2:
        raise ValueError(f"z must have shape [N, Z], got {data['z'].shape}.")
    if data["action_chunk"].ndim != 3 or data["action_chunk"].shape[1:] != (chunk_len, 14):
        raise ValueError(
            "action_chunk must have shape "
            f"[N, {chunk_len}, 14], got {data['action_chunk'].shape}."
        )
    if data["gate_scalar"].ndim != 2 or data["gate_scalar"].shape[-1] != 1:
        raise ValueError(f"gate_scalar must have shape [N, 1], got {data['gate_scalar'].shape}.")


def evaluate_arrays(
    model: ChunkAwareGateHead,
    data: dict[str, np.ndarray],
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    prob = predict_prob(model, data, device=device)
    label = data["gate_scalar"].reshape(-1)
    metrics = classification_metrics(label, prob.reshape(-1), threshold)
    logits = np.log(np.clip(prob, 1e-12, 1.0 - 1e-12)) - np.log(
        np.clip(1.0 - prob, 1e-12, 1.0)
    )
    loss = nn.BCEWithLogitsLoss()(
        torch.from_numpy(logits.reshape(-1, 1)).float(),
        torch.from_numpy(data["gate_scalar"]).float(),
    )
    metrics["loss"] = float(loss)
    return metrics


def threshold_sweep(
    model: ChunkAwareGateHead,
    data: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> list[dict[str, float]]:
    prob = predict_prob(model, data, device=device)
    return [
        {
            "threshold": float(threshold),
            **classification_metrics(data["gate_scalar"].reshape(-1), prob.reshape(-1), threshold),
        }
        for threshold in np.linspace(0.1, 0.9, 9)
    ]


def predict_prob(
    model: ChunkAwareGateHead,
    data: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, data["z"].shape[0], batch_size):
            z = torch.from_numpy(data["z"][start : start + batch_size]).to(device=device, dtype=torch.float32)
            action_chunk = torch.from_numpy(data["action_chunk"][start : start + batch_size]).to(
                device=device,
                dtype=torch.float32,
            )
            probs.append(torch.sigmoid(model(z, action_chunk)).detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.empty((0, 1), dtype=np.float32)


def save_predictions(
    path: Path,
    model: ChunkAwareGateHead,
    val: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> None:
    prob = predict_prob(model, val, device=device)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_index", "frame_index", "label", "prob"])
        for ep, frame, label, pred in zip(
            val["episode_index"],
            val["frame_index"],
            val["gate_scalar"].reshape(-1),
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
