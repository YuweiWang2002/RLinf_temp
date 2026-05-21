import torch
from torch import nn

from rlinf.algorithms.residual_td3.gate_head import (
    GateHeadMLP,
    GateHeadRuntime,
    save_gate_head_checkpoint,
)


def test_gate_head_mlp_outputs_logits_shape():
    model = GateHeadMLP(feature_dim=8, hidden_dim=16, dropout=0.0)

    logits = model(torch.randn(4, 8))

    assert logits.shape == (4, 1)


def test_gate_head_bce_loss_backpropagates():
    model = GateHeadMLP(feature_dim=8, hidden_dim=16, dropout=0.0)
    loss = nn.BCEWithLogitsLoss()(model(torch.randn(4, 8)), torch.ones(4, 1))

    loss.backward()

    assert all(param.grad is not None for param in model.parameters())


def test_runtime_save_load_and_predict(tmp_path):
    model = GateHeadMLP(feature_dim=4, hidden_dim=8, dropout=0.0)
    checkpoint = tmp_path / "gate_head.pt"
    save_gate_head_checkpoint(
        checkpoint,
        model=model,
        config={"feature_dim": 4, "hidden_dim": 8, "dropout": 0.0},
        threshold=0.4,
    )

    runtime = GateHeadRuntime.load_from_checkpoint(checkpoint, device="cpu")
    prob, gate = runtime.predict_from_feature(torch.zeros(3, 4))

    assert prob.shape == (3, 1)
    assert gate.shape == (3, 1)
    assert torch.all((prob >= 0.0) & (prob <= 1.0))
    assert set(torch.unique(gate).tolist()).issubset({0.0, 1.0})
    assert runtime.cfg.threshold == 0.4


def test_runtime_threshold_override_takes_effect(tmp_path):
    model = GateHeadMLP(feature_dim=2, hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "gate_head.pt"
    save_gate_head_checkpoint(
        checkpoint,
        model=model,
        config={"feature_dim": 2, "hidden_dim": 4, "dropout": 0.0},
        threshold=0.9,
    )

    runtime = GateHeadRuntime.load_from_checkpoint(checkpoint, device="cpu", threshold=0.1)

    assert runtime.cfg.threshold == 0.1
