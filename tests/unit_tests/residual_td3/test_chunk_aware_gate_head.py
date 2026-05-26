import pytest
import torch
from torch import nn

from rlinf.algorithms.residual_td3.gate_head import (
    ActionChunkEncoder,
    ChunkAwareGateHead,
    ChunkAwareGateRuntime,
    gate_seq_to_scalar,
    save_chunk_aware_gate_checkpoint,
)


def test_action_chunk_encoder_outputs_hidden_shape():
    encoder = ActionChunkEncoder(action_dim=14, chunk_len=50, hidden_dim=256, dropout=0.0)

    hidden = encoder(torch.randn(3, 50, 14))

    assert hidden.shape == (3, 256)


def test_chunk_aware_gate_head_outputs_logits_shape():
    model = ChunkAwareGateHead(z_dim=1024, action_dim=14, chunk_len=50, hidden_dim=256, dropout=0.0)

    logits = model(torch.randn(4, 1024), torch.randn(4, 50, 14))

    assert logits.shape == (4, 1)


def test_chunk_aware_gate_head_bce_loss_backpropagates():
    model = ChunkAwareGateHead(z_dim=8, action_dim=3, chunk_len=5, hidden_dim=16, dropout=0.0)
    loss = nn.BCEWithLogitsLoss()(
        model(torch.randn(4, 8), torch.randn(4, 5, 3)),
        torch.ones(4, 1),
    )

    loss.backward()

    assert all(param.grad is not None for param in model.parameters())


def test_wrong_action_chunk_shape_raises_value_error():
    model = ChunkAwareGateHead(z_dim=8, action_dim=3, chunk_len=5, hidden_dim=16, dropout=0.0)

    with pytest.raises(ValueError, match="action_chunk"):
        model(torch.randn(4, 8), torch.randn(4, 4, 3))


def test_chunk_aware_checkpoint_save_load(tmp_path):
    model = ChunkAwareGateHead(z_dim=8, action_dim=3, chunk_len=5, hidden_dim=16, dropout=0.0)
    checkpoint = tmp_path / "chunk_aware_gate_head.pt"
    save_chunk_aware_gate_checkpoint(
        checkpoint,
        model=model,
        config={"z_dim": 8, "action_dim": 3, "chunk_len": 5, "hidden_dim": 16, "dropout": 0.0},
        threshold=0.4,
    )

    runtime = ChunkAwareGateRuntime.load_from_checkpoint(checkpoint, device="cpu")
    prob, gate = runtime.predict_from_inputs(torch.zeros(2, 8), torch.zeros(2, 5, 3))

    assert prob.shape == (2, 1)
    assert gate.shape == (2, 1)
    assert runtime.cfg.threshold == 0.4


def test_gate_seq_to_scalar_uses_max_by_default():
    gate_seq = torch.tensor([[[0.0], [1.0], [0.0]], [[0.0], [0.0], [0.0]]])

    scalar = gate_seq_to_scalar(gate_seq)

    torch.testing.assert_close(scalar, torch.tensor([[1.0], [0.0]]))


def test_gate_seq_to_scalar_respects_min_positive_frames():
    gate_seq = torch.tensor([[[1.0], [0.0], [0.0]], [[1.0], [1.0], [0.0]]])

    scalar = gate_seq_to_scalar(gate_seq, min_positive_frames=2)

    torch.testing.assert_close(scalar, torch.tensor([[0.0], [1.0]]))
