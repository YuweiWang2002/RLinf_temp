import pytest
import torch

from rlinf.algorithms.residual_td3.residual_actor import (
    ResidualActorConfig,
    ZeroInitResidualActorMLP,
)


def test_zero_init_residual_actor_outputs_zero_chunk():
    actor = ZeroInitResidualActorMLP(
        ResidualActorConfig(obs_dim=21, hidden_dim=16, chunk_len=4, delta_max=0.02)
    )

    out = actor(torch.randn(2, 21))

    assert out.shape == (2, 4, 3)
    assert torch.count_nonzero(out) == 0
    assert float(out.abs().max()) <= 0.02


def test_zero_init_residual_actor_rejects_wrong_obs_shape():
    actor = ZeroInitResidualActorMLP(ResidualActorConfig(obs_dim=21, hidden_dim=16))

    with pytest.raises(ValueError, match="obs must have shape"):
        actor(torch.randn(2, 20))


def test_zero_init_residual_actor_backpropagates_to_hidden_layers():
    actor = ZeroInitResidualActorMLP(
        ResidualActorConfig(obs_dim=21, hidden_dim=16, chunk_len=2, zero_init_output=False)
    )
    obs = torch.randn(3, 21)

    loss = actor(obs).square().mean()
    loss.backward()

    first_layer = actor.net[0]
    assert first_layer.weight.grad is not None
    assert torch.count_nonzero(first_layer.weight.grad) > 0
