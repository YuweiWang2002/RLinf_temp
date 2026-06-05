import sys

from scripts.rollout_pi05_gate_controlled_hybrid_zero_residual import parse_args


def test_rollout_cli_accepts_td3_residual_actor(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollout",
            "--checkpoint",
            "model.safetensors",
            "--save-dir",
            "out",
            "--residual-actor",
            "td3",
            "--residual-actor-checkpoint",
            "actor_td3.pt",
        ],
    )

    args = parse_args()

    assert args.residual_actor == "td3"
    assert args.residual_actor_checkpoint == "actor_td3.pt"
