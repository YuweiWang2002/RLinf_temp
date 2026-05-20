import pytest
import torch

from rlinf.algorithms.residual_td3.fk_bridge import AlohaFKBridge, AlohaFKBridgeConfig


class FakeArmFK:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def qpos6_to_pose7(self, qpos6: torch.Tensor) -> torch.Tensor:
        batch_size = qpos6.shape[0]
        pose = torch.zeros(batch_size, 7, dtype=qpos6.dtype, device=qpos6.device)
        pose[:, 0] = qpos6[:, 0] + self.offset
        pose[:, 1] = qpos6[:, 1] + self.offset
        pose[:, 2] = qpos6[:, 2] + self.offset
        pose[:, 3] = 1.0
        return pose


def _fake_bridge() -> AlohaFKBridge:
    return AlohaFKBridge(
        left_backend=FakeArmFK(offset=10.0),
        right_backend=FakeArmFK(offset=20.0),
    )


def test_qpos14_chunk_to_endpose16_shape():
    batch_size, chunk_len = 2, 3
    qpos14 = torch.arange(batch_size * chunk_len * 14, dtype=torch.float32).reshape(
        batch_size,
        chunk_len,
        14,
    )

    out = _fake_bridge().qpos14_to_endpose16(qpos14)

    assert out.shape == (batch_size, chunk_len, 16)
    torch.testing.assert_close(out[..., 0], qpos14[..., 0] + 10.0)
    torch.testing.assert_close(out[..., 8], qpos14[..., 7] + 20.0)


def test_qpos14_step_to_endpose16_copies_grippers():
    qpos14 = torch.randn(4, 14)

    out = _fake_bridge().qpos14_step_to_endpose16(qpos14)

    assert out.shape == (4, 16)
    torch.testing.assert_close(out[..., 7], qpos14[..., 6])
    torch.testing.assert_close(out[..., 15], qpos14[..., 13])


def test_qpos14_chunk_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"\[B, C, 14\]"):
        _fake_bridge().qpos14_to_endpose16(torch.zeros(2, 14))


def test_unavailable_fk_backend_raises_clear_error():
    cfg = AlohaFKBridgeConfig(
        robotwin_path=None,
        left_curobo_yml_path="missing_left.yml",
        right_curobo_yml_path="missing_right.yml",
    )

    with pytest.raises(RuntimeError, match="requires a working CuRobo FK backend"):
        AlohaFKBridge(cfg)
