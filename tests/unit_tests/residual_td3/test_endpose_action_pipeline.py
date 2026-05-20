import pytest
import torch

from rlinf.algorithms.residual_td3.action_adapter import (
    ResidualActionAdapter,
    ResidualActionSpec,
)
from rlinf.algorithms.residual_td3.endpose_action_pipeline import (
    EndposeActionPipeline,
    EndposeActionPipelineConfig,
)


class FakeFKBridge:
    def qpos14_to_endpose16(self, qpos14_chunk: torch.Tensor, current_obs=None) -> torch.Tensor:
        del current_obs
        out = torch.zeros(
            *qpos14_chunk.shape[:2],
            16,
            dtype=qpos14_chunk.dtype,
            device=qpos14_chunk.device,
        )
        out[..., 0:3] = qpos14_chunk[..., 0:3] + 10.0
        out[..., 3] = 1.0
        out[..., 7] = qpos14_chunk[..., 6]
        out[..., 8:11] = qpos14_chunk[..., 7:10] + 20.0
        out[..., 11] = 1.0
        out[..., 15] = qpos14_chunk[..., 13]
        return out


def _adapter(chunk_len: int = 3) -> ResidualActionAdapter:
    return ResidualActionAdapter(
        ResidualActionSpec(
            base_action_space="robotwin_endpose16",
            residual_mode="right_xyz_world_frame",
            residual_frame="world",
            base_action_dim=16,
            residual_chunk_len=chunk_len,
            env_action_chunk_len=chunk_len,
            right_xyz_indices=[8, 9, 10],
        )
    )


def _pipeline(chunk_len: int = 3) -> EndposeActionPipeline:
    return EndposeActionPipeline(
        fk_bridge=FakeFKBridge(),
        residual_adapter=_adapter(chunk_len),
        config=EndposeActionPipelineConfig(
            residual_chunk_len=chunk_len,
            env_action_chunk_len=chunk_len,
        ),
    )


def test_qpos14_chunk_to_base_endpose16_shape():
    qpos14 = torch.arange(2 * 3 * 14, dtype=torch.float32).reshape(2, 3, 14)

    base = _pipeline().qpos14_chunk_to_base_endpose16(qpos14)

    assert base.shape == (2, 3, 16)
    assert base.is_contiguous()
    torch.testing.assert_close(base[..., 7], qpos14[..., 6])
    torch.testing.assert_close(base[..., 15], qpos14[..., 13])


def test_extract_right_xyz_ref_shape():
    qpos14 = torch.zeros(2, 3, 14)
    pipeline = _pipeline()
    base = pipeline.qpos14_chunk_to_base_endpose16(qpos14)

    ref = pipeline.extract_right_xyz_ref(base)

    assert ref.shape == (2, 3, 3)
    assert ref.is_contiguous()
    torch.testing.assert_close(ref, base[..., 8:11])


def test_zero_residual_executed_endpose16_equals_base():
    qpos14 = torch.randn(2, 3, 14)

    out = _pipeline().build_zero_residual_action(qpos14)

    torch.testing.assert_close(out.executed_endpose16_chunk, out.base_endpose16_chunk)
    torch.testing.assert_close(out.zero_residual_chunk, torch.zeros_like(out.residual_ref_chunk))


def test_small_residual_only_modifies_right_xyz():
    qpos14 = torch.randn(2, 3, 14)
    pipeline = _pipeline()
    base = pipeline.qpos14_chunk_to_base_endpose16(qpos14)
    residual = torch.full((2, 3, 3), 0.25)
    gate = torch.ones(2, 1)

    executed = pipeline.compose_executed_endpose16(base, residual, gate)

    torch.testing.assert_close(executed[..., 8:11], base[..., 8:11] + residual)
    unchanged_indices = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15]
    torch.testing.assert_close(executed[..., unchanged_indices], base[..., unchanged_indices])


def test_left_arm_quat_and_grippers_remain_unchanged():
    qpos14 = torch.randn(2, 3, 14)
    pipeline = _pipeline()
    base = pipeline.qpos14_chunk_to_base_endpose16(qpos14)
    residual = torch.randn(2, 3, 3)
    gate = torch.ones(2, 3, 1)

    executed = pipeline.compose_executed_endpose16(base, residual, gate)

    torch.testing.assert_close(executed[..., 0:8], base[..., 0:8])
    torch.testing.assert_close(executed[..., 11:16], base[..., 11:16])


def test_gate_zero_disables_residual():
    qpos14 = torch.randn(2, 3, 14)
    pipeline = _pipeline()
    base = pipeline.qpos14_chunk_to_base_endpose16(qpos14)
    residual = torch.ones(2, 3, 3)
    gate = torch.zeros(2, 1)

    executed = pipeline.compose_executed_endpose16(base, residual, gate)

    torch.testing.assert_close(executed, base)


def test_wrong_qpos_dim_raises_value_error():
    with pytest.raises(ValueError, match=r"\[B, C, 14\]"):
        _pipeline().qpos14_chunk_to_base_endpose16(torch.zeros(2, 3, 13))


def test_wrong_residual_dim_raises_value_error():
    pipeline = _pipeline()
    base = pipeline.qpos14_chunk_to_base_endpose16(torch.zeros(2, 3, 14))

    with pytest.raises(ValueError, match="last dim 3"):
        pipeline.compose_executed_endpose16(base, torch.zeros(2, 3, 2), torch.ones(2, 1))
