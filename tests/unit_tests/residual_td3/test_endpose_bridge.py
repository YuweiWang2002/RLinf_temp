import pytest
import torch

from rlinf.algorithms.residual_td3.action_adapter import (
    ResidualActionAdapter,
    ResidualActionSpec,
)
from rlinf.algorithms.residual_td3.endpose_bridge import (
    EndposeBridge,
    EndposeBridgeSpec,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def _direct_bridge(residual_chunk_len: int = 5) -> EndposeBridge:
    return EndposeBridge(
        EndposeBridgeSpec(
            mode="direct_endpose16",
            input_action_space="robotwin_endpose16",
            base_action_dim=16,
            residual_chunk_len=residual_chunk_len,
            env_action_chunk_len=10,
        )
    )


def _current_state_bridge(residual_chunk_len: int = 5) -> EndposeBridge:
    return EndposeBridge(
        EndposeBridgeSpec(
            mode="current_state_ee_pose",
            input_action_space="aloha_qpos14",
            base_action_dim=14,
            residual_chunk_len=residual_chunk_len,
            env_action_chunk_len=10,
            require_current_ee_pose=True,
        )
    )


def _obs(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "left_ee_pos": torch.arange(batch_size * 3, dtype=torch.float32).reshape(
            batch_size, 3
        ),
        "left_ee_quat_xyzw": torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]] * batch_size,
            dtype=torch.float32,
        ),
        "left_gripper": torch.arange(batch_size, dtype=torch.float32),
        "right_ee_pos": torch.arange(
            batch_size * 3,
            batch_size * 6,
            dtype=torch.float32,
        ).reshape(batch_size, 3),
        "right_ee_quat_xyzw": torch.tensor(
            [[0.0, 0.0, 1.0, 0.0]] * batch_size,
            dtype=torch.float32,
        ),
        "right_gripper": torch.arange(batch_size, dtype=torch.float32).reshape(
            batch_size, 1
        ),
    }


def test_direct_endpose16_returns_prefix_contiguous_chunk():
    batch_size, action_chunk_len, residual_chunk_len = 2, 8, 5
    base = torch.arange(
        batch_size * action_chunk_len * 16,
        dtype=torch.float32,
    ).reshape(batch_size, action_chunk_len, 16)

    out = _direct_bridge(residual_chunk_len).build_base_endpose16_chunk(
        base_action_chunk=base
    )

    assert out.shape == (batch_size, residual_chunk_len, 16)
    assert out.is_contiguous()
    torch.testing.assert_close(out, base[:, :residual_chunk_len, :])


def test_direct_endpose16_wrong_dim_raises():
    bridge = _direct_bridge()

    with pytest.raises(ValueError, match="expected base_action_dim=16"):
        bridge.build_base_endpose16_chunk(base_action_chunk=torch.zeros(2, 8, 14))


def test_direct_endpose16_residual_chunk_len_gt_available_raises():
    bridge = _direct_bridge(residual_chunk_len=9)

    with pytest.raises(ValueError, match="available base action chunk length"):
        bridge.build_base_endpose16_chunk(base_action_chunk=torch.zeros(2, 8, 16))


def test_current_state_ee_pose_repeats_current_pose_as_endpose16():
    batch_size, action_chunk_len, residual_chunk_len = 2, 8, 5
    obs = _obs(batch_size)
    base = torch.zeros(batch_size, action_chunk_len, 14)

    out = _current_state_bridge(residual_chunk_len).build_base_endpose16_chunk(
        base_action_chunk=base,
        obs=obs,
    )

    assert out.shape == (batch_size, residual_chunk_len, 16)
    torch.testing.assert_close(out[..., 0:3], obs["left_ee_pos"][:, None, :].repeat(1, 5, 1))
    torch.testing.assert_close(
        out[..., 3:7],
        obs["left_ee_quat_xyzw"][:, None, :].repeat(1, 5, 1),
    )
    torch.testing.assert_close(out[..., 7], obs["left_gripper"][:, None].repeat(1, 5))
    torch.testing.assert_close(out[..., 8:11], obs["right_ee_pos"][:, None, :].repeat(1, 5, 1))
    torch.testing.assert_close(
        out[..., 11:15],
        obs["right_ee_quat_xyzw"][:, None, :].repeat(1, 5, 1),
    )
    torch.testing.assert_close(
        out[..., 15],
        obs["right_gripper"].repeat(1, 5),
    )


@pytest.mark.parametrize("missing_key", ["left_ee_quat_xyzw", "right_ee_pos"])
def test_current_state_ee_pose_missing_field_raises(missing_key: str):
    obs = _obs(batch_size=2)
    obs.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        _current_state_bridge().build_base_endpose16_chunk(
            base_action_chunk=torch.zeros(2, 8, 14),
            obs=obs,
        )


def test_learned_qpos_to_endpose16_is_explicitly_unimplemented():
    bridge = EndposeBridge(
        EndposeBridgeSpec(
            mode="learned_qpos_to_endpose16",
            input_action_space="aloha_qpos14",
            base_action_dim=14,
            residual_chunk_len=5,
        )
    )

    with pytest.raises(NotImplementedError, match="future learned action bridge"):
        bridge.build_base_endpose16_chunk(base_action_chunk=torch.zeros(2, 8, 14))


def test_fk_qpos_to_endpose16_is_explicitly_unimplemented():
    bridge = EndposeBridge(
        EndposeBridgeSpec(
            mode="fk_qpos_to_endpose16",
            input_action_space="aloha_qpos14",
            base_action_dim=14,
            residual_chunk_len=5,
        )
    )

    with pytest.raises(NotImplementedError, match="future FK bridge"):
        bridge.build_base_endpose16_chunk(base_action_chunk=torch.zeros(2, 8, 14))


def test_endpose_bridge_integrates_with_residual_ref_extraction():
    batch_size, action_chunk_len, residual_chunk_len = 2, 8, 5
    base = torch.arange(
        batch_size * action_chunk_len * 16,
        dtype=torch.float32,
    ).reshape(batch_size, action_chunk_len, 16)
    base_endpose16 = _direct_bridge(residual_chunk_len).build_base_endpose16_chunk(
        base_action_chunk=base
    )
    adapter = ResidualActionAdapter(
        ResidualActionSpec(
            base_action_space="robotwin_endpose16",
            residual_mode="right_xyz_world_frame",
            residual_frame="world",
            base_action_dim=16,
            residual_chunk_len=residual_chunk_len,
            env_action_chunk_len=action_chunk_len,
            right_xyz_indices=[8, 9, 10],
        )
    )

    residual_ref = adapter.extract_residual_ref(base_endpose16)

    assert residual_ref.shape == (batch_size, residual_chunk_len, 3)
    torch.testing.assert_close(residual_ref, base_endpose16[..., 8:11])


def test_quat_order_helpers_convert_xyzw_and_wxyz():
    q_xyzw = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    q_wxyz = xyzw_to_wxyz(q_xyzw)
    round_trip = wxyz_to_xyzw(q_wxyz)

    torch.testing.assert_close(q_wxyz, torch.tensor([[4.0, 1.0, 2.0, 3.0]]))
    torch.testing.assert_close(round_trip, q_xyzw)
