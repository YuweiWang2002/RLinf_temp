import argparse
from pathlib import Path

import numpy as np
import pytest

from scripts.extract_pi05_action_cache import (
    build_manifest,
    episode_cache_path,
    select_episode_paths,
    should_skip_existing,
    validate_cache_schema,
    write_episode_cache_atomic,
)


def _payload(t_valid=3, chunk_size=5):
    return {
        "base_action_chunk": np.zeros((t_valid, chunk_size, 14), dtype=np.float32),
        "frame_index": np.arange(t_valid, dtype=np.int64),
        "episode_index": np.zeros((t_valid,), dtype=np.int64),
        "expert_action_chunk": np.zeros((t_valid, chunk_size, 14), dtype=np.float32),
        "residual_gate_seq": np.zeros((t_valid, chunk_size), dtype=np.uint8),
        "gate_label_chunk": np.zeros((t_valid,), dtype=np.uint8),
    }


def test_select_episode_paths_uses_episode_id_mod_shard():
    paths = [Path(f"episode_{idx:06d}.parquet") for idx in range(8)]

    selected = select_episode_paths(
        paths,
        num_shards=4,
        shard_id=2,
        episode_start=None,
        episode_end=None,
        max_episodes=None,
    )

    assert [path.stem for path in selected] == ["episode_000002", "episode_000006"]


def test_select_episode_paths_applies_range_before_max_episodes():
    paths = [Path(f"episode_{idx:06d}.parquet") for idx in range(10)]

    selected = select_episode_paths(
        paths,
        num_shards=1,
        shard_id=0,
        episode_start=3,
        episode_end=8,
        max_episodes=2,
    )

    assert [path.stem for path in selected] == ["episode_000003", "episode_000004"]


def test_should_skip_existing_respects_overwrite(tmp_path):
    path = tmp_path / "episode_000001.npz"
    path.write_bytes(b"x")

    assert should_skip_existing(path, skip_existing=True, overwrite=False)
    assert not should_skip_existing(path, skip_existing=True, overwrite=True)
    assert not should_skip_existing(path, skip_existing=False, overwrite=False)


def test_write_episode_cache_atomic_writes_final_path_and_removes_tmp(tmp_path):
    path = episode_cache_path(tmp_path, 1)

    write_episode_cache_atomic(path, _payload())

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    with np.load(path) as cache:
        assert cache["base_action_chunk"].shape == (3, 5, 14)


def test_validate_cache_schema_rejects_wrong_base_shape():
    payload = _payload()
    payload["base_action_chunk"] = np.zeros((3, 4, 14), dtype=np.float32)

    with pytest.raises(ValueError, match="base_action_chunk"):
        validate_cache_schema(payload, chunk_size=5)


def test_validate_cache_schema_rejects_non_contiguous_frame_index():
    payload = _payload()
    payload["frame_index"] = np.array([0, 2, 3], dtype=np.int64)

    with pytest.raises(ValueError, match="frame_index"):
        validate_cache_schema(payload, chunk_size=5)


def test_build_manifest_contains_core_fields():
    args = argparse.Namespace(
        dataset_dir="/data",
        output_dir="/out",
        chunk_size=50,
        pi05_config="pi05",
        pi05_checkpoint="/ckpt/model.safetensors",
        norm_stats_path="/ckpt/norm_stats.json",
        shard_id=1,
        num_shards=4,
        device="cuda:1",
        episode_start=None,
        episode_end=None,
        max_episodes=2,
        save_expert_action_chunk=True,
        save_residual_gate_seq=True,
        save_action_head_hidden=False,
    )

    manifest = build_manifest(
        args,
        [Path("episode_000001.parquet")],
        completed=[1],
        skipped=[],
        failed=[],
        per_episode=[{"episode_index": 1, "num_frames_valid": 3}],
    )

    assert manifest["dataset_path"] == "/data"
    assert manifest["chunk_size"] == 50
    assert manifest["action_dim"] == 14
    assert manifest["completed_episodes"] == [1]
    assert manifest["shard"]["shard_id"] == 1
    assert manifest["model"]["pi05_config"] == "pi05"
