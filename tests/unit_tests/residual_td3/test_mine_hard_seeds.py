from pathlib import Path

from scripts.mine_hard_seeds import (
    EpisodeResult,
    build_hard_seed_groups,
    cleanup_candidates,
    condition_action_source,
    condition_horizon_k,
    parse_conditions,
    shard_seeds,
    success_video_cleanup_candidates,
)


def make_result(seed: int, condition: str, success: bool) -> EpisodeResult:
    root = Path("logs/hard_seed_mining/run/per_condition") / condition / f"seed_{seed}"
    return EpisodeResult(
        seed=seed,
        condition=condition,
        success=success,
        failure_reason="" if success else "timeout",
        episode_len=300 if success else 600,
        first_gate_step=150 if condition != "qpos14" else None,
        num_interventions=0,
        total_ee_steps=0,
        video_path=str(root / "videos" / "episode_0000_seed_1.mp4"),
        debug_path=str(root / "episode_0000_debug.json"),
        log_path=str(root / "hybrid_log_episode_0000.parquet"),
        rollout_status="completed",
        return_code=0,
    )


def test_shard_seeds_uses_seed_index() -> None:
    seeds = [100, 105, 110, 115, 120]

    assert shard_seeds(seeds, num_shards=2, shard_id=0) == [100, 110, 120]
    assert shard_seeds(seeds, num_shards=2, shard_id=1) == [105, 115]


def test_build_hard_seed_groups() -> None:
    results = [
        make_result(1, "qpos14", False),
        make_result(1, "zero_k10", True),
        make_result(2, "qpos14", False),
        make_result(2, "zero_k10", False),
        make_result(3, "qpos14", True),
        make_result(3, "zero_k10", False),
        make_result(4, "qpos14", True),
        make_result(4, "zero_k10", True),
    ]

    groups = build_hard_seed_groups(results)

    assert groups["qpos_fail"] == [1, 2]
    assert groups["zero_k10_fail"] == [2, 3]
    assert groups["qpos_fail_zero_success"] == [1]
    assert groups["qpos_success_zero_fail"] == [3]
    assert groups["all_fail"] == [2]
    assert groups["infra_fail"] == []
    assert groups["candidate_hard_eval_seeds"] == [1, 2, 3]


def test_build_hard_seed_groups_prefers_k50_conditions() -> None:
    results = [
        make_result(1, "qpos14", False),
        make_result(1, "zero_k50", True),
        make_result(2, "qpos14", True),
        make_result(2, "zero_k50", False),
        make_result(3, "bc_k50", False),
    ]

    groups = build_hard_seed_groups(results)

    assert groups["zero_k50_fail"] == [2]
    assert groups["bc_k50_fail"] == [3]
    assert groups["qpos_fail_zero_success"] == [1]
    assert groups["qpos_success_zero_fail"] == [2]
    assert groups["candidate_hard_eval_seeds"] == [1, 2, 3]


def test_k50_conditions_are_supported() -> None:
    assert parse_conditions("qpos14,zero_k50,bc_k50") == ["qpos14", "zero_k50", "bc_k50"]
    assert condition_horizon_k("zero_k50") == 50
    assert condition_horizon_k("bc_k50") == 50
    assert condition_horizon_k("zero_k10") == 10
    assert condition_horizon_k("bc_s05") == 10
    assert condition_action_source("zero_k50") == "zero"
    assert condition_action_source("bc_k50") == "bc"


def test_build_hard_seed_groups_excludes_infra_failures_from_hard_sets() -> None:
    infra = make_result(5, "qpos14", False)
    infra = infra.__class__(
        **{**infra.__dict__, "rollout_status": "missing_summary", "return_code": 1}
    )

    groups = build_hard_seed_groups([infra])

    assert groups["qpos_fail"] == []
    assert groups["all_fail"] == []
    assert groups["infra_fail"] == [5]
    assert groups["candidate_hard_eval_seeds"] == []


def test_build_hard_seed_groups_accepts_completed_summary_with_sigsegv() -> None:
    completed = make_result(6, "qpos14", False)
    completed = completed.__class__(
        **{**completed.__dict__, "rollout_status": "completed", "return_code": -11}
    )

    groups = build_hard_seed_groups([completed])

    assert groups["qpos_fail"] == [6]
    assert groups["infra_fail"] == []


def test_cleanup_candidates_keeps_hard_successes(tmp_path) -> None:
    easy = make_result(10, "qpos14", True)
    hard_success = make_result(11, "zero_k10", True)
    failure = make_result(11, "qpos14", False)
    hard_groups = build_hard_seed_groups([easy, hard_success, failure])
    easy_dir = tmp_path / "per_condition" / "qpos14" / "seed_10"
    hard_dir = tmp_path / "per_condition" / "zero_k10" / "seed_11"
    easy_log = easy_dir / "hybrid_log_episode_0000.parquet"
    hard_log = hard_dir / "hybrid_log_episode_0000.parquet"
    easy_log.parent.mkdir(parents=True)
    hard_log.parent.mkdir(parents=True)
    easy_log.write_text("large", encoding="utf-8")
    hard_log.write_text("large", encoding="utf-8")
    easy = easy.__class__(
        **{**easy.__dict__, "log_path": str(easy_log), "video_path": "", "debug_path": ""}
    )
    hard_success = hard_success.__class__(
        **{
            **hard_success.__dict__,
            "log_path": str(hard_log),
            "video_path": "",
            "debug_path": "",
        }
    )

    candidates = cleanup_candidates(
        [easy, hard_success, failure],
        hard_groups,
    )

    assert candidates == [easy_log]


def test_success_video_cleanup_candidates_removes_only_success_videos(
    tmp_path,
) -> None:
    success = make_result(20, "zero_k10", True)
    failure = make_result(21, "zero_k10", False)
    success_video = tmp_path / "success.mp4"
    failure_video = tmp_path / "failure.mp4"
    success_video.write_text("video", encoding="utf-8")
    failure_video.write_text("video", encoding="utf-8")
    success = success.__class__(
        **{**success.__dict__, "video_path": str(success_video)}
    )
    failure = failure.__class__(
        **{**failure.__dict__, "video_path": str(failure_video)}
    )

    assert success_video_cleanup_candidates([success, failure]) == [success_video]
