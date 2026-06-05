"""Collect successful RoboTwin scripted expert demos for hard seeds."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-path", default=os.environ.get("ROBOTWIN_PATH", "/home/user/wyw/RoboTwin"))
    parser.add_argument("--task-name", default="handover_block")
    parser.add_argument("--task-config", default="codex_openpi05_randomized_200")
    parser.add_argument("--hard-seeds-json", required=True)
    parser.add_argument("--hard-seed-groups", default="all_fail,candidate_hard_eval_seeds,rows")
    parser.add_argument("--num-hard-seeds", type=int, default=None)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--render-freq", type=int, default=0)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = select_hard_seed_rows(
        args.hard_seeds_json,
        [group.strip() for group in args.hard_seed_groups.split(",") if group.strip()],
        args.num_hard_seeds,
    )
    if args.episodes_per_seed < 1:
        raise ValueError("--episodes-per-seed must be positive.")

    save_dir = Path(args.save_dir).resolve()
    args.save_dir = str(save_dir)
    demo_dir = save_dir / "successful_demos"
    save_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    config = {
        **vars(args),
        "selected_seeds": selected,
        "successful_demo_dir": str(demo_dir),
        "expert_source": "RoboTwin scripted task play_once + motion planner",
        "privileged_pose_note": (
            "RoboTwin scripted expert uses simulator object poses and functional points during "
            "offline demo generation only. These privileged poses are not online residual actor inputs."
        ),
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = collect_successful_demos(args, selected, demo_dir)
    write_summary_csv(save_dir / "expert_success_summary.csv", rows)
    write_success_json(save_dir / "expert_success_seeds.json", rows, config)
    return 0


def collect_successful_demos(args: argparse.Namespace, selected: list[dict[str, Any]], demo_dir: Path) -> list[dict[str, Any]]:
    """Plan and collect successful demos for selected hard seeds."""

    robotwin_path = Path(args.robotwin_path).resolve()
    with pushd(robotwin_path):
        collect_helpers = load_robotwin_collect_helpers(robotwin_path)
        task = collect_helpers["class_decorator"](args.task_name)
        rt_args = build_robotwin_args(collect_helpers, args, demo_dir)
        rows: list[dict[str, Any]] = []
        success_episode_id = 0
        for selected_row in selected:
            seed = int(selected_row["seed"])
            for repeat_id in range(args.episodes_per_seed):
                row = {
                    "rank": selected_row.get("rank", ""),
                    "seed": seed,
                    "source_group": selected_row.get("source_group", ""),
                    "repeat_id": repeat_id,
                    "expert_success": 0,
                    "plan_success": 0,
                    "check_success": 0,
                    "episode_id": "",
                    "episode_length": "",
                    "failure_reason": "",
                    "traj_path": "",
                    "hdf5_path": "",
                    "video_path": "",
                }
                try:
                    if should_skip_existing(demo_dir, success_episode_id, args.skip_existing):
                        fill_existing_row(row, demo_dir, success_episode_id)
                        rows.append(row)
                        success_episode_id += 1
                        continue
                    row, success_episode_id = try_collect_one(
                        task,
                        rt_args,
                        row,
                        seed=seed,
                        episode_id=success_episode_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    row["failure_reason"] = f"{type(exc).__name__}: {exc}"
                    row["traceback"] = traceback.format_exc(limit=8)
                    safe_close(task)
                rows.append(row)
                flush_outputs(args.save_dir, rows, config_seed_count=len(selected))
        return rows


def try_collect_one(
    task: Any,
    rt_args: dict[str, Any],
    row: dict[str, Any],
    *,
    seed: int,
    episode_id: int,
) -> tuple[dict[str, Any], int]:
    """Plan one expert demo and save HDF5 only when the expert succeeds."""

    plan_args = dict(rt_args)
    plan_args.update(
        {
            "need_plan": True,
            "save_data": False,
            "now_ep_num": episode_id,
            "seed": seed,
        }
    )
    task.setup_demo(**plan_args)
    task.play_once()
    plan_success = bool(task.plan_success)
    check_success = bool(task.check_success()) if plan_success else False
    row["plan_success"] = int(plan_success)
    row["check_success"] = int(check_success)
    if not (plan_success and check_success):
        row["failure_reason"] = "expert_plan_or_task_failed"
        safe_close(task)
        return row, episode_id

    task.save_traj_data(episode_id)
    safe_close(task)

    collect_args = dict(rt_args)
    collect_args.update(
        {
            "need_plan": False,
            "save_data": True,
            "render_freq": 0,
            "now_ep_num": episode_id,
            "seed": seed,
        }
    )
    task.setup_demo(**collect_args)
    traj_data = task.load_tran_data(episode_id)
    collect_args["left_joint_path"] = traj_data["left_joint_path"]
    collect_args["right_joint_path"] = traj_data["right_joint_path"]
    task.set_path_lst(collect_args)
    info = task.play_once()
    collect_success = bool(task.check_success())
    write_scene_info(Path(rt_args["save_path"]), episode_id, seed, info)
    safe_close(task)
    if not collect_success:
        row["failure_reason"] = "expert_replay_failed"
        return row, episode_id

    task.merge_pkl_to_hdf5_video()
    task.remove_data_cache()
    hdf5_path = Path(rt_args["save_path"]) / "data" / f"episode{episode_id}.hdf5"
    video_path = Path(rt_args["save_path"]) / "video" / f"episode{episode_id}.mp4"
    row.update(
        {
            "expert_success": 1,
            "episode_id": episode_id,
            "episode_length": read_hdf5_length(hdf5_path),
            "traj_path": str(Path(rt_args["save_path"]) / "_traj_data" / f"episode{episode_id}.pkl"),
            "hdf5_path": str(hdf5_path),
            "video_path": str(video_path) if video_path.exists() else "",
        }
    )
    return row, episode_id + 1


def load_robotwin_collect_helpers(robotwin_path: Path) -> dict[str, Any]:
    """Load RoboTwin collect_data helpers without importing them at module import time."""

    sys.path.insert(0, str(robotwin_path))
    collect_data = importlib.import_module("script.collect_data")
    global_configs = importlib.import_module("envs._GLOBAL_CONFIGS")

    return {
        "class_decorator": collect_data.class_decorator,
        "get_embodiment_config": collect_data.get_embodiment_config,
        "configs_path": global_configs.CONFIGS_PATH,
    }


def build_robotwin_args(helpers: dict[str, Any], args: argparse.Namespace, demo_dir: Path) -> dict[str, Any]:
    """Build the RoboTwin task args used by scripted demo collection."""

    config_path = Path(args.robotwin_path) / "task_config" / f"{args.task_config}.yml"
    with config_path.open("r", encoding="utf-8") as f:
        rt_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    rt_args["task_name"] = args.task_name
    rt_args["task_config"] = args.task_config
    rt_args["save_path"] = str(demo_dir)
    rt_args["render_freq"] = args.render_freq
    rt_args["eval_video_log"] = bool(args.save_video)
    rt_args["step_lim"] = args.max_steps

    embodiment_type = rt_args.get("embodiment")
    embodiment_config_path = Path(helpers["configs_path"]) / "_embodiment_config.yml"
    with embodiment_config_path.open("r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def embodiment_file(embodiment: str) -> str:
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise FileNotFoundError(f"missing embodiment file for {embodiment}")
        return robot_file

    if len(embodiment_type) == 1:
        rt_args["left_robot_file"] = embodiment_file(embodiment_type[0])
        rt_args["right_robot_file"] = embodiment_file(embodiment_type[0])
        rt_args["dual_arm_embodied"] = True
        rt_args["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        rt_args["left_robot_file"] = embodiment_file(embodiment_type[0])
        rt_args["right_robot_file"] = embodiment_file(embodiment_type[1])
        rt_args["embodiment_dis"] = embodiment_type[2]
        rt_args["dual_arm_embodied"] = False
        rt_args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise ValueError("embodiment must contain either 1 or 3 entries.")

    rt_args["left_embodiment_config"] = helpers["get_embodiment_config"](rt_args["left_robot_file"])
    rt_args["right_embodiment_config"] = helpers["get_embodiment_config"](rt_args["right_robot_file"])
    return rt_args


def select_hard_seed_rows(path: str | Path, groups: list[str], limit: int | None) -> list[dict[str, Any]]:
    """Load seed rows from hard_eval_v1 or hard_seed_summary JSON."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows: list[dict[str, Any]] = []
    for group in groups:
        rows.extend(extract_group_rows(data, group))
    rows = unique_seed_rows(rows)
    if limit is not None:
        if limit < 1:
            raise ValueError("--num-hard-seeds must be positive.")
        rows = rows[:limit]
    return rows


def extract_group_rows(data: object, group: str) -> list[dict[str, Any]]:
    """Extract one priority group from supported hard seed JSON schemas."""

    if not isinstance(data, dict):
        return []
    value = data.get(group)
    if isinstance(value, list):
        return rows_from_values(value, group)

    rows = data.get("rows")
    if isinstance(rows, list):
        if group == "rows":
            return rows_from_values(rows, group)
        return rows_from_values(
            [row for row in rows if isinstance(row, dict) and row.get("source_group") == group],
            group,
        )
    return []


def rows_from_values(values: list[object], group: str) -> list[dict[str, Any]]:
    """Convert JSON seed values to normalized rows."""

    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(values, start=1):
        if isinstance(value, dict):
            if "seed" not in value:
                continue
            rows.append(
                {
                    "rank": value.get("rank", idx),
                    "seed": int(value["seed"]),
                    "source_group": value.get("source_group", group),
                }
            )
        else:
            rows.append({"rank": idx, "seed": int(value), "source_group": group})
    return rows


def unique_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate rows by seed while preserving priority order."""

    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["seed"])
        if seed in seen:
            continue
        seen.add(seed)
        unique.append(row)
    return unique


def should_skip_existing(demo_dir: Path, episode_id: int, skip_existing: bool) -> bool:
    return skip_existing and (demo_dir / "data" / f"episode{episode_id}.hdf5").exists()


def fill_existing_row(row: dict[str, Any], demo_dir: Path, episode_id: int) -> None:
    hdf5_path = demo_dir / "data" / f"episode{episode_id}.hdf5"
    row.update(
        {
            "expert_success": 1,
            "plan_success": 1,
            "check_success": 1,
            "episode_id": episode_id,
            "episode_length": read_hdf5_length(hdf5_path),
            "hdf5_path": str(hdf5_path),
            "traj_path": str(demo_dir / "_traj_data" / f"episode{episode_id}.pkl"),
            "video_path": str(demo_dir / "video" / f"episode{episode_id}.mp4"),
        }
    )


def read_hdf5_length(path: Path) -> int:
    try:
        import h5py

        with h5py.File(path, "r") as f:
            return int(f["/joint_action/vector"].shape[0])
    except Exception:  # noqa: BLE001
        return -1


def write_scene_info(save_path: Path, episode_id: int, seed: int, info: dict[str, Any]) -> None:
    path = save_path / "scene_info.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data[f"episode_{episode_id}"] = {"seed": seed, "info": info}
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "seed",
        "source_group",
        "repeat_id",
        "expert_success",
        "plan_success",
        "check_success",
        "episode_id",
        "episode_length",
        "failure_reason",
        "traj_path",
        "hdf5_path",
        "video_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_success_json(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    successful = [row for row in rows if int(row.get("expert_success") or 0) == 1]
    failed = [row for row in rows if int(row.get("expert_success") or 0) == 0]
    payload = {
        "num_requested_seeds": len(config["selected_seeds"]),
        "num_attempts": len(rows),
        "num_successful_demos": len(successful),
        "success_seeds": [int(row["seed"]) for row in successful],
        "failed_seeds": [int(row["seed"]) for row in failed],
        "successful_hdf5": [row["hdf5_path"] for row in successful],
        "failed_rows": failed,
        "config_path": str(Path(config["save_dir"]) / "config.json"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def flush_outputs(save_dir: str, rows: list[dict[str, Any]], *, config_seed_count: int) -> None:
    out = Path(save_dir)
    write_summary_csv(out / "expert_success_summary.csv", rows)
    payload = {
        "num_requested_seeds": config_seed_count,
        "num_attempts": len(rows),
        "num_successful_demos": sum(int(row.get("expert_success") or 0) for row in rows),
        "success_seeds": [int(row["seed"]) for row in rows if int(row.get("expert_success") or 0) == 1],
        "failed_seeds": [int(row["seed"]) for row in rows if int(row.get("expert_success") or 0) == 0],
    }
    (out / "expert_success_seeds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_close(task: Any) -> None:
    try:
        task.close_env()
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
