#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
MODE="${MODE:-normal}"
CONDITIONS="${CONDITIONS:-qpos14,zero_k50,bc_k50}"
LIMIT="${LIMIT:-20}"
SEED_START="${SEED_START:-100100000}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
BC_ACTOR_CHECKPOINT="${BC_ACTOR_CHECKPOINT:-logs/residual_actor_bc/hard_pi05_cached_base_k50_s001/residual_actor_bc.pt}"
OUT="${OUT:-logs/gate_controlled_hybrid/k50_eval_matrix_${MODE}_${LIMIT}}"

REPO=/home/user/wyw/RLinf
cd "$REPO"
source scripts/setup_master40_robotwin_env.sh
export ROBOTWIN_PATH=/home/user/wyw/RoboTwin
export PYTHONPATH=/home/user/wyw/RLinf:/home/user/wyw/RoboTwin:${PYTHONPATH:-}

mkdir -p "$OUT"

python3 - "$MODE" "$LIMIT" "$SEED_START" "$OUT/seeds.txt" <<'PY'
import csv
import sys
from pathlib import Path

mode = sys.argv[1]
limit = int(sys.argv[2])
seed_start = int(sys.argv[3])
out_path = Path(sys.argv[4])
if mode == "hard":
    rows = list(csv.DictReader(Path("logs/hard_eval_v1/summary.csv").open()))
    seeds = [int(row["seed"]) for row in rows[:limit]]
elif mode == "normal":
    seeds = [seed_start + idx for idx in range(limit)]
else:
    raise ValueError(f"Unsupported MODE={mode!r}; use normal or hard.")
out_path.write_text("\n".join(str(seed) for seed in seeds) + "\n")
print(" ".join(str(seed) for seed in seeds))
PY

echo "gpu=$GPU_ID mode=$MODE limit=$LIMIT max_steps=$MAX_STEPS out=$OUT" | tee "$OUT/run.log"
echo "conditions=$CONDITIONS" | tee -a "$OUT/run.log"
echo "seeds=$(tr '\n' ' ' < "$OUT/seeds.txt")" | tee -a "$OUT/run.log"

run_rollout() {
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$@" 2>&1 | tee -a "$OUT/run.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -eq 0 ]; then
    return 0
  fi
  if [ "$rc" -eq 139 ] && [ -f "$seed_dir/summary.json" ]; then
    echo "WARN rollout exited with segfault after writing summary: condition=$condition seed=$seed" \
      | tee -a "$OUT/run.log"
    return 0
  fi
  return "$rc"
}

IFS=',' read -r -a condition_array <<< "$CONDITIONS"
for condition in "${condition_array[@]}"; do
  condition="${condition//[[:space:]]/}"
  [ -n "$condition" ] || continue
  while read -r seed; do
    [ -n "$seed" ] || continue
    seed_dir="$OUT/per_condition/$condition/seed_$seed"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$seed_dir/summary.json" ]; then
      echo "SKIP condition=$condition seed=$seed" | tee -a "$OUT/run.log"
      continue
    fi
    rm -rf "$seed_dir"
    mkdir -p "$seed_dir"
    echo "RUN condition=$condition seed=$seed" | tee -a "$OUT/run.log"

    video_args=()
    if [ "$SAVE_VIDEO" = "1" ]; then
      video_args=(--save-video --save-debug --video-fps 10 --video-frame-mode replan)
    fi

    common_args=(
      --config pi05_aloha_robotwin_handover
      --checkpoint /nfs/data3/rlinf_data/pytorch_checkpoint/model.safetensors
      --norm-stats-path /nfs/data3/rlinf_data/pytorch_checkpoint/handover_expert/norm_stats.json
      --max-steps "$MAX_STEPS"
      --num-episodes 1
      --seed "$seed"
      --save-dir "$seed_dir"
      --device cuda
      --planner-backend curobo
      "${video_args[@]}"
    )

    if [ "$condition" = "qpos14" ]; then
      run_rollout python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py \
        "${common_args[@]}" \
        --execution-mode qpos14_baseline
    elif [ "$condition" = "zero_k50" ] || [ "$condition" = "handover_only_k50" ]; then
      run_rollout python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py \
        "${common_args[@]}" \
        --chunk-aware-gate-checkpoint logs/chunk_aware_gate/full/gate_head/chunk_aware_gate_head.pt \
        --execution-mode handover_only_k50 \
        --enable-residual-intervention \
        --ee16-execution-strategy pointwise \
        --residual-actor zero \
        --residual-scale 1.0 \
        --residual-horizon-k 50
    elif [ "$condition" = "pregrasp_only_k50" ]; then
      run_rollout python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py \
        "${common_args[@]}" \
        --execution-mode pregrasp_only_k50 \
        --enable-residual-intervention \
        --enable-pregrasp-intervention \
        --gripper-close-direction "${GRIPPER_CLOSE_DIRECTION:-lower}" \
        --gripper-close-threshold "${GRIPPER_CLOSE_THRESHOLD:-0.25}" \
        --gripper-closing-delta-threshold "${GRIPPER_CLOSING_DELTA_THRESHOLD:-0.5}" \
        --pregrasp-cooldown-steps "${PREGRASP_COOLDOWN_STEPS:-50}" \
        --ee16-execution-strategy pointwise \
        --residual-actor zero \
        --residual-scale 1.0 \
        --residual-horizon-k 50
    elif [ "$condition" = "pregrasp_plus_handover_k50" ]; then
      run_rollout python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py \
        "${common_args[@]}" \
        --chunk-aware-gate-checkpoint logs/chunk_aware_gate/full/gate_head/chunk_aware_gate_head.pt \
        --execution-mode pregrasp_plus_handover_k50 \
        --enable-residual-intervention \
        --enable-pregrasp-intervention \
        --gripper-close-direction "${GRIPPER_CLOSE_DIRECTION:-lower}" \
        --gripper-close-threshold "${GRIPPER_CLOSE_THRESHOLD:-0.25}" \
        --gripper-closing-delta-threshold "${GRIPPER_CLOSING_DELTA_THRESHOLD:-0.5}" \
        --pregrasp-cooldown-steps "${PREGRASP_COOLDOWN_STEPS:-50}" \
        --ee16-execution-strategy pointwise \
        --residual-actor zero \
        --residual-scale 1.0 \
        --residual-horizon-k 50
    elif [ "$condition" = "bc_k50" ]; then
      run_rollout python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py \
        "${common_args[@]}" \
        --chunk-aware-gate-checkpoint logs/chunk_aware_gate/full/gate_head/chunk_aware_gate_head.pt \
        --execution-mode gate_controlled_hybrid \
        --enable-residual-intervention \
        --ee16-execution-strategy pointwise \
        --residual-actor bc \
        --residual-actor-checkpoint "$BC_ACTOR_CHECKPOINT" \
        --enable-learned-residual-control \
        --residual-scale 1.0 \
        --residual-horizon-k 50
    else
      echo "Unsupported condition=$condition" >&2
      exit 2
    fi
  done < "$OUT/seeds.txt"
done

python3 - "$OUT" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for summary_path in sorted(out.glob("per_condition/*/seed_*/summary.json")):
    data = json.loads(summary_path.read_text())
    condition = summary_path.parents[1].name
    for row in data.get("episodes") or []:
        row = dict(row)
        row["condition"] = condition
        row["run_dir"] = str(summary_path.parent)
        rows.append(row)

fieldnames = sorted({key for row in rows for key in row})
with (out / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def as_float(row, key, default=0.0):
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)

by_condition = {}
for row in rows:
    by_condition.setdefault(row["condition"], []).append(row)
metrics = {}
for condition, condition_rows in sorted(by_condition.items()):
    successes = [row for row in condition_rows if str(row.get("success", "")).lower() == "true"]
    metrics[condition] = {
        "num_episodes": len(condition_rows),
        "successes": len(successes),
        "success_rate": len(successes) / len(condition_rows) if condition_rows else 0.0,
        "mean_episode_len": sum(as_float(row, "episode_length") for row in condition_rows)
        / len(condition_rows)
        if condition_rows
        else 0.0,
        "mean_num_interventions": sum(as_float(row, "num_interventions") for row in condition_rows)
        / len(condition_rows)
        if condition_rows
        else 0.0,
        "mean_total_ee_steps": sum(as_float(row, "total_ee_intervention_steps") for row in condition_rows)
        / len(condition_rows)
        if condition_rows
        else 0.0,
        "max_applied_norm": max((as_float(row, "applied_norm_max") for row in condition_rows), default=0.0),
        "nan_inf_total": sum(as_float(row, "nan_inf_count") for row in condition_rows),
        "max_saturation_ratio": max((as_float(row, "saturation_ratio") for row in condition_rows), default=0.0),
    }
(out / "aggregate_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True))
PY
