#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONDITIONS="${CONDITIONS:-zero_k50,bc_k50}"
SEEDS="${SEEDS:-100100048 100100115 100100172 100100194 100100073 100100081 100100094 100100095}"
MAX_STEPS="${MAX_STEPS:-600}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
BC_ACTOR_CHECKPOINT="${BC_ACTOR_CHECKPOINT:-logs/residual_actor_bc/hard_pi05_cached_base_k50_s001/residual_actor_bc.pt}"
OUT="${OUT:-logs/hard_eval_v1/failure_review/video_capture_v1}"

REPO=/home/user/wyw/RLinf
cd "$REPO"
source scripts/setup_master40_robotwin_env.sh
export ROBOTWIN_PATH=/home/user/wyw/RoboTwin
export PYTHONPATH=/home/user/wyw/RLinf:/home/user/wyw/RoboTwin:${PYTHONPATH:-}

mkdir -p "$OUT"
printf "%s\n" $SEEDS > "$OUT/seeds.txt"

echo "gpu=$GPU_ID max_steps=$MAX_STEPS out=$OUT" | tee "$OUT/run.log"
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
    mkdir -p "$seed_dir"
    echo "RUN condition=$condition seed=$seed" | tee -a "$OUT/run.log"

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
      --save-video
      --save-debug
      --video-fps 10
      --video-frame-mode replan
    )

    if [ "$condition" = "zero_k50" ]; then
      run_rollout python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py \
        "${common_args[@]}" \
        --chunk-aware-gate-checkpoint logs/chunk_aware_gate/full/gate_head/chunk_aware_gate_head.pt \
        --execution-mode gate_controlled_hybrid \
        --enable-residual-intervention \
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
with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

metrics = {}
for row in rows:
    condition = row["condition"]
    item = metrics.setdefault(condition, {"num_episodes": 0, "successes": 0})
    item["num_episodes"] += 1
    item["successes"] += int(str(row.get("success", "")).lower() == "true")
for item in metrics.values():
    item["success_rate"] = item["successes"] / item["num_episodes"] if item["num_episodes"] else 0.0
(out / "aggregate_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True))
PY
