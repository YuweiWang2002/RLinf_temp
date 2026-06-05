#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-1}"
LIMIT="${LIMIT:-5}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
OUT="${OUT:-logs/hard_eval_v1/bc_hard_k50_s001_eval${LIMIT}}"

REPO=/home/user/wyw/RLinf
cd "$REPO"
source scripts/setup_master40_robotwin_env.sh
export ROBOTWIN_PATH=/home/user/wyw/RoboTwin
export PYTHONPATH=/home/user/wyw/RLinf:/home/user/wyw/RoboTwin:${PYTHONPATH:-}

mkdir -p "$OUT"

python3 - "$LIMIT" "$OUT/seeds.txt" <<'PY'
import csv
import sys
from pathlib import Path

limit = int(sys.argv[1])
out_path = Path(sys.argv[2])
rows = list(csv.DictReader(Path("logs/hard_eval_v1/summary.csv").open()))
seeds = [row["seed"] for row in rows[:limit]]
out_path.write_text("\n".join(seeds) + "\n")
print(" ".join(seeds))
PY

echo "gpu=$GPU_ID limit=$LIMIT max_steps=$MAX_STEPS out=$OUT" | tee "$OUT/run.log"
echo "seeds=$(tr '\n' ' ' < "$OUT/seeds.txt")" | tee -a "$OUT/run.log"

while read -r seed; do
  [ -n "$seed" ] || continue
  seed_dir="$OUT/seed_$seed"
  rm -rf "$seed_dir"
  mkdir -p "$seed_dir"
  echo "RUN seed=$seed" | tee -a "$OUT/run.log"

  video_args=()
  if [ "$SAVE_VIDEO" = "1" ]; then
    video_args=(--save-video --save-debug --video-fps 10 --video-frame-mode replan)
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py \
    --config pi05_aloha_robotwin_handover \
    --checkpoint /nfs/data3/rlinf_data/pytorch_checkpoint/model.safetensors \
    --norm-stats-path /nfs/data3/rlinf_data/pytorch_checkpoint/handover_expert/norm_stats.json \
    --chunk-aware-gate-checkpoint logs/chunk_aware_gate/full/gate_head/chunk_aware_gate_head.pt \
    --execution-mode gate_controlled_hybrid \
    --enable-residual-intervention \
    --ee16-execution-strategy pointwise \
    --residual-actor bc \
    --residual-actor-checkpoint logs/residual_actor_bc/hard_pi05_cached_base_k50_s001/residual_actor_bc.pt \
    --enable-learned-residual-control \
    --residual-scale 1.0 \
    --residual-horizon-k 50 \
    --max-steps "$MAX_STEPS" \
    --num-episodes 1 \
    --seed "$seed" \
    --save-dir "$seed_dir" \
    --device cuda \
    --planner-backend curobo \
    "${video_args[@]}" 2>&1 | tee -a "$OUT/run.log"
done < "$OUT/seeds.txt"

python3 - "$OUT" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for summary_path in sorted(out.glob("seed_*/summary.json")):
    data = json.loads(summary_path.read_text())
    episode_rows = data.get("episodes") or []
    for row in episode_rows:
        row = dict(row)
        row["run_dir"] = str(summary_path.parent)
        rows.append(row)

fieldnames = sorted({key for row in rows for key in row})
summary_csv = out / "summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def as_float(row, key, default=0.0):
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)

successes = [row for row in rows if str(row.get("success", "")).lower() == "true"]
metrics = {
    "num_episodes": len(rows),
    "successes": len(successes),
    "success_rate": len(successes) / len(rows) if rows else 0.0,
    "mean_episode_len": sum(as_float(row, "episode_length") for row in rows) / len(rows)
    if rows
    else 0.0,
    "mean_num_interventions": sum(as_float(row, "num_interventions") for row in rows) / len(rows)
    if rows
    else 0.0,
    "mean_total_ee_steps": sum(as_float(row, "total_ee_intervention_steps") for row in rows) / len(rows)
    if rows
    else 0.0,
    "max_applied_norm": max((as_float(row, "applied_norm_max") for row in rows), default=0.0),
    "nan_inf_total": sum(as_float(row, "nan_inf_count") for row in rows),
    "max_saturation_ratio": max((as_float(row, "saturation_ratio") for row in rows), default=0.0),
}
(out / "aggregate_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True))
PY
