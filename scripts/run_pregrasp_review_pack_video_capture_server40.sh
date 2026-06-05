#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
OUT="${OUT:-logs/pregrasp_k50_review/review_pack_v1}"
SEEDS="${SEEDS:-100100000 100100004}"
MAX_STEPS="${MAX_STEPS:-600}"
VIDEO_FPS="${VIDEO_FPS:-20}"

REPO=/home/user/wyw/RLinf
cd "$REPO"
source scripts/setup_master40_robotwin_env.sh
export ROBOTWIN_PATH=/home/user/wyw/RoboTwin
export PYTHONPATH=/home/user/wyw/RLinf:/home/user/wyw/RoboTwin:${PYTHONPATH:-}

mkdir -p "$OUT"
printf "%s\n" $SEEDS > "$OUT/seeds.txt"
echo "gpu=$GPU_ID out=$OUT seeds=$SEEDS max_steps=$MAX_STEPS" | tee "$OUT/run.log"

run_one() {
  local condition="$1"
  local seed="$2"
  local seed_dir="$OUT/per_condition/$condition/seed_$seed"
  rm -rf "$seed_dir"
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
    --video-fps "$VIDEO_FPS"
    --video-frame-mode executed_step
  )

  if [ "$condition" = "handover_only_k50" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py \
      "${common_args[@]}" \
      --chunk-aware-gate-checkpoint logs/chunk_aware_gate/full/gate_head/chunk_aware_gate_head.pt \
      --execution-mode handover_only_k50 \
      --enable-residual-intervention \
      --ee16-execution-strategy pointwise \
      --residual-actor zero \
      --residual-scale 1.0 \
      --residual-horizon-k 50 2>&1 | tee -a "$OUT/run.log"
  elif [ "$condition" = "pregrasp_plus_handover_k50" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py \
      "${common_args[@]}" \
      --chunk-aware-gate-checkpoint logs/chunk_aware_gate/full/gate_head/chunk_aware_gate_head.pt \
      --execution-mode pregrasp_plus_handover_k50 \
      --enable-residual-intervention \
      --enable-pregrasp-intervention \
      --gripper-close-direction lower \
      --gripper-close-threshold 0.25 \
      --gripper-closing-delta-threshold 0.5 \
      --pregrasp-cooldown-steps 50 \
      --ee16-execution-strategy pointwise \
      --residual-actor zero \
      --residual-scale 1.0 \
      --residual-horizon-k 50 2>&1 | tee -a "$OUT/run.log"
  else
    echo "unsupported condition=$condition" >&2
    exit 2
  fi
}

for seed in $SEEDS; do
  run_one handover_only_k50 "$seed"
  run_one pregrasp_plus_handover_k50 "$seed"
done

python3 scripts/build_pregrasp_review_pack.py --pack-dir "$OUT"
