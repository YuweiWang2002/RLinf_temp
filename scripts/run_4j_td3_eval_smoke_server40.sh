#!/usr/bin/env bash
set -uo pipefail

cd /home/user/wyw/RLinf
source scripts/setup_master40_robotwin_env.sh
export ROBOTWIN_PATH=/home/user/wyw/RoboTwin
export PYTHONPATH=/home/user/wyw/RLinf:/home/user/wyw/RoboTwin:${PYTHONPATH:-}

GPU_ID=${GPU_ID:-4}
LOG_ROOT=${LOG_ROOT:-logs/eval_smoke_4j_hard_expert}
mkdir -p "$LOG_ROOT"

COMMON_ARGS=(
  --config pi05_aloha_robotwin_handover
  --checkpoint /nfs/data3/rlinf_data/pytorch_checkpoint/model.safetensors
  --norm-stats-path /nfs/data3/rlinf_data/pytorch_checkpoint/handover_expert/norm_stats.json
  --chunk-aware-gate-checkpoint logs/chunk_aware_gate/full/gate_head/chunk_aware_gate_head.pt
  --execution-mode gate_controlled_hybrid
  --enable-residual-intervention
  --ee16-execution-strategy pointwise
  --residual-actor td3
  --enable-learned-residual-control
  --residual-scale 1.0
  --residual-horizon-k 10
  --device cuda
  --planner-backend curobo
)

run_rollout() {
  local name="$1"
  shift
  local log_path="$LOG_ROOT/${name}.log"
  echo "START ${name}" | tee -a "$LOG_ROOT/run.log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python3 scripts/rollout_pi05_gate_controlled_hybrid_zero_residual.py "$@" \
    2>&1 | tee "$log_path"
  local code=${PIPESTATUS[0]}
  echo "END ${name} exit=${code}" | tee -a "$LOG_ROOT/run.log"
}

run_rollout normal_s005 \
  "${COMMON_ARGS[@]}" \
  --residual-actor-checkpoint logs/residual_td3/mainline_4j_hard_expert_s005/actor_td3.pt \
  --max-steps 600 \
  --num-episodes 5 \
  --save-dir logs/gate_controlled_hybrid/td3_hard_expert_s005_eval5

run_rollout normal_s010 \
  "${COMMON_ARGS[@]}" \
  --residual-actor-checkpoint logs/residual_td3/mainline_4j_hard_expert_s010/actor_td3.pt \
  --max-steps 600 \
  --num-episodes 5 \
  --save-dir logs/gate_controlled_hybrid/td3_hard_expert_s010_eval5

HARD_SEEDS=(100100005 100100029 100100033 100100034 100100035)
for seed in "${HARD_SEEDS[@]}"; do
  run_rollout "hard_s005_${seed}" \
    "${COMMON_ARGS[@]}" \
    --residual-actor-checkpoint logs/residual_td3/mainline_4j_hard_expert_s005/actor_td3.pt \
    --max-steps 800 \
    --num-episodes 1 \
    --seed "$seed" \
    --save-dir "logs/hard_eval_v1/td3_hard_expert_s005_eval5/seed_${seed}"
done

for seed in "${HARD_SEEDS[@]}"; do
  run_rollout "hard_s010_${seed}" \
    "${COMMON_ARGS[@]}" \
    --residual-actor-checkpoint logs/residual_td3/mainline_4j_hard_expert_s010/actor_td3.pt \
    --max-steps 800 \
    --num-episodes 1 \
    --seed "$seed" \
    --save-dir "logs/hard_eval_v1/td3_hard_expert_s010_eval5/seed_${seed}"
done

date > "$LOG_ROOT/done.txt"
