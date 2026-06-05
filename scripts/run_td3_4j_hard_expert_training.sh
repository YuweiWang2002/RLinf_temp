#!/usr/bin/env bash
set -euo pipefail

cd /home/user/wyw/RLinf
source scripts/setup_master40_robotwin_env.sh
export ROBOTWIN_PATH=/home/user/wyw/RoboTwin
export PYTHONPATH=/home/user/wyw/RLinf:/home/user/wyw/RoboTwin:${PYTHONPATH:-}

REPLAY=${REPLAY:-logs/residual_replay/mainline_4j_hard_expert_merged/replay.npz}
BC_ACTOR=${BC_ACTOR:-logs/residual_actor_bc/pi05_cached_base_k10/residual_actor_bc.pt}
GPU_ID=${GPU_ID:-4}
UPDATES=${UPDATES:-1000}
BATCH_SIZE=${BATCH_SIZE:-256}
LOG_ROOT=${LOG_ROOT:-logs/residual_td3/mainline_4j_train_pair}
mkdir -p "$LOG_ROOT"

train_one() {
  local delta_max="$1"
  local out_dir="$2"
  local log_path="$3"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python3 scripts/train_residual_td3_from_replay.py \
    --replay "$REPLAY" \
    --bc-actor-checkpoint "$BC_ACTOR" \
    --output-dir "$out_dir" \
    --obs-dim 21 \
    --action-dim 3 \
    --actor-delta-max "$delta_max" \
    --actor-bc-weight 1000 \
    --actor-l2-weight 1.0 \
    --actor-bc-mode all \
    --target-noise 0.0005 \
    --target-noise-clip 0.0015 \
    --updates "$UPDATES" \
    --batch-size "$BATCH_SIZE" \
    --log-interval 100 \
    --device cuda \
    2>&1 | tee "$log_path"
}

train_one 0.005 logs/residual_td3/mainline_4j_hard_expert_s005 "$LOG_ROOT/s005.log"
echo "s005_exit=0" | tee -a "$LOG_ROOT/run.log"

train_one 0.01 logs/residual_td3/mainline_4j_hard_expert_s010 "$LOG_ROOT/s010.log"
echo "s010_exit=0" | tee -a "$LOG_ROOT/run.log"

date > "$LOG_ROOT/done.txt"
