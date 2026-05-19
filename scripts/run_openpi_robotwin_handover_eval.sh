#!/usr/bin/env bash
# Run a 10-episode OpenPI pi05 RoboTwin handover eval on server40.
set -euo pipefail

cd /home/user/wyw/RLinf
source scripts/setup_master40_robotwin_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:-logs/openpi_handover_eval_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-data/openpi_handover_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR" "$DATA_DIR"

python scripts/eval_openpi_robotwin_handover.py \
  --episodes "${EPISODES:-10}" \
  --max-steps "${MAX_STEPS:-400}" \
  --device cuda:0 \
  --output-dir "$OUTPUT_DIR" \
  --data-dir "$DATA_DIR" \
  "$@"
