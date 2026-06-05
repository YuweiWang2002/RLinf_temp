#!/usr/bin/env bash
set -euo pipefail

REPO=/home/user/wyw/RLinf
SESSION="${SESSION:-codex_pregrasp_review_videos}"
OUT="${OUT:-logs/pregrasp_k50_review/review_pack_v1}"
LOG_DIR="${LOG_DIR:-logs/pregrasp_k50_review}"

cd "$REPO"
mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "bash -lc 'cd $REPO && GPU_ID=${GPU_ID:-2} OUT=$OUT bash scripts/run_pregrasp_review_pack_video_capture_server40.sh > ${LOG_DIR}/review_pack_v1.tmux.log 2>&1; echo \$? > ${LOG_DIR}/review_pack_v1.exit_code'"
tmux list-sessions | grep "$SESSION"
