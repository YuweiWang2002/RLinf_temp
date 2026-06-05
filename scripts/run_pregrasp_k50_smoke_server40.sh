#!/usr/bin/env bash
set -euo pipefail

REPO=/home/user/wyw/RLinf
cd "$REPO"

mkdir -p logs/pregrasp_k50_smoke

tmux kill-session -t codex_pregrasp_only_smoke 2>/dev/null || true
tmux kill-session -t codex_pregrasp_plus_smoke 2>/dev/null || true

tmux new-session -d -s codex_pregrasp_only_smoke \
  "bash -lc 'cd $REPO && GPU_ID=0 MODE=normal LIMIT=5 CONDITIONS=pregrasp_only_k50 OUT=logs/pregrasp_k50_smoke/pregrasp_only_eval5 SKIP_EXISTING=0 GRIPPER_CLOSE_DIRECTION=lower GRIPPER_CLOSE_THRESHOLD=0.25 GRIPPER_CLOSING_DELTA_THRESHOLD=0.5 bash scripts/run_k50_eval_matrix_server40.sh > logs/pregrasp_k50_smoke/pregrasp_only_eval5.tmux.log 2>&1; echo \$? > logs/pregrasp_k50_smoke/pregrasp_only_eval5.exit_code'"

tmux new-session -d -s codex_pregrasp_plus_smoke \
  "bash -lc 'cd $REPO && GPU_ID=1 MODE=normal LIMIT=5 CONDITIONS=pregrasp_plus_handover_k50 OUT=logs/pregrasp_k50_smoke/pregrasp_plus_handover_eval5 SKIP_EXISTING=0 GRIPPER_CLOSE_DIRECTION=lower GRIPPER_CLOSE_THRESHOLD=0.25 GRIPPER_CLOSING_DELTA_THRESHOLD=0.5 bash scripts/run_k50_eval_matrix_server40.sh > logs/pregrasp_k50_smoke/pregrasp_plus_handover_eval5.tmux.log 2>&1; echo \$? > logs/pregrasp_k50_smoke/pregrasp_plus_handover_eval5.exit_code'"

tmux list-sessions | grep codex_pregrasp
