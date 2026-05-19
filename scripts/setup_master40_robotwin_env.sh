#!/usr/bin/env bash
# Source this file from the RLinf repo root before running RoboTwin/OpenPI probes.
set -e

export RLINF_PATH=/home/user/wyw/RLinf
export REPO_PATH=/home/user/wyw/RLinf
export ROBOTWIN_PATH=/home/user/wyw/RoboTwin
export CUDA_HOME=/usr/local/cuda-11.8
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$HOME/.local/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROBOTWIN_PATH:$RLINF_PATH:${PYTHONPATH:-}
export ROBOT_PLATFORM=ALOHA
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}

cd "$RLINF_PATH"
source "$RLINF_PATH/.venv/bin/activate"
