#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PYTHON="${PYTHON:-python}"
PI05_CONFIG="${PI05_CONFIG:-pi05_aloha_robotwin_handover}"
PI05_CHECKPOINT="${PI05_CHECKPOINT:-/tmp/run_extract/pytorch_checkpoint/model.safetensors}"
DATASET_DIR="${DATASET_DIR:-/tmp/run_extract/handover_expert_with_gate/}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/run_extract/gate_head_vla_feature/full_multigpu}"
FEATURE_SOURCE="${FEATURE_SOURCE:-action_head_hidden}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SEED="${SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/run_extract/cache/xdg}"
export HF_HOME="${HF_HOME:-/tmp/run_extract/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-/tmp/run_extract/cache/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/run_extract/cache/torchinductor}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_SHARDS="${#GPU_LIST[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "GPUS is empty." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/shards" "$OUTPUT_DIR/logs"

echo "Launching $NUM_SHARDS feature extraction shards on GPUs: $GPUS"
for shard_idx in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$shard_idx]}"
  shard_dir="$OUTPUT_DIR/shards/shard_${shard_idx}"
  log_file="$OUTPUT_DIR/logs/shard_${shard_idx}.log"
  mkdir -p "$shard_dir"
  echo "  shard $shard_idx/$NUM_SHARDS -> GPU $gpu, log $log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/extract_pi05_features_for_gate.py \
    --pi05-config "$PI05_CONFIG" \
    --pi05-checkpoint "$PI05_CHECKPOINT" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$shard_dir" \
    --feature-source "$FEATURE_SOURCE" \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --train-ratio "$TRAIN_RATIO" \
    --seed "$SEED" \
    --episode-shard-index "$shard_idx" \
    --episode-num-shards "$NUM_SHARDS" \
    $EXTRA_ARGS \
    > "$log_file" 2>&1 &
done

failed=0
for job in $(jobs -p); do
  if ! wait "$job"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "At least one shard failed. Check logs under $OUTPUT_DIR/logs." >&2
  exit 1
fi

"$PYTHON" scripts/merge_gate_feature_shards.py \
  --shards-dir "$OUTPUT_DIR/shards" \
  --out-dir "$OUTPUT_DIR/features"

echo "Merged features written to: $OUTPUT_DIR/features"
