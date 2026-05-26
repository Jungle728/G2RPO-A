#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-g2rpoa_guided_gpu23_$(date +%Y%m%d_%H%M%S)}"
GPU_PAIR="${GPU_PAIR:-2,3}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29579}"
CONFIG="${CONFIG:-recipes/config/G2RPO-Atrain_g2rpoa.yaml}"
PYTHON_ENV="${PYTHON_ENV:-/home/ubuntu/miniconda3/envs/g2rpoa-1}"

OUTPUT_DIR="data/${RUN_NAME}"
STEP_DIR="logs/steps_${RUN_NAME}"
TRAIN_LOG="logs/train_${RUN_NAME}.log"

if [[ -z "${E2B_API_KEY:-}" ]]; then
  echo "E2B_API_KEY is required for code_reward." >&2
  exit 1
fi

if pgrep -af 'accelerate|src/open_r1/grpo.py' >/dev/null; then
  echo "A training process is already running. Stop it before launching this experiment." >&2
  pgrep -af 'accelerate|src/open_r1/grpo.py' >&2 || true
  exit 1
fi

if [[ -e "$OUTPUT_DIR" || -e "$STEP_DIR" || -e "$TRAIN_LOG" ]]; then
  echo "Refusing to overwrite existing artifacts:" >&2
  echo "  $OUTPUT_DIR" >&2
  echo "  $STEP_DIR" >&2
  echo "  $TRAIN_LOG" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$STEP_DIR" logs data

echo "[$(date '+%F %T')] Starting G2RPO-A guided run: ${RUN_NAME}"
echo "  gpu_pair=${GPU_PAIR} (visible cuda:0 trains, visible cuda:1 runs vLLM)"
echo "  output_dir=${OUTPUT_DIR}"
echo "  step_dir=${STEP_DIR}"
echo "  train_log=${TRAIN_LOG}"

env \
  HF_HOME=/share/users/luhailun/hf_cache \
  HF_DATASETS_CACHE=/share/users/luhailun/hf_cache/datasets \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES="$GPU_PAIR" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ACCELERATE_LOG_LEVEL=info \
  SWANLAB_MODE=disabled \
  E2B_API_KEY="$E2B_API_KEY" \
  STEP_LOG_DIR="$STEP_DIR" \
  PYTHONPATH=src \
  "$PYTHON_ENV/bin/accelerate" launch \
    --num_processes=1 \
    --main_process_port="$MAIN_PROCESS_PORT" \
    src/open_r1/grpo.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    > "$TRAIN_LOG" 2>&1

echo "[$(date '+%F %T')] Finished G2RPO-A guided run: ${RUN_NAME}"
