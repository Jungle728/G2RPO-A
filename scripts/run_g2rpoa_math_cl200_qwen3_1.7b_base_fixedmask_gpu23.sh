#!/usr/bin/env bash
# Launch G2RPO-A math training on the 200-row CL subset, model = Qwen3-1.7B-Base.
# This run name marks the corrected implementation where GRPO loss matches TRL
# and externally injected guidance tokens are masked out of the policy loss.
# GPU layout: visible cuda:0 (= phys 3) trains, visible cuda:1 (= phys 2) runs vLLM.
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-g2rpoa_math_cl200_qwen3_1.7b_base_fixedmask_$(date +%Y%m%d_%H%M%S)}"
GPU_PAIR="${GPU_PAIR:-3,2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29595}"
CONFIG="${CONFIG:-recipes/config/G2RPO-Atrain_g2rpoa_math_cl200_qwen3_1.7b_base.yaml}"
PYTHON_ENV="${PYTHON_ENV:-/home/ubuntu/miniconda3/envs/g2rpoa-1}"

OUTPUT_DIR="data/${RUN_NAME}"
STEP_DIR="logs/steps_${RUN_NAME}"
TRAIN_LOG="logs/train_${RUN_NAME}.log"

if pgrep -af 'src/open_r1/grpo\.py' >/dev/null; then
  echo "A training process is already running. Stop it before launching this experiment." >&2
  pgrep -af 'src/open_r1/grpo\.py' >&2 || true
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

echo "[$(date '+%F %T')] Starting fixed-mask G2RPO-A math CL200 run on Qwen3-1.7B-Base: ${RUN_NAME}"
echo "  gpu_pair=${GPU_PAIR} (visible cuda:0 trains, visible cuda:1 runs vLLM)"
echo "  config=${CONFIG}"
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
  PYTHONUNBUFFERED=1 \
  ACCELERATE_LOG_LEVEL=info \
  SWANLAB_MODE=disabled \
  STEP_LOG_DIR="$STEP_DIR" \
  PYTHONPATH=src \
  "$PYTHON_ENV/bin/accelerate" launch \
    --num_processes=1 \
    --main_process_port="$MAIN_PROCESS_PORT" \
    src/open_r1/grpo.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    > "$TRAIN_LOG" 2>&1

echo "[$(date '+%F %T')] Finished fixed-mask G2RPO-A math CL200 run on Qwen3-1.7B-Base: ${RUN_NAME}"
