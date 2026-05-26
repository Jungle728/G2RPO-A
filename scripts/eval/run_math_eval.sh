#!/usr/bin/env bash
# Run MATH-500 evaluation on a checkpoint.
#
# Usage:
#   scripts/eval/run_math_eval.sh <checkpoint_path> [extra args forwarded to math500.py]
#
# Examples:
#   # quick smoke (10 problems)
#   scripts/eval/run_math_eval.sh /share/models/Qwen3/Qwen3-1.7B --limit 10
#
#   # full eval on a trained checkpoint
#   scripts/eval/run_math_eval.sh data/g2rpoa_math_cl200_20260525_233548
#
#   # eval an intermediate epoch checkpoint
#   scripts/eval/run_math_eval.sh data/g2rpoa_math_cl200_20260525_233548/checkpoint-60
#
# Environment knobs (override on the command line):
#   GPU=3                          # which physical GPU to use
#   GPU_MEM_UTIL=0.85              # vLLM gpu_memory_utilization
#   MAX_MODEL_LEN=4096
#   MAX_TOKENS=3584
#   OUT_ROOT=eval_results
#   TEMPERATURE=0.6 / TOP_P=0.95 / TOP_K=20    # paper §5.1 decoding

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <checkpoint_path> [extra args]" >&2
    exit 1
fi
CKPT="$1"; shift || true

if [[ ! -d "$CKPT" ]]; then
    echo "checkpoint dir not found: $CKPT" >&2
    exit 1
fi

GPU="${GPU:-3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-3584}"
OUT_ROOT="${OUT_ROOT:-eval_results}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-42}"

ckpt_name="$(basename "$(realpath "$CKPT")")"
# If the ckpt is a checkpoint-N inside a run dir, prefix with the run name so
# different epochs don't overwrite each other under eval_results/.
parent="$(basename "$(dirname "$(realpath "$CKPT")")")"
if [[ "$ckpt_name" == checkpoint-* ]]; then
    out_dir="${OUT_ROOT}/${parent}/${ckpt_name}"
else
    out_dir="${OUT_ROOT}/${ckpt_name}"
fi
mkdir -p "$out_dir"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/ubuntu/miniconda3/envs/g2rpoa-1/bin/python"

export HF_HOME=/share/users/luhailun/hf_cache
export HF_DATASETS_CACHE=/share/users/luhailun/hf_cache/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Math grader is local (math_verify); ensure no stray e2b key leaks in.
unset E2B_API_KEY || true

echo "[run_math_eval] ckpt=$CKPT"
echo "[run_math_eval] gpu=$GPU  mem_util=$GPU_MEM_UTIL"
echo "[run_math_eval] out=$out_dir"
echo "[run_math_eval] decoding T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K max_tokens=$MAX_TOKENS seed=$SEED"
echo

"$PY" "$REPO_ROOT/scripts/eval/math500.py" \
    --model "$CKPT" \
    --output_dir "$out_dir" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --seed "$SEED" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    "$@"

echo
echo "==== summary ===="
cat "$out_dir/math500_summary.json"
echo
