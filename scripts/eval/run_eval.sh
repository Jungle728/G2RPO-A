#!/usr/bin/env bash
# Run HumanEval + LiveCodeBench (release_v1) on a checkpoint.
#
# Usage:
#   scripts/eval/run_eval.sh <checkpoint_path> [extra args forwarded to both runners]
#
# Examples:
#   # quick smoke test, 8 problems each, on a 1.7B ckpt
#   scripts/eval/run_eval.sh archives/run1_format01/model_dir --limit 8
#
#   # full eval on the latest run
#   scripts/eval/run_eval.sh data/fix306code1k1epoch_fmt05
#
# Environment knobs (override on the command line):
#   GPU=3                          # which physical GPU to use
#   GPU_MEM_UTIL=0.5               # vllm gpu_memory_utilization
#   MAX_MODEL_LEN=8192
#   MAX_TOKENS=4096
#   OUT_ROOT=eval_results
#   LCB_RELEASE=release_v1

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
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
OUT_ROOT="${OUT_ROOT:-eval_results}"
LCB_RELEASE="${LCB_RELEASE:-release_v1}"

ckpt_name="$(basename "$(realpath "$CKPT")")"
out_dir="${OUT_ROOT}/${ckpt_name}"
mkdir -p "$out_dir"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/ubuntu/miniconda3/envs/g2rpoa-1/bin/python"

shared=(
    --model "$CKPT"
    --max_model_len "$MAX_MODEL_LEN"
    --max_tokens "$MAX_TOKENS"
    --gpu_memory_utilization "$GPU_MEM_UTIL"
)

export HF_HOME=/share/users/luhailun/hf_cache
export HF_DATASETS_CACHE=/share/users/luhailun/hf_cache/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Avoid e2b accidentally getting picked up; eval uses local subprocess only.
unset E2B_API_KEY || true

echo "[run_eval] ckpt=$CKPT"
echo "[run_eval] gpu=$GPU  mem_util=$GPU_MEM_UTIL  out=$out_dir"
echo

echo "==== HumanEval ===="
"$PY" "$REPO_ROOT/scripts/eval/humaneval.py" \
    "${shared[@]}" \
    --out_dir "$out_dir" \
    "$@"
echo

echo "==== LiveCodeBench ($LCB_RELEASE) ===="
"$PY" "$REPO_ROOT/scripts/eval/livecodebench.py" \
    "${shared[@]}" \
    --release "$LCB_RELEASE" \
    --out_dir "$out_dir" \
    "$@"

echo
echo "==== summary ===="
for f in "$out_dir"/*_summary.json; do
    echo "-- $f --"
    cat "$f"
    echo
done
