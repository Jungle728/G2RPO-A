#!/usr/bin/env bash
# Evaluate the Qwen3-1.7B base-GRPO checkpoint and the current fixed-mask
# G2RPO-A checkpoint in parallel on physical GPUs 2 and 3.
#
# Usage:
#   scripts/eval/run_gpu23_grpo_vs_g2rpoa.sh [extra args forwarded to run_eval.sh]
#
# Examples:
#   # quick smoke test before the full run
#   scripts/eval/run_gpu23_grpo_vs_g2rpoa.sh --limit 8
#
#   # full HumanEval + LiveCodeBench release_v1
#   scripts/eval/run_gpu23_grpo_vs_g2rpoa.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

GPU_GRPO="${GPU_GRPO:-2}"
GPU_G2RPOA="${GPU_G2RPOA:-3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
LCB_RELEASE="${LCB_RELEASE:-release_v1}"
OUT_ROOT="${OUT_ROOT:-eval_results}"
LOG_DIR="${LOG_DIR:-logs/eval_gpu23_$(date +%Y%m%d_%H%M%S)}"

GRPO_CKPT="${GRPO_CKPT:-$ROOT/archives/grpo_math_cl200_baseline_qwen3_1.7b_base_20260526_184219/data/checkpoint-60}"
G2RPOA_CKPT="${G2RPOA_CKPT:-$ROOT/data/g2rpoa_math_cl200_qwen3_1.7b_base_fixedmask_3epoch_20260526_210716/checkpoint-60}"

mkdir -p "$LOG_DIR"

run_one() {
    local name="$1"
    local gpu="$2"
    local ckpt="$3"
    local log="$LOG_DIR/${name}.log"

    if [[ ! -d "$ckpt" ]]; then
        echo "[$(date '+%F %T')] skip $name: checkpoint not found: $ckpt" | tee "$log"
        return 2
    fi

    echo "[$(date '+%F %T')] start $name on GPU $gpu: $ckpt" | tee "$log"
    GPU="$gpu" \
    GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MAX_TOKENS="$MAX_TOKENS" \
    LCB_RELEASE="$LCB_RELEASE" \
    OUT_ROOT="$OUT_ROOT" \
        bash scripts/eval/run_eval.sh "$ckpt" "$@" >> "$log" 2>&1
    echo "[$(date '+%F %T')] done $name" | tee -a "$log"
}

echo "logs: $LOG_DIR"
echo "GRPO:   GPU $GPU_GRPO   $GRPO_CKPT"
echo "G2RPOA: GPU $GPU_G2RPOA $G2RPOA_CKPT"
echo

run_one "qwen3_1.7b_base_grpo_ckpt60" "$GPU_GRPO" "$GRPO_CKPT" "$@" &
pid_grpo=$!

run_one "qwen3_1.7b_g2rpoa_fixedmask_ckpt60" "$GPU_G2RPOA" "$G2RPOA_CKPT" "$@" &
pid_g2rpoa=$!

status=0
wait "$pid_grpo" || status=$?
wait "$pid_g2rpoa" || status=$?

echo
echo "finished with status=$status"
echo "logs: $LOG_DIR"
echo "summaries:"
for f in "$OUT_ROOT"/*/*_summary.json; do
    [[ -e "$f" ]] || continue
    echo "-- $f --"
    cat "$f"
    echo
done

exit "$status"
