#!/usr/bin/env bash
# Full MATH-500 comparison on physical GPUs 2 and 3.
#
# Targets:
#   Qwen3-1.7B:      original vs GRPO vs G2RPO-A
#   Qwen3-1.7B-Base: original vs GRPO vs G2RPO-A fixed-mask after loss fix
#                    vs G2RPO-A before loss fix
#
# Usage:
#   scripts/eval/run_gpu23_math_comparison.sh [extra args forwarded to math500.py]
#
# Examples:
#   scripts/eval/run_gpu23_math_comparison.sh --limit 20
#   scripts/eval/run_gpu23_math_comparison.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-3584}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-eval_results/math_compare_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/math_compare_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR" "$OUT_ROOT"

# label|path|queue
declare -a TARGETS=(
    "qwen3_1.7b_original|/share/models/Qwen3/Qwen3-1.7B|A"
    "qwen3_1.7b_grpo|$ROOT/archives/grpo_math_cl200_baseline_20260526_075134|B"
    "qwen3_1.7b_g2rpoa|$ROOT/archives/g2rpoa_math_cl200_20260525_233548|A"
    "qwen3_1.7b_base_original|/share/models/Qwen3/Qwen3-1.7B-Base|B"
    "qwen3_1.7b_base_grpo|$ROOT/archives/grpo_math_cl200_baseline_qwen3_1.7b_base_20260526_184219/data/checkpoint-60|A"
    "qwen3_1.7b_base_g2rpoa_loss_fixed|$ROOT/data/g2rpoa_math_cl200_qwen3_1.7b_base_fixedmask_3epoch_20260526_210716/checkpoint-60|B"
    "qwen3_1.7b_base_g2rpoa_loss_old|$ROOT/archives/g2rpoa_math_cl200_qwen3_1.7b_base_20260526_124027/data/checkpoint-60|A"
)

run_one() {
    local name="$1"
    local gpu="$2"
    local ckpt="$3"
    shift 3
    local log="$LOG_DIR/${name}.log"
    local out="$OUT_ROOT/$name"

    if [[ ! -d "$ckpt" ]]; then
        echo "[$(date '+%F %T')] skip $name: checkpoint not found: $ckpt" | tee "$log"
        return 0
    fi

    mkdir -p "$out"
    echo "[$(date '+%F %T')] start $name on GPU $gpu" | tee "$log"
    echo "checkpoint: $ckpt" | tee -a "$log"
    echo "out: $out" | tee -a "$log"

    GPU="$gpu" \
    GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MAX_TOKENS="$MAX_TOKENS" \
    TEMPERATURE="$TEMPERATURE" \
    TOP_P="$TOP_P" \
    TOP_K="$TOP_K" \
    SEED="$SEED" \
    OUT_ROOT="$out" \
        bash scripts/eval/run_math_eval.sh "$ckpt" "$@" >> "$log" 2>&1

    echo "[$(date '+%F %T')] done $name" | tee -a "$log"
}

run_queue() {
    local queue="$1"
    local gpu="$2"
    shift 2
    local entry name path q
    for entry in "${TARGETS[@]}"; do
        IFS='|' read -r name path q <<< "$entry"
        [[ "$q" == "$queue" ]] || continue
        run_one "$name" "$gpu" "$path" "$@"
    done
}

print_summary() {
    OUT_ROOT="$OUT_ROOT" "$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
runs = [
    ("Qwen3-1.7B original", "qwen3_1.7b_original/Qwen3-1.7B"),
    ("Qwen3-1.7B GRPO", "qwen3_1.7b_grpo/grpo_math_cl200_baseline_20260526_075134"),
    ("Qwen3-1.7B G2RPO-A", "qwen3_1.7b_g2rpoa/g2rpoa_math_cl200_20260525_233548"),
    ("Qwen3-1.7B-Base original", "qwen3_1.7b_base_original/Qwen3-1.7B-Base"),
    ("Qwen3-1.7B-Base GRPO", "qwen3_1.7b_base_grpo/checkpoint-60"),
    ("Qwen3-1.7B-Base G2RPO-A fixed", "qwen3_1.7b_base_g2rpoa_loss_fixed/checkpoint-60"),
    ("Qwen3-1.7B-Base G2RPO-A old", "qwen3_1.7b_base_g2rpoa_loss_old/checkpoint-60"),
]
print("| model | MATH-500 pass@1 |")
print("|---|---:|")
for label, rel in runs:
    path = root / rel / "math500_summary.json"
    if not path.exists():
        print(f"| {label} | missing |")
        continue
    s = json.loads(path.read_text())
    print(f"| {label} | {s['n_passed']}/{s['n_problems']} = {s['pass_at_1']*100:.2f}% |")
PY
}

PY="/home/ubuntu/miniconda3/envs/g2rpoa-1/bin/python"

echo "logs: $LOG_DIR"
echo "outputs: $OUT_ROOT"
echo "GPU queue A: $GPU_A"
echo "GPU queue B: $GPU_B"
echo "gpu_memory_utilization: $GPU_MEM_UTIL"
echo "decoding: T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K max_tokens=$MAX_TOKENS seed=$SEED"
echo
printf '%s\n' "${TARGETS[@]}"
echo

run_queue A "$GPU_A" "$@" &
pid_a=$!
run_queue B "$GPU_B" "$@" &
pid_b=$!

status=0
wait "$pid_a" || status=$?
wait "$pid_b" || status=$?

echo
echo "finished with status=$status"
echo "logs: $LOG_DIR"
echo "outputs: $OUT_ROOT"
echo
print_summary

exit "$status"
