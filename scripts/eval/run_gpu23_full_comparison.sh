#!/usr/bin/env bash
# Full comparison eval on physical GPUs 2 and 3.
#
# Targets:
#   Qwen3-1.7B:      original vs GRPO vs G2RPO-A
#   Qwen3-1.7B-Base: original vs GRPO vs G2RPO-A fixed-mask after loss fix
#                    vs G2RPO-A before loss fix
#
# Usage:
#   scripts/eval/run_gpu23_full_comparison.sh [extra args forwarded to run_eval.sh]
#
# Examples:
#   scripts/eval/run_gpu23_full_comparison.sh --limit 8
#   scripts/eval/run_gpu23_full_comparison.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
LCB_RELEASE="${LCB_RELEASE:-release_v1}"
OUT_ROOT="${OUT_ROOT:-eval_results/full_compare_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/eval_full_compare_$(date +%Y%m%d_%H%M%S)}"

# If set to 1, wait for missing checkpoints instead of skipping them. Useful
# when launching right before the current training writes checkpoint-60.
WAIT_FOR_CKPT="${WAIT_FOR_CKPT:-0}"
WAIT_SECONDS="${WAIT_SECONDS:-7200}"
POLL_SECONDS="${POLL_SECONDS:-60}"

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

wait_for_ckpt() {
    local ckpt="$1"
    local waited=0
    while [[ ! -d "$ckpt" && "$WAIT_FOR_CKPT" == "1" && "$waited" -lt "$WAIT_SECONDS" ]]; do
        echo "[$(date '+%F %T')] waiting for checkpoint: $ckpt"
        sleep "$POLL_SECONDS"
        waited=$((waited + POLL_SECONDS))
    done
    [[ -d "$ckpt" ]]
}

run_one() {
    local name="$1"
    local gpu="$2"
    local ckpt="$3"
    shift 3
    local log="$LOG_DIR/${name}.log"
    local out="$OUT_ROOT/$name"

    if ! wait_for_ckpt "$ckpt"; then
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
    LCB_RELEASE="$LCB_RELEASE" \
    OUT_ROOT="$out" \
        bash scripts/eval/run_eval.sh "$ckpt" "$@" >> "$log" 2>&1

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

echo "logs: $LOG_DIR"
echo "outputs: $OUT_ROOT"
echo "GPU queue A: $GPU_A"
echo "GPU queue B: $GPU_B"
echo "gpu_memory_utilization: $GPU_MEM_UTIL"
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
echo "summaries:"
for f in "$OUT_ROOT"/*/*/*_summary.json; do
    [[ -e "$f" ]] || continue
    echo "-- $f --"
    cat "$f"
    echo
done

exit "$status"
