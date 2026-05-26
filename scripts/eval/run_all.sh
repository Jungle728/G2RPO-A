#!/usr/bin/env bash
# Run scripts/eval/run_eval.sh sequentially for the three checkpoints we care
# about: pretrained Qwen3-1.7B, run-1 (format weight 0.1) and run-3 (1.0).
#
# We run them sequentially rather than in parallel so the single GPU we are
# allowed to use (GPU 0 here) is not split between processes. Each ckpt's
# results land under eval_results/<short-name>/.

set -uo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

GPU="${GPU:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
LOG_DIR="${LOG_DIR:-logs/eval}"
mkdir -p "$LOG_DIR"

# label|path
declare -a CKPTS=(
    "base|/share/models/Qwen3/Qwen3-1.7B"
    "run1_format01|$ROOT/archives/run1_format01/model_dir"
    "run3_format10|$ROOT/data/fix306code1k1epoch_fmt10"
)

for entry in "${CKPTS[@]}"; do
    name="${entry%%|*}"
    path="${entry##*|}"
    echo "[$(date +%H:%M:%S)] eval: $name  ckpt=$path"
    if [[ ! -d "$path" ]]; then
        echo "skip ($path not found)"
        continue
    fi
    GPU="$GPU" GPU_MEM_UTIL="$GPU_MEM_UTIL" OUT_ROOT="eval_results" \
        bash scripts/eval/run_eval.sh "$path" \
        > "$LOG_DIR/${name}.log" 2>&1
    canon="$(basename "$(realpath "$path")")"
    if [[ "$canon" != "$name" && -d "eval_results/$canon" ]]; then
        rm -rf "eval_results/$name"
        mv "eval_results/$canon" "eval_results/$name"
    fi
    echo "[$(date +%H:%M:%S)] $name done -> eval_results/$name"
done

echo
echo "all evaluations finished"
for s in eval_results/*/*_summary.json; do
    echo "-- $s --"
    cat "$s"
    echo
done
