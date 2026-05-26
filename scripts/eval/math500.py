"""MATH-500 evaluator for G2RPO-A checkpoints.

Runs the model on `HuggingFaceH4/MATH-500` (500 problems, paper §5.1
benchmark) using vLLM, then grades each completion with `math_verify`
(same grader used by the training-time `accuracy_reward`).

Decoding follows the paper: temperature=0.6, top_p=0.95, top_k=20,
single sample (pass@1).

Usage:
    python scripts/eval/math500.py \
        --model data/g2rpoa_math_cl200_20260525_233548 \
        --output_dir eval_results/g2rpoa_cl200 \
        --gpu_memory_utilization 0.85

Environment:
    Caller is expected to set CUDA_VISIBLE_DEVICES, HF_*, etc. (see
    `scripts/eval/run_math_eval.sh`).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

# Allow `from common import ...` when invoked as a script.
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    GenConfig,
    build_chat,
    generate_chat,
    load_vllm,
    write_jsonl,
    write_summary,
)


SYSTEM_PROMPT = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed "
    "responses. You first think about the reasoning process as an internal "
    "monologue and then provide the user with the answer. Respond in the "
    "following format: <think>\n...\n</think>\n<answer>\n...\n</answer>. "
    "Put your final mathematical answer inside \\boxed{...} within the "
    "<answer> block."
)


def _wrap_for_math_verify(answer: str) -> str:
    a = "" if answer is None else str(answer).strip()
    if not a:
        return a
    if (a.startswith("$") and a.endswith("$")) or "\\boxed" in a:
        return a
    return f"${a}$"


def _grade_one(completion: str, gold: str) -> tuple[bool, str]:
    """Return (passed, diagnostic). Mirrors `accuracy_reward` from training."""
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    gold_parsed = parse(
        gold,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if not gold_parsed:
        # Unparseable gold; we conservatively skip grading and mark as missing.
        return False, "gold_unparsed"
    answer_parsed = parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    if not answer_parsed:
        return False, "answer_unparsed"
    try:
        ok = bool(verify(answer_parsed, gold_parsed))
    except Exception as e:  # noqa: BLE001
        return False, f"verify_error:{type(e).__name__}"
    return ok, "match" if ok else "mismatch"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path.")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--dataset",
        default="HuggingFaceH4/MATH-500",
        help="Pre-cached HF dataset id.",
    )
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None, help="Optional: only evaluate first N.")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_tokens", type=int, default=3584)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    completions_path = out_dir / "math500.jsonl"
    summary_path = out_dir / "math500_summary.json"

    print(f"[math500] model={args.model}")
    print(f"[math500] dataset={args.dataset} split={args.split} limit={args.limit}")
    print(f"[math500] decoding T={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    print(f"[math500] max_model_len={args.max_model_len} max_tokens={args.max_tokens}")
    print(f"[math500] writing to {out_dir}/")

    # ---------- Load dataset ----------
    from datasets import load_dataset

    ds = load_dataset(args.dataset)[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)
    print(f"[math500] {n} problems")

    # ---------- Build chat prompts ----------
    chats = [build_chat(row["problem"], system=args.system_prompt) for row in ds]

    # ---------- Generate ----------
    t0 = time.time()
    llm = load_vllm(
        args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    cfg = GenConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=1,
        seed=args.seed,
    )
    completions = generate_chat(llm, chats, cfg)
    gen_secs = time.time() - t0
    print(f"[math500] generated {len(completions)} completions in {gen_secs:.1f}s")

    # ---------- Grade ----------
    records = []
    by_subject: dict[str, list[bool]] = {}
    by_level: dict[int, list[bool]] = {}
    for row, completion in zip(ds, completions):
        gold = _wrap_for_math_verify(row["answer"])
        passed, diag = _grade_one(completion, gold)
        records.append(
            {
                "task_id": row.get("unique_id", "") or "",
                "subject": row.get("subject", ""),
                "level": row.get("level", -1),
                "problem": row["problem"],
                "gold": row["answer"],
                "completion": completion,
                "passed": passed,
                "diagnostic": diag,
            }
        )
        by_subject.setdefault(row.get("subject", ""), []).append(passed)
        by_level.setdefault(row.get("level", -1), []).append(passed)

    write_jsonl(completions_path, records)

    # ---------- Summary ----------
    n_passed = sum(1 for r in records if r["passed"])
    diag_counts = Counter(r["diagnostic"] for r in records)
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_problems": n,
        "n_passed": n_passed,
        "pass_at_1": n_passed / n if n else 0.0,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "diagnostics": dict(diag_counts),
        "by_subject": {
            k: {"n": len(v), "n_passed": sum(v), "pass_at_1": sum(v) / len(v)}
            for k, v in sorted(by_subject.items())
        },
        "by_level": {
            str(k): {"n": len(v), "n_passed": sum(v), "pass_at_1": sum(v) / len(v)}
            for k, v in sorted(by_level.items())
        },
        "gen_seconds": round(gen_secs, 1),
    }
    write_summary(summary_path, summary)

    print(
        f"[math500] pass@1 = {n_passed}/{n} = {summary['pass_at_1']:.2%}  "
        f"diagnostics={dict(diag_counts)}"
    )
    print(f"[math500] per-level: {summary['by_level']}")
    print(f"[math500] saved {completions_path} and {summary_path}")


if __name__ == "__main__":
    main()
