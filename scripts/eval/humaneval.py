#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# HumanEval pass@1 evaluator.
#
# Reads `openai/openai_humaneval` from the local HF cache, generates one
# completion per task with vLLM at T=0.6 / top-p=0.95 / top-k=20, extracts the
# last python code block (matching `extract_code` from training), and grades
# by running `def check(candidate): ...; check(<entry_point>)` in a fresh
# Python subprocess with a 10s timeout.

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

# allow `python scripts/eval/humaneval.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


HUMAN_EVAL_INSTRUCTION = (
    "Complete the following Python function. Return your full solution "
    "(including the original signature and docstring) inside a single fenced "
    "```python``` block. Do not include any explanation outside of the code "
    "block.\n\n"
    "```python\n{prompt}```"
)


def _grade_one(code: str, test_src: str, entry_point: str, timeout: float) -> common.ExecResult:
    """Run candidate + tests in a fresh subprocess."""
    script = (
        f"{code}\n\n"
        f"{test_src}\n\n"
        f"check({entry_point})\n"
    )
    return common.run_python(script, timeout=timeout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path or HF id of the model")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="evaluate first N tasks (smoke test)")
    ap.add_argument("--grade_timeout", type=float, default=10.0)
    ap.add_argument("--grade_workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--no_system_prompt", action="store_true",
                    help="omit the open-r1 system prompt; use raw user message only")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load dataset
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[humaneval] {len(ds)} tasks")

    # 2) build chat prompts
    chats = [
        common.build_chat(
            HUMAN_EVAL_INSTRUCTION.format(prompt=row["prompt"]),
            system=None if args.no_system_prompt else common.DEFAULT_SYSTEM,
        )
        for row in ds
    ]

    # 3) generate
    llm = common.load_vllm(
        args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    completions = common.generate_chat(
        llm,
        chats,
        common.GenConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )

    # 4) extract + grade in parallel subprocess pool
    items = []
    for row, comp in zip(ds, completions):
        code = common.extract_code(comp)
        items.append((row["task_id"], _grade_one,
                     (code, row["test"], row["entry_point"], args.grade_timeout)))
    # sequential extraction for output order, parallel grading via futures
    grades = common.parallel_grade(items, max_workers=args.grade_workers)

    # 5) write per-task records + aggregate
    records = []
    n_pass = 0
    for row, comp, g in zip(ds, completions, grades):
        if isinstance(g, tuple) and g and g[0] == "error":
            ok = False
            diag = f"grader_error: {g[1]}"
        else:
            ok = bool(g.passed)
            diag = ""
            if not ok:
                diag = (f"timeout={g.timeout} stderr=" + g.stderr[-500:].replace("\n", "\\n"))
        if ok:
            n_pass += 1
        records.append(common.TaskResult(
            task_id=row["task_id"],
            passed=ok,
            completion=comp,
            extracted=common.extract_code(comp),
            diagnostic=diag,
        ))

    common.write_jsonl(out_dir / "humaneval.jsonl", records)
    summary = {
        "benchmark": "humaneval",
        "model": args.model,
        "n_tasks": len(ds),
        "n_pass": n_pass,
        "pass@1": n_pass / len(ds) if len(ds) else 0.0,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    }
    common.write_summary(out_dir / "humaneval_summary.json", summary)
    print(f"[humaneval] pass@1 = {summary['pass@1']:.4f}  ({n_pass}/{len(ds)})")


if __name__ == "__main__":
    main()
