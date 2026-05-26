#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# LiveCodeBench (code_generation_lite) pass@1 evaluator.
#
# Reads `livecodebench/code_generation_lite` *.jsonl directly from the local
# HF snapshot at $HF_HOME/hub/datasets--livecodebench--code_generation_lite.
# We avoid `datasets.load_dataset(...)` because LCB ships a loading script and
# `datasets >= 4` refuses to execute remote scripts.
#
# Decoding follows the paper (T=0.6, top-p=0.95, top-k=20, single sample),
# matching scripts/eval/humaneval.py.
#
# Grading: every problem ships {public,private}_test_cases. private cases are
# zlib+pickle base64 (the official harness format). For each test case we run
# the candidate as a fresh Python subprocess; if the row is `stdin` we pipe
# the input on stdin and string-compare stdout (whitespace-stripped); if the
# row is `functional` we wrap the candidate + a generated `Solution` driver.

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import re
import sys
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

# allow `python scripts/eval/livecodebench.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


LCB_INSTRUCTION = (
    "Solve the following coding problem using Python. Read input from stdin "
    "(or implement the requested function/class for functional problems) and "
    "print results to stdout. Return only the final solution inside a single "
    "fenced ```python``` block.\n\n"
    "{question}\n"
    "{starter}"
)


def _decode_private(raw: str) -> list[dict]:
    """LCB private_test_cases format: base64(zlib(pickle(json_str)))."""
    if not raw:
        return []
    try:
        decoded = json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))
        return decoded
    except Exception:
        return []


def _load_lcb_records(jsonl_path: Path, release: str | None = None,
                      difficulty: str | None = None) -> list[dict]:
    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            if difficulty and r.get("difficulty") != difficulty:
                continue
            rows.append(r)
    return rows


_FN_NAME_RE = re.compile(r"def\s+(\w+)\s*\(")


def _grade_stdin(code: str, test: dict, timeout: float) -> common.ExecResult:
    """Run code as script, feed test['input'], compare stdout to test['output']."""
    res = common.run_python(code, stdin=test.get("input", ""), timeout=timeout)
    if res.timeout or not res.passed:
        return res
    if res.stdout.strip() == test.get("output", "").strip():
        return res
    res = common.ExecResult(passed=False, stdout=res.stdout, stderr=res.stderr,
                            error="output_mismatch")
    return res


def _grade_functional(code: str, test: dict, fn_name: str | None,
                      timeout: float) -> common.ExecResult:
    """Wrap candidate + a tiny driver that calls Solution().<fn>(*test['input'])."""
    fn = test.get("fn_name") or fn_name
    if not fn:
        return common.ExecResult(passed=False, error="no_fn_name")
    inputs = test.get("input")
    expected = test.get("output")
    if isinstance(inputs, str):
        try:
            inputs = json.loads(inputs)
        except Exception:
            inputs = [inputs]
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except Exception:
            pass
    driver = (
        f"{code}\n\n"
        "import json, sys\n"
        f"_args = json.loads({json.dumps(json.dumps(inputs))})\n"
        f"_expected = json.loads({json.dumps(json.dumps(expected))})\n"
        "try:\n"
        f"    _result = Solution().{fn}(*_args)\n"
        "except NameError:\n"
        f"    _result = {fn}(*_args)\n"
        "if _result != _expected:\n"
        "    print('MISMATCH', repr(_result), 'vs', repr(_expected), file=sys.stderr)\n"
        "    sys.exit(1)\n"
    )
    return common.run_python(driver, timeout=timeout)


def _grade_problem(code: str, tests: list[dict], fn_name: str | None,
                   timeout: float, max_tests: int) -> tuple[bool, str]:
    """Pass iff *every* test case passes (LCB strict semantics)."""
    if not tests:
        return False, "no_tests"
    for t in tests[:max_tests]:
        ttype = t.get("testtype") or ("stdin" if "input" in t else "functional")
        if ttype == "stdin":
            res = _grade_stdin(code, t, timeout)
        else:
            res = _grade_functional(code, t, fn_name, timeout)
        if not res.passed:
            diag = f"failed test ({ttype}): "
            if res.timeout: diag += "TIMEOUT"
            elif res.error: diag += res.error
            else: diag += "stderr=" + res.stderr[-200:].replace("\n", "\\n")
            return False, diag
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--release", default="release_v1",
                    help="release_v1 -> test.jsonl, release_v2 -> test2.jsonl, etc.")
    ap.add_argument("--lcb_root",
                    default="/share/users/luhailun/hf_cache/hub/"
                            "datasets--livecodebench--code_generation_lite/snapshots")
    ap.add_argument("--difficulty", default=None,
                    choices=[None, "easy", "medium", "hard"])
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate first N tasks (smoke test)")
    ap.add_argument("--grade_timeout", type=float, default=10.0)
    ap.add_argument("--grade_workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--max_tests_per_problem", type=int, default=15,
                    help="cap LCB private-tests per problem (LCB releases "
                         "have up to ~30; capping speeds up grading).")
    ap.add_argument("--no_system_prompt", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # locate snapshot dir (we used snapshot_download earlier)
    snap_root = Path(args.lcb_root)
    snaps = [p for p in snap_root.iterdir() if p.is_dir()] if snap_root.is_dir() else []
    if not snaps:
        raise SystemExit(f"no LCB snapshot under {snap_root}")
    snap = sorted(snaps)[0]
    fname = {
        "release_v1": "test.jsonl",
        "release_v2": "test2.jsonl",
        "release_v3": "test3.jsonl",
        "release_v4": "test4.jsonl",
        "release_v5": "test5.jsonl",
        "release_v6": "test6.jsonl",
    }[args.release]
    jsonl = snap / fname
    if not jsonl.exists():
        raise SystemExit(f"missing {jsonl}")
    rows = _load_lcb_records(jsonl, difficulty=args.difficulty)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"[lcb] {len(rows)} tasks from {fname}"
          + (f" (difficulty={args.difficulty})" if args.difficulty else ""))

    # build prompts
    chats = []
    for r in rows:
        starter = r.get("starter_code") or ""
        if starter:
            starter = f"\nStarter code:\n```python\n{starter}\n```\n"
        chats.append(common.build_chat(
            LCB_INSTRUCTION.format(question=r["question_content"], starter=starter),
            system=None if args.no_system_prompt else common.DEFAULT_SYSTEM,
        ))

    # generate
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

    # grade
    items = []
    for row, comp in zip(rows, completions):
        code = common.extract_code(comp)
        public = json.loads(row.get("public_test_cases") or "[]")
        private = _decode_private(row.get("private_test_cases", ""))
        tests = public + private
        # Prefer first occurrence of fn_name across tests.
        fn_name = None
        for t in tests:
            if t.get("fn_name"):
                fn_name = t["fn_name"]; break
        items.append((row["question_id"], _grade_problem,
                      (code, tests, fn_name, args.grade_timeout, args.max_tests_per_problem)))

    grades = common.parallel_grade(items, max_workers=args.grade_workers)

    records = []
    n_pass = 0
    for row, comp, g in zip(rows, completions, grades):
        if isinstance(g, tuple) and g and g[0] == "error":
            ok, diag = False, f"grader_error: {g[1]}"
        else:
            ok, diag = g
        if ok:
            n_pass += 1
        records.append(common.TaskResult(
            task_id=row["question_id"],
            passed=ok,
            completion=comp,
            extracted=common.extract_code(comp),
            diagnostic=diag,
        ))

    common.write_jsonl(out_dir / f"livecodebench_{args.release}.jsonl", records)
    summary = {
        "benchmark": f"livecodebench/{args.release}",
        "model": args.model,
        "difficulty": args.difficulty,
        "n_tasks": len(rows),
        "n_pass": n_pass,
        "pass@1": n_pass / len(rows) if rows else 0.0,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "grading": {
            "timeout_s": args.grade_timeout,
            "max_tests_per_problem": args.max_tests_per_problem,
        },
    }
    common.write_summary(out_dir / f"livecodebench_{args.release}_summary.json", summary)
    print(f"[lcb] pass@1 = {summary['pass@1']:.4f}  ({n_pass}/{len(rows)})")


if __name__ == "__main__":
    main()
