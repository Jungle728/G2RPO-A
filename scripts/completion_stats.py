#!/usr/bin/env python
"""Offline statistics over per-step completion JSONL logs.

The training entrypoint (`src/open_r1/grpo.py`) and the G2RPO-A trainer dump one
JSONL file per global step under `${STEP_LOG_DIR}` (e.g. `logs/steps_<run>/step_00000.jsonl`).
Each line is `{"step", "prompt", "completion", "reward"}`.

This tool aggregates, over a run (or any set of step files):
  - tag-closure stats: `<think>`/`</think>`/`<answer>`/`</answer>` presence,
  - `format_reward` recomputed offline with the exact training regex,
  - optional token-length stats for prompt/completion (needs a tokenizer).

Examples
--------
Tag + format stats over a whole run (fast, no tokenizer):
    python scripts/completion_stats.py archives/run3_fmt10/steps

Multiple runs side by side:
    python scripts/completion_stats.py \
        archives/run1_format01/steps archives/run3_fmt10/steps

Add token-length percentiles (samples every 10th step by default):
    python scripts/completion_stats.py logs/steps_g2rpoa_run1 \
        --tokenizer /share/models/Qwen3/Qwen3-1.7B

Tokenize every step (slower) and emit JSON:
    python scripts/completion_stats.py logs/steps_g2rpoa_run1 \
        --tokenizer /share/models/Qwen3/Qwen3-1.7B --sample-every 1 --json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

# Must match `format_reward` in src/G2RPO-A/rewards.py.
FORMAT_PATTERN = r"^\s*<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*$"
_FORMAT_RE = re.compile(FORMAT_PATTERN, re.DOTALL | re.MULTILINE)

# A completion is considered "near the generation cap" (likely truncated) when its
# token count is within this margin of the run's max observed length. Only used for
# the heuristic truncation report when a tokenizer is provided.
TRUNCATION_MARGIN = 8


def format_reward(completion: str) -> float:
    """Recompute the training-time format reward for a single completion string."""
    return 1.0 if _FORMAT_RE.match(completion) else 0.0


@dataclass
class TagStats:
    total: int = 0
    no_think_open: int = 0          # missing <think>
    no_think_close: int = 0         # missing </think> (often truncation)
    think_no_answer: int = 0        # has </think> but no <answer> at all
    answer_open_no_close: int = 0   # has <answer> but no </answer>
    full_format: int = 0           # all four tags present
    format_reward_hits: int = 0    # regex-strict format_reward == 1.0
    code_block: int = 0            # contains ```
    python_block: int = 0          # contains ```python

    def add(self, completion: str) -> None:
        c = completion
        self.total += 1
        has_to = "<think>" in c
        has_tc = "</think>" in c
        has_ao = "<answer>" in c
        has_ac = "</answer>" in c
        if not has_to:
            self.no_think_open += 1
        if not has_tc:
            self.no_think_close += 1
        if has_tc and not has_ao:
            self.think_no_answer += 1
        if has_ao and not has_ac:
            self.answer_open_no_close += 1
        if has_to and has_tc and has_ao and has_ac:
            self.full_format += 1
        if format_reward(c) == 1.0:
            self.format_reward_hits += 1
        if "```" in c:
            self.code_block += 1
        if "```python" in c:
            self.python_block += 1

    def as_report(self) -> dict:
        t = max(self.total, 1)
        return {
            "total": self.total,
            "no_think_open": [self.no_think_open, self.no_think_open / t],
            "no_think_close": [self.no_think_close, self.no_think_close / t],
            "think_no_answer": [self.think_no_answer, self.think_no_answer / t],
            "answer_open_no_close": [self.answer_open_no_close, self.answer_open_no_close / t],
            "full_format": [self.full_format, self.full_format / t],
            "format_reward_hits": [self.format_reward_hits, self.format_reward_hits / t],
            "code_block": [self.code_block, self.code_block / t],
            "python_block": [self.python_block, self.python_block / t],
        }


@dataclass
class LenStats:
    sampled_steps: int = 0
    prompt_tokens: list = field(default_factory=list)
    completion_tokens: list = field(default_factory=list)

    def summary(self) -> Optional[dict]:
        if not self.completion_tokens:
            return None

        def describe(vals: list) -> dict:
            s = sorted(vals)
            n = len(s)

            def pctl(q: float) -> int:
                return s[min(n - 1, int(q * n))]

            return {
                "n": n,
                "mean": round(sum(s) / n, 1),
                "p50": pctl(0.50),
                "p90": pctl(0.90),
                "p99": pctl(0.99),
                "max": s[-1],
            }

        comp = describe(self.completion_tokens)
        near_cap = sum(1 for x in self.completion_tokens if x >= comp["max"] - TRUNCATION_MARGIN)
        return {
            "sampled_steps": self.sampled_steps,
            "prompt": describe(self.prompt_tokens),
            "completion": comp,
            "near_cap_count": near_cap,
            "near_cap_frac": round(near_cap / comp["n"], 4),
        }


def iter_step_files(path: str) -> list[str]:
    """Resolve a path argument into a sorted list of step JSONL files."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "step_*.jsonl")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
        return files
    # treat as a glob or a single file
    if any(ch in path for ch in "*?[]"):
        return sorted(glob.glob(path))
    return [path]


def iter_completions(files: list[str]):
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield row


def analyze(
    path: str,
    tokenizer=None,
    sample_every: int = 10,
    verbose: bool = False,
):
    files = iter_step_files(path)
    tags = TagStats()
    lens = LenStats()

    if verbose:
        print(f"[{path}] {len(files)} step files, scanning tags...", file=sys.stderr, flush=True)

    # Tag stats over all completions (cheap, string ops only).
    for row in iter_completions(files):
        tags.add(row.get("completion", ""))

    # Length stats only over sampled steps (tokenization is expensive).
    if tokenizer is not None:
        picked = files[::sample_every] if sample_every > 1 else files
        lens.sampled_steps = len(picked)
        if verbose:
            print(f"[{path}] tokenizing {len(picked)} sampled step files...", file=sys.stderr, flush=True)
        for idx, f in enumerate(picked):
            for line in open(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                comp = row.get("completion", "")
                prompt = row.get("prompt", "")
                lens.completion_tokens.append(len(tokenizer.encode(comp, add_special_tokens=False)))
                lens.prompt_tokens.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
            if verbose and (idx + 1) % 10 == 0:
                print(f"[{path}]   tokenized {idx + 1}/{len(picked)} step files", file=sys.stderr, flush=True)

    return {
        "path": path,
        "num_step_files": len(files),
        "tags": tags.as_report(),
        "lengths": lens.summary(),
    }


def print_report(rep: dict) -> None:
    tags = rep["tags"]
    t = tags["total"]
    print(f"=== {rep['path']}  (step files={rep['num_step_files']}, completions={t}) ===")
    if t == 0:
        print("  (no completions found)\n")
        return

    def line(label: str, key: str, note: str = "") -> None:
        cnt, frac = tags[key]
        suffix = f"   # {note}" if note else ""
        print(f"  {label:<26}: {cnt:7d} ({frac:.1%}){suffix}")

    line("no <think> open", "no_think_open")
    line("no </think> close", "no_think_close", "often truncation")
    line("</think> but no <answer>", "think_no_answer", "skipped answer wrapper")
    line("<answer> but no </answer>", "answer_open_no_close")
    line("full think+answer", "full_format")
    line("format_reward == 1.0", "format_reward_hits", "strict training regex")
    line("contains ```python", "python_block")

    lens = rep["lengths"]
    if lens:
        p = lens["prompt"]
        c = lens["completion"]
        print(f"  -- token lengths (sampled {lens['sampled_steps']} step files) --")
        print(f"  prompt    : mean={p['mean']:8.1f} p50={p['p50']:5d} p90={p['p90']:5d} p99={p['p99']:5d} max={p['max']:5d}")
        print(f"  completion: mean={c['mean']:8.1f} p50={c['p50']:5d} p90={c['p90']:5d} p99={c['p99']:5d} max={c['max']:5d}")
        print(f"  near generation cap: {lens['near_cap_count']}/{c['n']} = {lens['near_cap_frac']:.1%}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="step dirs, single jsonl files, or globs")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer path/name to enable token-length stats (e.g. /share/models/Qwen3/Qwen3-1.7B)")
    ap.add_argument("--sample-every", type=int, default=10,
                    help="when tokenizing, use every Nth step file (default 10; use 1 for all)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a text report")
    ap.add_argument("-v", "--verbose", action="store_true", help="print per-run progress to stderr")
    args = ap.parse_args()

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("transformers not installed; cannot tokenize. Drop --tokenizer.", file=sys.stderr)
            return 2
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    reports = [
        analyze(p, tokenizer=tokenizer, sample_every=args.sample_every, verbose=args.verbose)
        for p in args.paths
    ]

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for rep in reports:
            print_report(rep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
