# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Shared helpers for offline evaluation of G2RPO-A checkpoints.
#
# Goals:
#  * vLLM-based batched generation (same stack as training so behaviour matches).
#  * No outbound network: relies on `HF_HUB_OFFLINE=1` and pre-cached datasets.
#  * Code extraction matches `open_r1.rewards.extract_code` (last ```python``` block).
#  * Per-task subprocess evaluator with strict timeout, identical to the training
#    `code_reward` sandbox script. We deliberately avoid `e2b` here so eval can
#    run without internet / API quota.
#
# Decoding follows the paper (§5.1, "Evaluation protocol"): T=0.6, top-p=0.95,
# top-k=20, single sample per problem (effectively pass@1).

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence


# ----------------------------------------------------------------------------
# Code extraction
# ----------------------------------------------------------------------------
_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str:
    """Return the last fenced python block; fall back to the whole completion.

    Mirrors `open_r1.rewards.extract_code` but tolerates missing language tag
    so models that emit ``` ... ``` without `python` after RL drift still work.
    """
    matches = _CODE_BLOCK.findall(completion)
    if matches:
        return matches[-1].strip("\n")
    return completion.strip()


# ----------------------------------------------------------------------------
# Sandboxed exec
# ----------------------------------------------------------------------------
@dataclass
class ExecResult:
    passed: bool
    stdout: str = ""
    stderr: str = ""
    timeout: bool = False
    error: str = ""


def run_python(
    source: str,
    stdin: str = "",
    timeout: float = 5.0,
    extra_env: dict | None = None,
) -> ExecResult:
    """Run *source* as a fresh Python subprocess, feeding *stdin* on its stdin.

    Returns ExecResult with stdout/stderr captured. Used by both LCB
    (stdin/stdout style problems) and HumanEval (we wrap the candidate +
    `check(entry_point)` invocation into a single script).
    """
    env = os.environ.copy()
    # Disable user site-packages so the subprocess only sees the project env.
    env["PYTHONNOUSERSITE"] = "1"
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", source],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            passed=False,
            stdout=(e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            stderr=(e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""),
            timeout=True,
            error="timeout",
        )
    except Exception as e:  # noqa: BLE001
        return ExecResult(passed=False, error=f"{type(e).__name__}: {e}")
    return ExecResult(
        passed=proc.returncode == 0,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


# ----------------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------------
DEFAULT_SYSTEM = (
    "You are a helpful AI Assistant that provides well-reasoned and detailed "
    "responses. You first think about the reasoning process as an internal "
    "monologue and then provide the user with the answer. Respond in the "
    "following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
)


def build_chat(
    user: str,
    *,
    system: str | None = DEFAULT_SYSTEM,
) -> list[dict]:
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


# ----------------------------------------------------------------------------
# vLLM thin wrapper
# ----------------------------------------------------------------------------
@dataclass
class GenConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 4096
    n: int = 1
    seed: int | None = 42


def load_vllm(
    model: str,
    *,
    max_model_len: int,
    gpu_memory_utilization: float,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
):
    """Lazy import vLLM so callers can `import common` for utilities only."""
    from vllm import LLM

    return LLM(
        model=model,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=False,
        enforce_eager=False,
    )


def generate_chat(
    llm,
    chats: Sequence[Sequence[dict]],
    cfg: GenConfig,
) -> list[str]:
    """Apply chat template and run a single batched generate, returning strings."""
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in chats
    ]
    sp = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        max_tokens=cfg.max_tokens,
        n=cfg.n,
        seed=cfg.seed,
    )
    outs = llm.generate(prompts, sp, use_tqdm=True)
    # Preserve input order
    by_id = {o.request_id: o for o in outs}
    ordered = [by_id[o.request_id] for o in outs]  # vllm preserves order, but be safe
    return [o.outputs[0].text for o in ordered]


# ----------------------------------------------------------------------------
# Result IO
# ----------------------------------------------------------------------------
@dataclass
class TaskResult:
    task_id: str
    passed: bool
    completion: str
    extracted: str
    diagnostic: str = ""


def write_jsonl(path: Path, records: Iterable[dict | TaskResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            if hasattr(r, "__dict__"):
                r = asdict(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Parallel grader
# ----------------------------------------------------------------------------
def parallel_grade(items: Sequence[tuple[str, callable, tuple]], max_workers: int) -> list[Any]:
    """Run callables in a process pool. `items` is [(task_id, fn, args)]."""
    out: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, *args): tid for tid, fn, args in items}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                out[tid] = fut.result()
            except Exception as e:  # noqa: BLE001
                out[tid] = ("error", repr(e))
    return [out[tid] for tid, _, _ in items]
