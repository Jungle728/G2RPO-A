"""Build a curriculum-learning subset of OpenR1-Math-220k for G2RPO-A training.

Reproduces the dataset construction described in arXiv:2508.13023 §5.1:

> "We construct a clean subset of the Open-R1 math-220k corpus. Problems are
>  kept only if their solution trajectories are (i) complete, (ii) factually
>  correct, and (iii) syntactically parsable."

Curriculum order (paper §4.3): ascending difficulty by `source` field:
    cn_contest -> aops_forum -> amc_aime -> olympiads -> olympiads_ref

For each row we pick the *first* generation that is both
`is_reasoning_complete=True` and `correctness_math_verify=True` as the
canonical guidance trace (`generations` column). The `answer` column is
re-checked with `math_verify` (wrapped in `$...$` so the LaTeX extractor has
an anchor).

Result is saved with `Dataset.save_to_disk` so the trainer can load it
offline via `datasets.load_from_disk`.

Implementation note: the on-disk dataset has 93k rows; we never iterate the
HF Dataset by index because that re-materializes a Python dict per row. All
heavy passes go through columnar Python lists pulled once.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

# Paper's tier ordering: easy -> hard.
TIER_ORDER = [
    "cn_contest",
    "aops_forum",
    "amc_aime",
    "olympiads",
    "olympiads_ref",
]


def _first_good_gen_idx(is_complete, is_correct):
    if not is_complete or not is_correct:
        return None
    n = min(len(is_complete), len(is_correct))
    for i in range(n):
        if is_complete[i] and is_correct[i]:
            return i
    return None


def _wrap_for_math_verify(answer: str) -> str:
    a = "" if answer is None else str(answer).strip()
    if not a:
        return a
    if (a.startswith("$") and a.endswith("$")) or "\\boxed" in a:
        return a
    return f"${a}$"


_MCQ_LETTER_RE = __import__("re").compile(r"^[A-Z]$")
_MCQ_LETTER_DOLLAR_RE = __import__("re").compile(r"^\$[A-Z]\$$")
_MCQ_CHOICES_RE = __import__("re").compile(r"\(\s*[A-D]\s*\)")


def _looks_multiple_choice(answer: str, problem: str) -> bool:
    """Detect rows where the gold answer is a multiple-choice letter.

    These rows are unsolvable as RL signal: math_verify cannot map a model's
    numeric/algebraic completion back to a letter like "C" without explicit
    knowledge of the choice list, so accuracy_reward is always 0 even when
    the model's reasoning is right. Drop them.
    """
    a = "" if answer is None else str(answer).strip()
    if not a:
        return False
    if _MCQ_LETTER_DOLLAR_RE.match(a) or _MCQ_LETTER_RE.match(a):
        return True
    # Some rows have answer like "$C$" already covered above; also catch the
    # case where the problem text is a multiple-choice question even if the
    # stored answer is "$\\text{C}$" or similar.
    p = problem or ""
    if len(_MCQ_CHOICES_RE.findall(p)) >= 3:
        # Problem is MCQ-shaped. If the answer is a single token without any
        # arithmetic operator, treat it as an unparseable choice answer.
        stripped = a.strip("$").strip()
        if len(stripped) <= 3 and stripped.isalnum() and not stripped.isdigit():
            return True
    return False


def _wrap_generation_for_format(generation: str) -> str:
    """Reshape an OpenR1-Math `generations` entry into the
    `<think>...</think><answer>...</answer>` structure expected by
    `format_reward` and the system prompt.

    The raw OpenR1 traces look like:
        <think>
        ...reasoning...
        </think>

        ...prose recap... \\boxed{...} ...

    We keep the `<think>...</think>` block intact (modulo whitespace) and
    rewrap everything after the closing `</think>` inside `<answer>...</answer>`.
    The `\\boxed{...}` content stays where it is, so accuracy_reward still
    extracts the gold value and the prefix that gets fed to the model as
    guidance now demonstrates the desired output format.

    If the generation does not contain `</think>`, we fall back to wrapping the
    whole string as the answer body, keeping a synthetic short think prefix.
    """
    if not generation:
        return generation
    text = generation.lstrip()
    close_idx = text.find("</think>")
    if close_idx == -1:
        # No think block at all: wrap entire content as the answer.
        return f"<think>\n{text.strip()}\n</think>\n<answer>\n{text.strip()}\n</answer>"
    head = text[: close_idx + len("</think>")]
    tail = text[close_idx + len("</think>") :].strip()
    if not tail:
        # Nothing after </think>; degenerate case, keep the original.
        return text
    # Normalize: ensure head starts with `<think>` (it does for OpenR1) and
    # leave its inner content untouched.
    if not head.lstrip().startswith("<think>"):
        head = f"<think>\n{head}"
    return f"{head}\n<answer>\n{tail}\n</answer>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="data/openr1_math_cl1k")
    parser.add_argument("--total_size", type=int, default=1000)
    parser.add_argument("--per_tier_size", type=int, default=200)
    parser.add_argument(
        "--per_tier_sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated per-tier sizes in TIER_ORDER (cn_contest,aops_forum,amc_aime,"
            "olympiads,olympiads_ref). Overrides --per_tier_size when set. "
            "Example: --per_tier_sizes 200,250,250,200,100"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_problem_chars",
        type=int,
        default=2000,
        help="Drop very long prompts so they fit under max_prompt_length once tokenized.",
    )
    args = parser.parse_args()

    # Resolve per-tier targets.
    if args.per_tier_sizes:
        try:
            tier_targets_list = [int(x) for x in args.per_tier_sizes.split(",")]
        except ValueError as e:
            raise SystemExit(f"--per_tier_sizes must be comma-separated ints: {e}")
        if len(tier_targets_list) != len(TIER_ORDER):
            raise SystemExit(
                f"--per_tier_sizes must have {len(TIER_ORDER)} entries (one per tier "
                f"in {TIER_ORDER}); got {len(tier_targets_list)}."
            )
        tier_targets = dict(zip(TIER_ORDER, tier_targets_list))
    else:
        tier_targets = {t: args.per_tier_size for t in TIER_ORDER}
    print("[build] per-tier targets:")
    for t in TIER_ORDER:
        print(f"  {t}: {tier_targets[t]}")

    from datasets import Dataset, load_dataset, load_from_disk
    from math_verify import LatexExtractionConfig, parse

    print(f"[build] loading {args.dataset} ({args.config}/{args.split})")
    ds = load_dataset(args.dataset, args.config)[args.split]
    n_total = len(ds)
    print(f"[build] total rows: {n_total}")

    # Pull every column we need ONCE as Python lists.
    print("[build] materializing columns...")
    sources = ds["source"]
    answers = ds["answer"]
    isc = ds["is_reasoning_complete"]
    isv = ds["correctness_math_verify"]
    problems = ds["problem"]
    print("[build] columns ready")

    # Cheap-filter pass: collect (row_idx, gen_idx) per tier.
    candidates_by_tier: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i in range(n_total):
        src = sources[i]
        if src not in TIER_ORDER:
            continue
        gen_idx = _first_good_gen_idx(isc[i], isv[i])
        if gen_idx is None:
            continue
        ans = answers[i]
        if ans is None or not str(ans).strip():
            continue
        candidates_by_tier[src].append((i, gen_idx))
    print("[build] candidates after cheap filters:")
    for t in TIER_ORDER:
        print(f"  {t}: {len(candidates_by_tier[t])}")

    # Parse-check pass: random tier-internal order, stop at per_tier_size.
    rng = random.Random(args.seed)
    keep_by_tier: dict[str, list[tuple[int, int]]] = {}
    for tier in TIER_ORDER:
        target = tier_targets[tier]
        cands = list(candidates_by_tier[tier])
        rng.shuffle(cands)
        kept: list[tuple[int, int]] = []
        rejected_parse = 0
        rejected_long = 0
        rejected_mcq = 0
        for row_idx, gen_idx in cands:
            if len(kept) >= target:
                break
            ans_raw = answers[row_idx]
            if _looks_multiple_choice(ans_raw, problems[row_idx]):
                rejected_mcq += 1
                continue
            wrapped = _wrap_for_math_verify(ans_raw)
            if not wrapped:
                continue
            if not parse(
                wrapped,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            ):
                rejected_parse += 1
                continue
            if (
                args.max_problem_chars
                and len(problems[row_idx] or "") > args.max_problem_chars
            ):
                rejected_long += 1
                continue
            kept.append((row_idx, gen_idx))
        keep_by_tier[tier] = kept
        print(
            f"[build] tier={tier}: kept={len(kept)} (target={target}), "
            f"rejected_mcq={rejected_mcq}, rejected_parse={rejected_parse}, "
            f"rejected_long={rejected_long}"
        )
        if len(kept) < target:
            print(
                f"[build] WARN: tier {tier} only yielded {len(kept)} rows < target {target}"
            )
        if len(kept) < target:
            print(
                f"[build] WARN: tier {tier} only yielded {len(kept)} rows < target {target}"
            )

    # Concatenate tiers in CL order: easy first.
    ordered: list[tuple[int, int]] = []
    for tier in TIER_ORDER:
        ordered.extend(keep_by_tier[tier])
    if args.total_size and len(ordered) > args.total_size:
        ordered = ordered[: args.total_size]
    print(f"[build] final selected: {len(ordered)}")
    final_tier_counts = Counter(sources[row_idx] for row_idx, _ in ordered)
    for t in TIER_ORDER:
        print(f"  {t}: {final_tier_counts[t]}")

    # Pull `generations` and `uuid` columns lazily for only the kept indices.
    # `ds["generations"]` is a 93k-list-of-lists -- avoid full materialization.
    keep_indices = [row_idx for row_idx, _ in ordered]
    keep_gen_idx = [gen_idx for _, gen_idx in ordered]
    sub = ds.select(keep_indices)  # cheap view
    sub_gens = sub["generations"]
    sub_uuids = sub["uuid"] if "uuid" in sub.column_names else [None] * len(sub)

    out_rows = []
    for k in range(len(sub)):
        gens = sub_gens[k]
        gi = keep_gen_idx[k]
        if isinstance(gens, list) and gi < len(gens):
            chosen = gens[gi]
        elif isinstance(gens, str):
            chosen = gens
        else:
            chosen = ""
        # Wrap so the guidance prefix matches the format reward structure.
        chosen_wrapped = _wrap_generation_for_format(chosen)
        row_idx = keep_indices[k]
        out_rows.append(
            {
                "problem": problems[row_idx],
                "answer": _wrap_for_math_verify(answers[row_idx]),
                "answer_raw": "" if answers[row_idx] is None else str(answers[row_idx]),
                "generations": chosen_wrapped,
                "generations_raw": chosen,
                "source": sources[row_idx],
                "tier_index": TIER_ORDER.index(sources[row_idx]),
                "uuid": sub_uuids[k],
            }
        )

    out_ds = Dataset.from_list(out_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build] saving to {out_dir}")
    out_ds.save_to_disk(str(out_dir))

    rt = load_from_disk(str(out_dir))
    print(f"[build] saved size: {len(rt)}, columns: {rt.column_names}")
    sample_idxs = [0, len(rt) // 4, len(rt) // 2, 3 * len(rt) // 4, len(rt) - 1]
    print("[build] tier_index at quartiles (should be ascending):", [rt[i]["tier_index"] for i in sample_idxs])


if __name__ == "__main__":
    main()
