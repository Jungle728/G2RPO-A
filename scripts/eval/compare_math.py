"""Side-by-side comparison of MATH-500 eval results from multiple checkpoints.

Reads `math500.jsonl` (per-task records) and `math500_summary.json` (aggregate)
from each given run directory and prints a comparison table:
  - overall pass@1
  - pass@1 by level (1..5)
  - pass@1 by subject
  - per-task win/loss/tie matrix between the two main runs

Usage:
    python scripts/eval/compare_math.py \
        --runs base=eval_results/Qwen3-1.7B \
               g2rpoa=eval_results/g2rpoa_math_cl200_20260525_233548 \
               grpo=eval_results/grpo_math_cl200_baseline_20260526_075134

The first run becomes the reference for win/loss/tie counting.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_run(label: str, path: Path) -> dict:
    summary_path = path / "math500_summary.json"
    jsonl_path = path / "math500.jsonl"
    if not summary_path.exists():
        raise SystemExit(f"[{label}] missing {summary_path}")
    if not jsonl_path.exists():
        raise SystemExit(f"[{label}] missing {jsonl_path}")
    with summary_path.open() as f:
        summary = json.load(f)
    records = []
    with jsonl_path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return {"label": label, "path": str(path), "summary": summary, "records": records}


def _format_pct(p: float, n: int = None) -> str:
    if n is None:
        return f"{p*100:>5.1f}%"
    return f"{p*100:>5.1f}% ({n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more `label=path` pairs. First label is the reference.",
    )
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"--runs entries must be label=path; got {spec!r}")
        label, path = spec.split("=", 1)
        runs.append(_load_run(label, Path(path)))

    # Sanity: same task set / order
    base = runs[0]
    base_ids = [r["task_id"] for r in base["records"]]
    for r in runs[1:]:
        ids = [x["task_id"] for x in r["records"]]
        if ids != base_ids:
            print(
                f"[warn] task order mismatch between {base['label']} and {r['label']}; "
                "alignment will use task_id matching."
            )

    # ---------- Overall ----------
    print("=" * 70)
    print(" Overall pass@1")
    print("=" * 70)
    print(f"{'run':<28} {'pass@1':>10}  {'(n_passed/n)':>16}")
    for r in runs:
        s = r["summary"]
        passed_n = f"{s['n_passed']}/{s['n_problems']}"
        print(
            f"{r['label']:<28} {_format_pct(s['pass_at_1']):>10}  "
            f"{passed_n:>16}"
        )

    # ---------- By level ----------
    print()
    print("=" * 70)
    print(" pass@1 by level")
    print("=" * 70)
    levels = sorted(
        {int(k) for r in runs for k in r["summary"].get("by_level", {}).keys()}
    )
    header = f"{'run':<28}" + "".join(f"  L{lvl:<5}" for lvl in levels)
    print(header)
    for r in runs:
        bl = r["summary"].get("by_level", {})
        cells = []
        for lvl in levels:
            entry = bl.get(str(lvl))
            if not entry:
                cells.append("   --   ")
                continue
            cells.append(f"{entry['pass_at_1']*100:>5.1f}% ")
        print(f"{r['label']:<28}" + "".join(f"  {c}" for c in cells))

    # ---------- By subject ----------
    print()
    print("=" * 70)
    print(" pass@1 by subject")
    print("=" * 70)
    subjects = sorted({k for r in runs for k in r["summary"].get("by_subject", {}).keys() if k})
    header = f"{'subject':<24}" + "".join(f"{r['label']:>14}" for r in runs)
    print(header)
    for subj in subjects:
        cells = []
        for r in runs:
            entry = r["summary"].get("by_subject", {}).get(subj)
            if not entry:
                cells.append(f"{'--':>14}")
                continue
            cells.append(
                f"{entry['pass_at_1']*100:>6.1f}% ({entry['n_passed']:>3}/{entry['n']:<3})"
            )
        print(f"{subj:<24}" + "".join(cells))

    # ---------- Per-task wins/losses ----------
    if len(runs) >= 2:
        print()
        print("=" * 70)
        print(f" Per-task comparison: {runs[1]['label']} vs {runs[0]['label']}")
        print("=" * 70)
        # Index by task_id
        idx_a = {r["task_id"]: r for r in runs[0]["records"]}
        idx_b = {r["task_id"]: r for r in runs[1]["records"]}
        common = sorted(set(idx_a) & set(idx_b))
        wins = losses = ties_pass = ties_fail = 0
        win_examples = []
        loss_examples = []
        for tid in common:
            pa = bool(idx_a[tid]["passed"])
            pb = bool(idx_b[tid]["passed"])
            if pa and pb:
                ties_pass += 1
            elif (not pa) and (not pb):
                ties_fail += 1
            elif pb and not pa:
                wins += 1
                if len(win_examples) < 3:
                    win_examples.append(idx_a[tid])
            else:
                losses += 1
                if len(loss_examples) < 3:
                    loss_examples.append(idx_a[tid])
        print(f"  both pass : {ties_pass}")
        print(f"  both fail : {ties_fail}")
        print(f"  {runs[1]['label']} wins ({runs[1]['label']} pass, {runs[0]['label']} fail): {wins}")
        print(f"  {runs[0]['label']} wins ({runs[0]['label']} pass, {runs[1]['label']} fail): {losses}")
        net = wins - losses
        print(f"  net delta ({runs[1]['label']} - {runs[0]['label']}): {net:+d}  "
              f"(={net/len(common)*100:+.2f}pt over {len(common)} tasks)")

        if win_examples:
            print()
            print(f" Example tasks where {runs[1]['label']} wins ({runs[0]['label']} fails):")
            for ex in win_examples:
                p = ex["problem"][:120].replace("\n", " ")
                print(f"   - {ex['task_id']} (level {ex['level']}, {ex['subject']}): {p}...")
        if loss_examples:
            print()
            print(f" Example tasks where {runs[0]['label']} wins ({runs[1]['label']} fails):")
            for ex in loss_examples:
                p = ex["problem"][:120].replace("\n", " ")
                print(f"   - {ex['task_id']} (level {ex['level']}, {ex['subject']}): {p}...")


if __name__ == "__main__":
    main()
