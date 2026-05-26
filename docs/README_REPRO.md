# G2RPO-A Reproduction Notes

This repository contains local reproduction work for **G2RPO-A: Guided GRPO with Adaptive Guidance** on top of the released code dump.

## What Was Fixed

The original tree had the adaptive trainer present but not wired into `grpo.py`. The local changes add a `use_g2rpoa` switch and route training to `open_r1.trainer.g2rpoa_trainer.G2RPOATrainer` when enabled.

Important trainer fixes:

- G2RPO-A-specific config fields were added to `GRPOConfig`.
- `grpo.py` now supports local `Dataset.save_to_disk` directories.
- Math datasets using `problem`/`answer` are normalized for `accuracy_reward`.
- G2RPO-A guidance uses `generations`, falling back to `solution`/`answer` when needed.
- Guidance tokens are kept in attention context but masked out of policy loss, KL, and clip statistics.
- GRPO loss aggregation was aligned to TRL-style token normalization.
- The old `self.num_iterations` bug was fixed to `self.args.num_iterations`.
- vLLM generation in G2RPO-A is batched by remaining token budget for throughput.
- Optional curriculum sampling preserves on-disk easy-to-hard order.

## Local Package Layout

Code imports itself as `open_r1`, so this repo uses:

```text
src/open_r1 -> src/G2RPO-A
```

Launch scripts generally set `PYTHONPATH=src`.

## Main Scripts

Training:

```text
scripts/run_grpo_math_cl200_baseline_qwen3_1.7b_base_gpu23.sh
scripts/run_g2rpoa_math_cl200_qwen3_1.7b_base_fixedmask_gpu23.sh
```

Evaluation:

```text
scripts/eval/run_math_eval.sh
scripts/eval/run_eval.sh
scripts/eval/run_gpu23_math_comparison.sh
scripts/eval/run_gpu23_full_comparison.sh
```

Dataset tooling:

```text
scripts/build_math_cl_subset.py
scripts/completion_stats.py
```

## Experimental Result Summary

The local CL200 experiment did not show G2RPO-A outperforming vanilla GRPO on MATH-500.

MATH-500 pass@1, single sample, `temperature=0.6`, `top_p=0.95`, `top_k=20`, seed 42:

| Group | Model | MATH-500 pass@1 |
|---|---|---:|
| Qwen3-1.7B | original | 374/500 = 74.80% |
| Qwen3-1.7B | GRPO | 369/500 = 73.80% |
| Qwen3-1.7B | G2RPO-A | 371/500 = 74.20% |
| Qwen3-1.7B-Base | original | 268/500 = 53.60% |
| Qwen3-1.7B-Base | GRPO | 303/500 = 60.60% |
| Qwen3-1.7B-Base | G2RPO-A loss fixed | 300/500 = 60.00% |
| Qwen3-1.7B-Base | G2RPO-A loss old | 301/500 = 60.20% |

Interpretation:

- Base-model RL training improved MATH-500 by about 6-7 points.
- Vanilla GRPO was slightly better than the fixed G2RPO-A run in this small CL200 setup.
- Differences are small and single-seed; larger data and multiple seeds are needed before drawing method-level conclusions.
- The old loss implementation should not be treated as a valid result because its loss/masking accounting was incorrect.

## What Is Not Tracked

The following are intentionally ignored and should not be pushed:

- `data/` checkpoints and saved datasets
- `logs/` training/eval logs
- `eval_results/`
- `archives/`
- local wheels, cache files, and `__pycache__`

Keep this repository focused on reproducible code, configs, scripts, and documentation.
