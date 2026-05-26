# G2RPO-A Reproduction

This repository contains a practical reproduction of **G2RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance**.

The public release this work started from was incomplete: the adaptive trainer existed in the tree but was not wired into the main training entrypoint, several imports pointed to missing modules, and the documented launch command referenced files that were not present. This fork fills in the missing wiring and adds scripts/configs for reproducing GRPO and G2RPO-A experiments on Qwen3 math tasks.

Paper:

```text
G2RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance
https://arxiv.org/abs/2508.13023
```

## What This Repository Provides

- A runnable `open_r1` package layout via `src/open_r1 -> src/G2RPO-A`.
- A `use_g2rpoa` switch in `GRPOConfig` and `grpo.py`.
- A repaired `G2RPOATrainer` using TRL utilities instead of missing local modules.
- Correct masking so externally injected guidance tokens are context only and are excluded from policy loss, KL, and clip statistics.
- Local/offline math dataset handling with `Dataset.save_to_disk`.
- Qwen3 math training configs for vanilla GRPO and G2RPO-A.
- vLLM-based MATH-500, HumanEval, and LiveCodeBench evaluation scripts.
- Utility scripts for building curriculum subsets and inspecting completion logs.

## Repository Layout

```text
src/G2RPO-A/                 Python source code, imported as open_r1
src/open_r1 -> G2RPO-A       Symlink used by scripts and editable installs
recipes/config/              GRPO and G2RPO-A YAML configs
recipes/accelerate_configs/  accelerate/deepspeed configs from the release
scripts/                     Training and data helper scripts
scripts/eval/                MATH-500, HumanEval, LiveCodeBench evaluators
docs/README_REPRO.md         Additional reproduction notes and local results
```

The following are intentionally ignored and should not be committed:

```text
data/          saved datasets, checkpoints, models
logs/          training/eval logs and per-step generations
eval_results/  benchmark outputs
archives/      local experiment archives
swanlog/       local swanlab output
```

## Environment

Python requirement from `setup.py`:

```text
python >= 3.10.9
```

Pinned heavy dependencies include:

```text
torch==2.5.1
transformers==4.49.0
accelerate==1.4.0
vllm==0.7.2
liger_kernel==0.5.3
math-verify==0.5.2
```

Install the project:

```bash
pip install -e .
```

Recommended additional packages for the reproduced runs:

```bash
pip install peft
pip install 'swanlab[dashboard]'
```

The original `setup.py` keeps the TRL dependency commented out because the project expects a compatible TRL revision to be installed separately. Install a TRL version compatible with the local code before training.

For Qwen3 models, `transformers==4.49.0` may not recognize `model_type=qwen3` in some environments. In our local runs, using a newer compatible Transformers build was required. If loading Qwen3 fails with `ValueError: model type 'qwen3' not recognised`, upgrade Transformers in your environment.

## Data Preparation

This reproduction uses a local curriculum-style math subset saved with Hugging Face Datasets.

Example: build a 200-problem math subset:

```bash
HF_HOME=/path/to/hf_cache \
HF_DATASETS_CACHE=/path/to/hf_cache/datasets \
PYTHONPATH=src \
python scripts/build_math_cl_subset.py \
  --output_dir data/openr1_math_cl200 \
  --total_size 200 \
  --per_tier_size 40
```

The resulting dataset is read with `load_from_disk`. Rows should contain:

- `problem`: prompt text
- `answer`: final answer used by `accuracy_reward`
- `generations`: reasoning trace used as the G2RPO-A guidance source

If `generations` is missing, the trainer falls back to `solution` or `answer`, but real reasoning traces are preferred.

## Training

Most scripts assume an offline cluster setup and set:

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONPATH=src
SWANLAB_MODE=disabled
```

They also use one visible GPU for training and one for vLLM generation. With `GPU_PAIR=3,2`, visible `cuda:0` is physical GPU 3 for training and visible `cuda:1` is physical GPU 2 for vLLM.

### Qwen3-1.7B-Base GRPO Baseline

```bash
RUN_NAME=grpo_math_cl200_baseline \
GPU_PAIR=3,2 \
MAIN_PROCESS_PORT=29595 \
scripts/run_grpo_math_cl200_baseline_qwen3_1.7b_base_gpu23.sh
```

### Qwen3-1.7B-Base G2RPO-A

```bash
RUN_NAME=g2rpoa_math_cl200 \
GPU_PAIR=3,2 \
MAIN_PROCESS_PORT=29597 \
scripts/run_g2rpoa_math_cl200_qwen3_1.7b_base_gpu23.sh
```

### Fixed-Mask G2RPO-A 3-Epoch Run

This config uses the corrected loss/guidance mask behavior used in our final comparison:

```bash
GPU_PAIR=3,2 \
MAIN_PROCESS_PORT=29595 \
scripts/run_g2rpoa_math_cl200_qwen3_1.7b_base_fixedmask_gpu23.sh
```

Outputs are written to:

```text
data/<RUN_NAME>/
logs/train_<RUN_NAME>.log
logs/steps_<RUN_NAME>/step_*.jsonl
```

## G2RPO-A Config Fields

`src/G2RPO-A/configs.py` extends TRL's GRPO config with:

```text
use_g2rpoa             choose local G2RPOATrainer instead of TRL GRPOTrainer
guided_ratio           fraction of rollouts per prompt that receive guidance
guidance_length_init   initial guidance prefix length in tokens
max_guidance_length    cap for adaptive guidance length
guidance_history_t     reward-history window for adaptive update
guidance_wrap_think    prepend <think> before the guidance trace
use_curriculum         preserve saved dataset order with sequential sampler
```

Baseline configs set `use_g2rpoa: false`. G2RPO-A configs set `use_g2rpoa: true` and require `use_vllm: true`.

## Evaluation

### MATH-500

Single checkpoint:

```bash
GPU=3 GPU_MEM_UTIL=0.85 scripts/eval/run_math_eval.sh data/<RUN_NAME> --limit 10
GPU=3 GPU_MEM_UTIL=0.85 scripts/eval/run_math_eval.sh data/<RUN_NAME>
```

Full 7-model comparison on GPU2/GPU3:

```bash
bash scripts/eval/run_gpu23_math_comparison.sh
```

Outputs go under:

```text
eval_results/math_compare_<timestamp>/
logs/math_compare_<timestamp>/
```

### HumanEval + LiveCodeBench

Single checkpoint:

```bash
GPU=3 GPU_MEM_UTIL=0.5 scripts/eval/run_eval.sh data/<RUN_NAME> --limit 8
GPU=3 GPU_MEM_UTIL=0.5 scripts/eval/run_eval.sh data/<RUN_NAME>
```

Full 7-model comparison on GPU2/GPU3:

```bash
bash scripts/eval/run_gpu23_full_comparison.sh
```

Code eval uses local subprocess grading and does not require E2B. Training-time `code_reward` still requires E2B.

## Completion Diagnostics

Per-step completion logs are JSONL files under `${STEP_LOG_DIR}`. Use:

```bash
python scripts/completion_stats.py logs/steps_<RUN_NAME>
python scripts/completion_stats.py logs/steps_<RUN_NAME> --tokenizer /path/to/tokenizer --sample-every 1
```

This is useful for detecting format collapse, missing `<answer>` tags, excessive output length, or reward-shaping failures.

## Local Reproduction Results

These are single-seed pass@1 results from the CL200 reproduction. They are included for transparency, not as definitive method-level claims.

### MATH-500

Decoding: `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=3584`, `seed=42`.

| Group | Model | MATH-500 pass@1 |
|---|---|---:|
| Qwen3-1.7B | original | 374/500 = 74.80% |
| Qwen3-1.7B | GRPO | 369/500 = 73.80% |
| Qwen3-1.7B | G2RPO-A | 371/500 = 74.20% |
| Qwen3-1.7B-Base | original | 268/500 = 53.60% |
| Qwen3-1.7B-Base | GRPO | 303/500 = 60.60% |
| Qwen3-1.7B-Base | G2RPO-A loss fixed | 300/500 = 60.00% |
| Qwen3-1.7B-Base | G2RPO-A loss old | 301/500 = 60.20% |

Main takeaways:

- On Qwen3-1.7B-Base, RL training improved MATH-500 by about 6-7 points.
- In this small CL200 setup, fixed G2RPO-A did not outperform vanilla GRPO.
- Differences are small and single-seed; larger datasets and multiple seeds are needed.
- The old G2RPO-A loss implementation should not be treated as valid because the guidance-token masking/loss accounting was incorrect.

### Code Benchmarks

HumanEval and LiveCodeBench were run as side evaluations, not primary metrics for math training.

| Group | Model | HumanEval pass@1 | LCB v1 pass@1 |
|---|---|---:|---:|
| Qwen3-1.7B | original | 122/164 = 74.39% | 80/400 = 20.00% |
| Qwen3-1.7B | GRPO | 119/164 = 72.56% | 88/400 = 22.00% |
| Qwen3-1.7B | G2RPO-A | 124/164 = 75.61% | 81/400 = 20.25% |
| Qwen3-1.7B-Base | original | 90/164 = 54.88% | 7/400 = 1.75% |
| Qwen3-1.7B-Base | GRPO | 90/164 = 54.88% | 14/400 = 3.50% |
| Qwen3-1.7B-Base | G2RPO-A loss fixed | 91/164 = 55.49% | 12/400 = 3.00% |
| Qwen3-1.7B-Base | G2RPO-A loss old | 84/164 = 51.22% | 15/400 = 3.75% |

## Known Caveats

- This is a reconstruction from an incomplete code release and the paper, not an official complete release.
- The scripts contain local path defaults such as `/share/models/Qwen3/...` and `/share/users/luhailun/hf_cache`; adjust these for your machine.
- `setup.py` still identifies the package as `open-r1` and inherits much of the upstream Open R1 packaging.
- G2RPO-A currently requires `use_vllm=True` in this implementation.
- For code reward training, `E2B_API_KEY` and `e2b-code-interpreter` are required. Math training/evaluation does not use E2B.
- Do not commit generated checkpoints, datasets, logs, or eval outputs.

## Citation

If this project is useful, please cite the original paper:

```bibtex
@inproceedings{guo2026g2rpoa,
  title={G2RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance},
  author={Guo, Yongxin and Deng, Wenbo and Cheng, Zhenglin and Tang, Xiaoying},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year={2026}
}
```

## Acknowledgements

This codebase is adapted from Hugging Face `open-r1` and builds on TRL, Transformers, Accelerate, DeepSpeed, vLLM, and math-verify.
