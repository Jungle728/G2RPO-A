# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This is research code for **G2RPO-A: Guided GRPO with Adaptive Guidance**, adapted from Hugging Face `open-r1`. The README is partly stale: its sample training command points at `src/open_r1/grpo_code_adagui.py` and `recipes/Qwen3-1.7B/grpo/qwen38code.yaml`, which do not exist in the current tree. Use the scripts and configs under `scripts/` and `recipes/config/` instead.

Python code imports as `open_r1`. The package path is wired by the symlink `src/open_r1 -> G2RPO-A`, so scripts are normally launched with `PYTHONPATH=src` and entrypoints such as `src/open_r1/grpo.py`.

## Environment and setup

- Python requirement in `setup.py`: `>=3.10.9`.
- Heavy dependencies are pinned in `setup.py` (`torch`, `accelerate`, `deepspeed`, `vllm`, `transformers`, `liger_kernel`, `math-verify`, etc.). The run scripts currently use `/home/ubuntu/miniconda3/envs/g2rpoa-1` directly.
- Qwen model paths and HF cache paths in the local scripts assume this machine layout:
  - Models: `/share/models/Qwen3/...`
  - HF cache: `/share/users/luhailun/hf_cache`
- Most training/eval scripts run offline with `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`; pre-cache models/datasets before launching new jobs.

Useful setup commands:

```bash
pip install -e .
pip install -e '.[dev]'
```

`setup.py` includes quality/test extras, but there is no project-specific lint config, test directory, Makefile, or CI config in the current tree.

## Common commands

### Lint / quality

```bash
ruff check src scripts
isort --check-only src scripts
flake8 src scripts
```

### Tests

No tests are currently present. If tests are added, use standard pytest commands:

```bash
pytest
pytest path/to/test_file.py -q
pytest path/to/test_file.py::test_name -q
```

### Build the curriculum math subset

`scripts/build_math_cl_subset.py` creates an offline `Dataset.save_to_disk` directory consumed by GRPO training. For the CL200-style dataset used by the checked-in configs:

```bash
HF_HOME=/share/users/luhailun/hf_cache \
HF_DATASETS_CACHE=/share/users/luhailun/hf_cache/datasets \
PYTHONPATH=src \
python scripts/build_math_cl_subset.py \
  --output_dir data/openr1_math_cl200 \
  --total_size 200 \
  --per_tier_size 40
```

The saved dataset must contain the columns the trainer expects, especially `problem`, `answer`, and `generations`.

### Run training

Prefer the wrapper scripts because they set offline HF env vars, `PYTHONPATH=src`, output/log paths, GPU visibility, and overwrite guards.

G2RPO-A math CL200 on Qwen3-1.7B-Base:

```bash
RUN_NAME=my_g2rpoa_run GPU_PAIR=3,2 MAIN_PROCESS_PORT=29595 \
  scripts/run_g2rpoa_math_cl200_qwen3_1.7b_base_gpu23.sh
```

Vanilla GRPO baseline on the same dataset/model:

```bash
RUN_NAME=my_grpo_baseline GPU_PAIR=3,2 MAIN_PROCESS_PORT=29597 \
  scripts/run_grpo_math_cl200_baseline_qwen3_1.7b_base_gpu23.sh
```

Other launch wrappers follow the same pattern (`scripts/run_g2rpoa_math_cl200_qwen3_0.6b_gpu23.sh`, `scripts/run_g2rpoa_math_gpu23.sh`, etc.). `GPU_PAIR` order matters: visible `cuda:0` is used for training and visible `cuda:1` is used by vLLM when `--num_processes=1` and `vllm_device: auto`.

Direct accelerate shape used by the wrappers:

```bash
HF_HOME=/share/users/luhailun/hf_cache \
HF_DATASETS_CACHE=/share/users/luhailun/hf_cache/datasets \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=3,2 PYTHONUNBUFFERED=1 ACCELERATE_LOG_LEVEL=info \
SWANLAB_MODE=disabled STEP_LOG_DIR=logs/steps_my_run PYTHONPATH=src \
/home/ubuntu/miniconda3/envs/g2rpoa-1/bin/accelerate launch \
  --num_processes=1 \
  --main_process_port=29595 \
  src/open_r1/grpo.py \
  --config recipes/config/G2RPO-Atrain_g2rpoa_math_cl200_qwen3_1.7b_base.yaml \
  --output_dir data/my_run
```

Artifacts are normally written as:

- Model/checkpoints: `data/<RUN_NAME>/`
- Training log: `logs/train_<RUN_NAME>.log`
- Per-step completion JSONL: `logs/steps_<RUN_NAME>/step_*.jsonl`

### Completion-log diagnostics

```bash
python scripts/completion_stats.py logs/steps_<RUN_NAME>
python scripts/completion_stats.py logs/steps_<RUN_NAME> --tokenizer /share/models/Qwen3/Qwen3-1.7B --sample-every 1
```

### Evaluation

MATH-500 evaluation:

```bash
GPU=3 GPU_MEM_UTIL=0.85 scripts/eval/run_math_eval.sh data/<RUN_NAME> --limit 10
GPU=3 GPU_MEM_UTIL=0.85 scripts/eval/run_math_eval.sh data/<RUN_NAME>
```

HumanEval + LiveCodeBench evaluation:

```bash
GPU=3 GPU_MEM_UTIL=0.5 scripts/eval/run_eval.sh data/<RUN_NAME> --limit 8
GPU=3 GPU_MEM_UTIL=0.5 scripts/eval/run_eval.sh data/<RUN_NAME>
```

Eval outputs go under `eval_results/`. The code-eval scripts use local subprocess grading and explicitly unset `E2B_API_KEY`; training-time `code_reward` is the part that needs E2B.

## High-level architecture

### Training entrypoint

`src/G2RPO-A/grpo.py` is the main GRPO training entrypoint. It parses `GRPOScriptArguments`, local `GRPOConfig`, and TRL `ModelConfig` with `TrlParser`; loads either a Hugging Face dataset or a local `Dataset.save_to_disk` directory; normalizes rows into chat-style `prompt`; and selects reward functions from the registry in `src/G2RPO-A/rewards.py`.

Important dataset schema behavior in `grpo.py`:

- Prompt text is read from `problem_statement`, `problem`, or `question`.
- If a row has `answer`, it is wrapped for `math_verify` and stored as `solution` for `accuracy_reward`.
- If a row lacks `generations`, the code falls back to `solution` or `answer`; G2RPO-A guidance works best when `generations` is a real reasoning trace.
- The file monkey-patches TRL completion logging so `log_completions=true` writes per-step JSONL files under `${STEP_LOG_DIR}` instead of flooding `train.log`.

Trainer selection happens in `grpo.py`: `use_g2rpoa: true` uses `open_r1.trainer.g2rpoa_trainer.G2RPOATrainer`; otherwise it uses TRL's vanilla `GRPOTrainer`.

### Configs

`src/G2RPO-A/configs.py` extends TRL configs. G2RPO-A-specific fields include:

- `use_g2rpoa`
- `guided_ratio`
- `guidance_length_init`
- `max_guidance_length`
- `guidance_history_t`
- `guidance_wrap_think`
- `use_curriculum`

Configs under `recipes/config/` encode experiment variants. Baseline configs set `use_g2rpoa: false`; G2RPO-A configs set `use_g2rpoa: true` and usually require `use_vllm: true`.

### G2RPO-A trainer

`src/G2RPO-A/trainer/g2rpoa_trainer.py` is a local, modified GRPO trainer. It requires `use_vllm=True`; non-vLLM generation raises `NotImplementedError`.

Core behavior:

- Loads a policy model, reference model, tokenizer, reward functions, and a vLLM engine.
- Uses each row's `generations` text to build a token prefix for guided rollouts.
- Applies guidance to a per-prompt fraction of rollouts according to `guided_ratio`.
- Counts guidance tokens against `max_completion_length`, then batches vLLM generation by remaining-token budget for throughput.
- Tracks `accuracy_reward` or `code_reward` per global step and updates `guidance_length` through `GuidanceLengthUpdateCallback` and `_compute_new_guidance_length`.
- `use_curriculum: true` swaps TRL's random repeat sampler for a sequential repeat sampler so the on-disk easy-to-hard order from `build_math_cl_subset.py` is preserved.

### Rewards

`src/G2RPO-A/rewards.py` contains the reward functions named by `reward_funcs` in YAML configs:

- Math: `accuracy`, `format`, `tag_count`, `reasoning_steps`, `cosine`, `repetition_penalty`, `length`
- Code: `code`, `code_format`

`accuracy_reward` uses `math_verify`. `format_reward` expects `<think>...</think><answer>...</answer>` structure with flexible whitespace. `code_reward` runs generated code through E2B and requires an E2B runtime/API key; do not launch configs containing `code` without checking that E2B is available.

### Evaluation and analysis scripts

`scripts/eval/` is a local vLLM-based evaluation harness:

- `math500.py` grades MATH-500 with `math_verify`.
- `humaneval.py` grades HumanEval by local subprocess execution.
- `livecodebench.py` grades LiveCodeBench JSONL snapshots by local subprocess execution.
- `run_math_eval.sh` and `run_eval.sh` are the preferred wrappers.

`scripts/completion_stats.py` analyzes `${STEP_LOG_DIR}` JSONL logs for tag-format collapse, code-block frequency, and optional token-length stats.

## Operational notes

- The training wrappers refuse to run if another `src/open_r1/grpo.py` process is already active or if output/log paths already exist.
- `use_vllm: true` reserves one visible GPU for generation. With `CUDA_VISIBLE_DEVICES=3,2` and `--num_processes=1`, training uses physical GPU 3 and vLLM uses physical GPU 2.
- `push_to_hub` is false in the current local math configs; keep it false unless the user explicitly asks to publish.
- `report_to: []` and `SWANLAB_MODE=disabled` are used by the current wrappers for local non-reporting runs.
- Preserve the existing Apache-2.0 headers in adapted Open R1 files when editing them.
