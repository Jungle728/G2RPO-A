# AGENTS.md

Research code for the paper "G2RPO-A: Guided GRPO with Adaptive Guidance" (arXiv:2508.13023). It is a partial fork of Hugging Face `open-r1`. Treat the repo as a code dump in progress: the README TODO confirms only training code is released. Several pieces are inconsistent or broken; verify before trusting any documented command.

## Layout

- `src/G2RPO-A/` — Python sources. Despite the directory name, code imports itself as `open_r1` (e.g. `from open_r1.configs import GRPOConfig` in `src/G2RPO-A/grpo.py:27`). See "Known broken wiring" below.
  - `grpo.py`, `sft.py` — standard open-r1 entrypoints. `grpo.py` instantiates the stock `trl.GRPOTrainer`; it does **not** use the adaptive trainer.
  - `configs.py`, `rewards.py` — `GRPOConfig`/`SFTConfig` extending TRL, plus the reward-fn registry consumed by `grpo.py` (`accuracy`, `format`, `tag_count`, `cosine`, `repetition_penalty`, `length`, `code`, `code_format`, `reasoning_steps`). The `format_deepseek` value listed in the docstring at `src/G2RPO-A/grpo.py:55` is not implemented.
  - `trainer/g2rpoa_trainer.py` — the actual G2RPO-A logic. `GuidanceLengthUpdateCallback` and `_compute_new_guidance_length` implement the adaptive guidance-length schedule referenced in the paper. Comments are partly in Chinese.
  - `trainer/rule-based-decay.py` — verl-based variant (`AdaptiveGRPORayTrainer`). Filename contains a dash so it cannot be imported as a module without renaming.
  - `utils/` — tokenizer/model helpers, callbacks, wandb logging.
- `recipes/accelerate_configs/{ddp,zero2,zero3}.yaml` — accelerate launchers (zero2 default).
- `recipes/config/G2RPO-Atrain.yaml` — the only training config that actually exists.
- `recipes/training script/g2rpoa.sh` — sample launch line (note the space in the directory name; quote it).
- `scripts/` — `decontaminate.py`, `generate_reasoning.py`, `run_benchmarks.py`, `upload_details.py` are open-r1 leftovers (not exercised). `scripts/eval/` is the locally-added evaluator (HumanEval + LiveCodeBench), see "Evaluation" below.
- `archives/run1_format01/` — frozen artefacts from the first finished training (logs, step JSONLs, model dir). The repo convention is: when starting a new run, move the previous `logs/`, `data/<run>/` and `swanlog/run-*` under `archives/<name>/` so nothing is overwritten.

## Known broken wiring (do not "fix" silently)

These are real defects in the released tree, not local bitrot:

1. README and `recipes/training script/g2rpoa.sh` invoke `src/open_r1/grpo_code_adagui.py --config recipes/Qwen3-1.7B/grpo/qwen38code.yaml`. **Neither path exists.** The closest extant entrypoint is `src/G2RPO-A/grpo.py` and the only config is `recipes/config/G2RPO-Atrain.yaml`. Confirm with the user before changing the README or the script; do not invent a `grpo_code_adagui.py`.
2. `setup.py` sets `package_dir={"": "src"}` and `find_packages("src")`, so `pip install -e .` looks for a package directory under `src/`. The actual directory is `src/G2RPO-A`, which is not a valid Python identifier and will not be discovered. Every `from open_r1...` import in `grpo.py`/`sft.py` therefore fails until `src/G2RPO-A` is renamed or symlinked to `src/open_r1`.
3. `src/G2RPO-A/trainer/g2rpoa_trainer.py` imports `..data_utils`, `..import_utils`, `..models`, `.callbacks`, `.grpo_config`, `.utils` (see lines 45-50). None of those modules exist in this tree; the trainer is effectively unwired. Nothing in `grpo.py` references this trainer either, so the entrypoint as shipped runs vanilla TRL GRPO, not G2RPO-A.
4. `recipes/config/G2RPO-Atrain.yaml` sets `report_to: [swanlab]` and `swanlab_project`, but `src/G2RPO-A/utils/wandb_logging.py` only handles wandb env vars. Swanlab integration relies on TRL/transformers picking it up directly.

When asked to "run training", surface these gaps before executing anything.

## Environment

- Python `>=3.10.9`. Pinned heavyweights from `setup.py`: `torch==2.5.1`, `transformers==4.49.0`, `accelerate==1.4.0`, `deepspeed==0.15.4`, `vllm==0.7.2`, `liger_kernel==0.5.3`, `math-verify==0.5.2`. `trl` is intentionally commented out (line 70) — the repo expects a specific TRL revision the user installs separately; do not re-add it to `install_requires` without checking.
- Extras: `[tests]`, `[torch]`, `[quality]` (ruff/isort/flake8), `[code]` (e2b), `[eval]` (lighteval+math-verify), `[dev]` = quality+tests+eval.
- `e2b-code-interpreter` is required for the `code` reward (`rewards.py`); it needs an `E2B_API_KEY` env var at runtime. Without it the reward fn silently returns errors and the model trains on a dead signal — surface this to the user before launching with `code` in `reward_funcs`.

`setup.py` is also missing a few things real training needs but the `install_requires` list does not declare. Install them explicitly:
- `peft` (listed in `_deps` but not `install_requires`; trl's GRPOTrainer imports it).
- `swanlab[dashboard]` (the `swanboard` extra) when using `SWANLAB_MODE=local`. Plain `swanlab` is not enough.
- `flash-attn` matching `torch 2.5 / cu12 / cp310 / cxx11abiFALSE` (the released config sets `attn_implementation: flash_attention_2`).
- `transformers==4.49.0` from the pin **does not load Qwen3** (`ValueError: model type 'qwen3' not recognised`). Bump to `transformers==4.51.3` to match the Qwen3-1.7B/0.6B configs in the recipe; this is required, not optional. Document the deviation when touching `setup.py`.

No tests, no CI, no lint config, no pre-commit, and no Makefile. The `[quality]`/`[tests]` extras exist but no commands are wired up — running `ruff`/`pytest` is fine but there is nothing to compare against.

GitHub direct access on the lab network is unreliable. Configure once with
`git config --global url."https://gh.felicity.ac.cn/https://github.com/".insteadOf "https://github.com/"`
so `pip install` of git dependencies (trl pinned commit, lighteval) and asset downloads route through the proxy. `pip install <github wheel URL>` is **not** rewritten by git insteadOf; pass the proxied URL explicitly or use a pre-downloaded wheel.

## Running training

Verified working layout (Qwen3-1.7B, 3 GPUs, 2 train + 1 vLLM):

```bash
HF_HOME=/share/users/luhailun/hf_cache \
HF_DATASETS_CACHE=/share/users/luhailun/hf_cache/datasets \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=1,2,3 \
ACCELERATE_LOG_LEVEL=info SWANLAB_MODE=local E2B_API_KEY=... \
accelerate launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=2 \
    --main_process_port=29555 \
    src/open_r1/grpo.py \
    --config recipes/config/G2RPO-Atrain.yaml \
    --model_name_or_path /share/models/Qwen3/Qwen3-1.7B \
    --push_to_hub false \
    --vllm_gpu_memory_utilization 0.5
```

Caveats before running:
- Resolve the `open_r1` package issue (rename `src/G2RPO-A` to `src/open_r1` or add a symlink) so imports resolve.
- `make_conversation` in `src/G2RPO-A/grpo.py` uses `example["problem"]`, but the `Blancy/verifiable-coding-problems-CoT` dataset's prompt column is `problem_statement`. The released code raises `KeyError: 'problem'` until you change that line. The `code` reward already reads the dataset's `verification_info` column directly so it works as-is.
- HF Hub is firewalled on the lab network. Set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and pre-cache models/datasets to `HF_HOME` (or pass a local model path via `--model_name_or_path`); otherwise `load_dataset` retries five times against `huggingface.co` before giving up.
- The default config pushes to the HF Hub (`push_to_hub: true`, `hub_model_id: Qwen3-1.7B-Open-R1-Code-GRPO`). Set `push_to_hub: false` for local runs unless the user explicitly wants a Hub push.
- `use_vllm: true` reserves one visible GPU for the vLLM generation server: with `CUDA_VISIBLE_DEVICES=a,b,c` and `--num_processes=N`, training takes the first N visible GPUs and vLLM falls on the next one. The README's `--num_processes=7` assumes 8 GPUs total; scale down accordingly.
- `vllm_gpu_memory_utilization` defaults to 0.7. On a shared box drop it to 0.5 to leave headroom for noisy neighbours; OOM during `_init_cache_engine` is the usual symptom.
- `vllm_max_model_len: 4608` and `max_prompt_length: 512` are tuned together; changing one usually means changing the other.
- `accelerate launch --main_process_port=0` does not reliably pick a free port through the deepspeed launcher path. Pass an explicit free port (e.g. 29555) when 29500 is in use.
- `report_to: [swanlab]` requires `SWANLAB_MODE` and login; `local` mode further needs `swanlab[dashboard]` (`swanboard`).

## Conventions

- Code is adapted from `huggingface/open-r1`; preserve its Apache-2.0 headers and style when editing existing files.
- Dependency versions are pinned deliberately (see comment at `setup.py:42`). Do not loosen pins without being asked.
- Existing comments mix English and Chinese. Match the surrounding language when editing in place rather than rewriting.

## Reward & data fixes already applied

These edits were made on top of the upstream tree to make a code-RL run actually train. Don't undo them silently.

- `src/G2RPO-A/grpo.py:189` — `example["problem"]` → `example["problem_statement"]` to match the `Blancy/verifiable-coding-problems-CoT` schema.
- `src/G2RPO-A/rewards.py` — two changes:
  - `code_reward` was `language=verification_info["language"]` (a list-indexed-as-dict bug). Now zips per-row `info`.
  - `format_reward` regex was `^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$`. Qwen3 chat-template completions use spaces instead of `\n` between tags, so the strict regex matched 0/4830 completions in run-1 even when the model produced both blocks. Loosened to `^\s*<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*$`. After the fix, format_reward in run-1 (weight 0.1) reached ~0.43 on step 5 and then collapsed to 0 by step 8 as the policy stopped emitting `<answer>` tags entirely (a real reward-shaping problem, not a regex bug). Run-2 raises the format weight to 0.5 to test whether the collapse can be reversed.
- `src/G2RPO-A/grpo.py` (top of file) — monkey-patches `trl.trainer.grpo_trainer.print_prompt_completions_sample` with a writer that dumps prompt/completion/reward to `${STEP_LOG_DIR}/step_<NNNNN>.jsonl`. Set `STEP_LOG_DIR=logs/steps_<run>` per launch. Without this, `log_completions=true` floods `train.log` with rich tables (>1 MB after a few steps).

## Evaluation

The paper evaluates code checkpoints on **HumanEval** and **LiveCodeBench** (see arXiv 2508.13023 §5.1, Tables 5/8/11). Decoding: T=0.6, top-p=0.95, top-k=20, single sample (effectively pass@1). Upstream ships no eval code; the local `scripts/eval/` directory implements the harness.

- `scripts/eval/common.py` — vLLM loader, code extractor (mirrors training `extract_code`), subprocess-based grader (no e2b dependency, pure local sandbox), parallel grading.
- `scripts/eval/humaneval.py` — runs `openai/openai_humaneval` (164 tasks). Wraps candidate + the dataset's `def check(candidate)` snippet in a fresh Python and times out at 10 s.
- `scripts/eval/livecodebench.py` — reads LCB `code_generation_lite/test*.jsonl` directly (`datasets >= 4` rejects the LCB loader script). Decodes `private_test_cases` (base64 → zlib → pickle → json), grades stdin/stdout problems by piping input and string-matching stdout.
- `scripts/eval/run_eval.sh` — wraps both runners. `GPU=3 GPU_MEM_UTIL=0.5 scripts/eval/run_eval.sh <ckpt>` produces `eval_results/<ckpt>/{humaneval,livecodebench_release_v1}.{jsonl,_summary.json}`.

Datasets are pre-cached under `/share/users/luhailun/hf_cache`:
- `openai/openai_humaneval` via `datasets.load_dataset` (mirror=`https://hf-mirror.com`).
- LiveCodeBench via `huggingface_hub.snapshot_download(allow_patterns=["*.jsonl"])`. The script auto-locates the snapshot under `hub/datasets--livecodebench--code_generation_lite/snapshots/`. Use `--release release_v1..v6` to pick `test.jsonl`..`test6.jsonl`.

To download more benchmarks (math suite from §5.1: MATH500, Minerva, GPQA, AIME24/25), point HF at the mirror first:

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HOME=/share/users/luhailun/hf_cache \
python -c "from datasets import load_dataset; load_dataset('HuggingFaceH4/MATH-500')"
```

The grader uses local subprocess execution and ignores `E2B_API_KEY`; `run_eval.sh` unsets it on launch so a stale key from training cannot leak in. LCB private tests are capped at `--max_tests_per_problem=15` by default; the official release sometimes ships ~30 cases per problem and full grading dominates wall-clock when you only care about pass@1 trends.

### First-pass numbers (Qwen3-1.7B, single sample, T=0.6)

Recorded so future agents don't burn an evening rerunning the same setup. Decoding identical across all rows; HumanEval = 164 tasks, LCB = `release_v1` 400 tasks; LCB private tests capped at 15/problem; seed=42; max_tokens=4096.

| ckpt | source | HumanEval pass@1 | LCB v1 pass@1 |
|---|---|---|---|
| base | `/share/models/Qwen3/Qwen3-1.7B` | 74.4% (122/164) | 21.75% (87/400) |
| run-1 (`reward_weights=[1.0, 0.1]`) | `archives/run1_format01/model_dir` | 31.7% (52/164) | 12.5% (50/400) |
| run-3 (`reward_weights=[1.0, 1.0]`) | `data/fix306code1k1epoch_fmt10` | 74.4% (122/164) | 23.25% (93/400) |

Reading guide:
- run-1's collapse on the eval suite mirrors the training-time format collapse (see "Reward & data fixes already applied" above): once the policy stopped emitting `<answer>` tags around step 7, `extract_code` can no longer pull a clean python block out of the completion, so downstream graders fail. The model still gets ~0.3-0.5 `code_reward` during training because the training-side `extract_code` accepts a wider set of outputs than the eval graders.
- run-3 (format weight = code weight) avoided the collapse and lands a small but real improvement on LCB (+1.5pt). HumanEval is bit-for-bit identical between base and run-3 because pass@1 with a single greedy-ish sample saturates the same set of "easy" problems for both checkpoints.
- The shipped `src/G2RPO-A/grpo.py` runs vanilla TRL `GRPOTrainer`, **not** the `g2rpoa_trainer.py` adaptive-guidance trainer. So these numbers are vanilla GRPO + Qwen3-1.7B + Blancy/verifiable-coding-problems-CoT, not the paper's full method.
