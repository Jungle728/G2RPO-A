# G2RPO-A 复现版

本仓库是对 **G2RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance** 的一个可运行复现版本。

本项目不是官方完整发布版。我们基于公开仓库中不完整的代码和论文内容，补齐了 G2RPO-A 的训练入口、配置、trainer 依赖、mask/loss 逻辑、训练脚本和评测脚本，使其可以在本地完成 GRPO 与 G2RPO-A 的对照实验。

论文：

```text
G2RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance
https://arxiv.org/abs/2508.13023
```

## 本仓库补全了什么

- 通过 `src/open_r1 -> src/G2RPO-A` 建立可运行的 `open_r1` 包路径。
- 在 `GRPOConfig` 和 `grpo.py` 中加入 `use_g2rpoa` 开关。
- 将主训练入口接到本地 `G2RPOATrainer`，而不是只跑 TRL 原生 `GRPOTrainer`。
- 修复 `g2rpoa_trainer.py` 中原本指向缺失模块的 import，改用 TRL/Open-R1 可用工具。
- 修正 guidance token 的 mask：外部注入的 guidance token 只作为上下文，不参与 policy loss、KL 和 clip ratio 统计。
- 将 GRPO loss 聚合方式对齐到 TRL 风格的 token normalization。
- 支持从本地 `Dataset.save_to_disk` 目录加载数学训练集。
- 提供 Qwen3 数学训练用的 GRPO/G2RPO-A 配置。
- 提供 MATH-500、HumanEval、LiveCodeBench 的 vLLM 评测脚本。
- 提供构建 curriculum 子集和分析 completion 日志的工具。

## 仓库结构

```text
src/G2RPO-A/                 Python 源码，实际以 open_r1 导入
src/open_r1 -> G2RPO-A       供脚本和 editable install 使用的软链接
recipes/config/              GRPO 和 G2RPO-A 训练配置
recipes/accelerate_configs/  accelerate/deepspeed 配置
scripts/                     训练、数据处理和诊断脚本
scripts/eval/                MATH-500、HumanEval、LiveCodeBench 评测脚本
docs/README_REPRO.md         复现说明和本地实验记录
```

以下目录和文件不会被提交到 GitHub：

```text
data/          数据集、checkpoint、模型权重
logs/          训练/评测日志、逐步 completion JSONL
eval_results/  评测输出
archives/      本地实验归档
swanlog/       swanlab 本地日志
```

## 环境配置

`setup.py` 中要求：

```text
python >= 3.10.9
```

主要依赖版本：

```text
torch==2.5.1
transformers==4.49.0
accelerate==1.4.0
vllm==0.7.2
liger_kernel==0.5.3
math-verify==0.5.2
```

安装项目：

```bash
pip install -e .
```

建议额外安装：

```bash
pip install peft
pip install 'swanlab[dashboard]'
```

注意：原始 `setup.py` 中 TRL 依赖仍然是注释状态，因为该项目依赖特定 TRL 版本。训练前需要自行安装与当前代码兼容的 TRL。

另外，部分环境下 `transformers==4.49.0` 不能识别 Qwen3 的 `model_type=qwen3`。如果加载 Qwen3 时出现 `ValueError: model type 'qwen3' not recognised`，需要升级 Transformers 到支持 Qwen3 的版本。

## 数据准备

本复现实验使用本地保存的数学 curriculum 子集。

示例：构建 200 条数学训练样本：

```bash
HF_HOME=/path/to/hf_cache \
HF_DATASETS_CACHE=/path/to/hf_cache/datasets \
PYTHONPATH=src \
python scripts/build_math_cl_subset.py \
  --output_dir data/openr1_math_cl200 \
  --total_size 200 \
  --per_tier_size 40
```

保存后的数据通过 `load_from_disk` 读取。建议每行包含以下字段：

- `problem`：题目文本
- `answer`：最终答案，用于 `accuracy_reward`
- `generations`：推理轨迹，用于构造 G2RPO-A guidance 前缀

如果缺少 `generations`，trainer 会退回使用 `solution` 或 `answer`，但真实推理轨迹更适合作为 guidance。

## 训练方法

仓库中的训练脚本默认面向离线集群环境，会设置：

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONPATH=src
SWANLAB_MODE=disabled
```

训练脚本通常使用两张可见 GPU：一张用于训练，一张用于 vLLM 生成。比如 `GPU_PAIR=3,2` 表示可见 `cuda:0` 是物理 GPU 3，用于训练；可见 `cuda:1` 是物理 GPU 2，用于 vLLM。

### Qwen3-1.7B-Base GRPO baseline

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

### fixed-mask 版本 G2RPO-A 三轮训练

该脚本使用修复后的 guidance mask 和 loss 逻辑，是我们最终对比中的主要 G2RPO-A 版本：

```bash
GPU_PAIR=3,2 \
MAIN_PROCESS_PORT=29595 \
scripts/run_g2rpoa_math_cl200_qwen3_1.7b_base_fixedmask_gpu23.sh
```

训练输出位置：

```text
data/<RUN_NAME>/
logs/train_<RUN_NAME>.log
logs/steps_<RUN_NAME>/step_*.jsonl
```

## G2RPO-A 配置字段

`src/G2RPO-A/configs.py` 在 TRL 的 GRPO 配置基础上加入了以下字段：

```text
use_g2rpoa             是否使用本地 G2RPOATrainer，而不是 TRL GRPOTrainer
guided_ratio           每个 prompt 中使用 guidance 的 rollout 比例
guidance_length_init   初始 guidance token 长度
max_guidance_length    adaptive guidance length 的上限
guidance_history_t     用于自适应更新的 reward 历史窗口大小
guidance_wrap_think    是否在 guidance 前补 <think>
use_curriculum         是否保留数据集磁盘顺序，使用 sequential sampler
```

baseline 配置设置 `use_g2rpoa: false`。G2RPO-A 配置设置 `use_g2rpoa: true`，并且当前实现要求 `use_vllm: true`。

## 评测方法

### MATH-500

评测单个 checkpoint：

```bash
GPU=3 GPU_MEM_UTIL=0.85 scripts/eval/run_math_eval.sh data/<RUN_NAME> --limit 10
GPU=3 GPU_MEM_UTIL=0.85 scripts/eval/run_math_eval.sh data/<RUN_NAME>
```

使用 GPU2/GPU3 对 7 个模型做完整对比：

```bash
bash scripts/eval/run_gpu23_math_comparison.sh
```

输出目录：

```text
eval_results/math_compare_<timestamp>/
logs/math_compare_<timestamp>/
```

### HumanEval + LiveCodeBench

评测单个 checkpoint：

```bash
GPU=3 GPU_MEM_UTIL=0.5 scripts/eval/run_eval.sh data/<RUN_NAME> --limit 8
GPU=3 GPU_MEM_UTIL=0.5 scripts/eval/run_eval.sh data/<RUN_NAME>
```

使用 GPU2/GPU3 对 7 个模型做完整对比：

```bash
bash scripts/eval/run_gpu23_full_comparison.sh
```

代码评测使用本地 subprocess 沙箱，不需要 E2B。只有训练时使用 `code_reward` 才需要 E2B。

## Completion 日志分析

训练时每步 completion 会写入 `${STEP_LOG_DIR}` 下的 JSONL 文件。可以使用：

```bash
python scripts/completion_stats.py logs/steps_<RUN_NAME>
python scripts/completion_stats.py logs/steps_<RUN_NAME> --tokenizer /path/to/tokenizer --sample-every 1
```

这可以帮助检查：

- `<think>` / `<answer>` 格式是否崩掉
- 是否缺少最终答案块
- 输出是否过长
- reward shaping 是否出现异常

## 本地复现实验结果

以下结果来自 CL200 小数据复现实验，均为单 seed、单样本 pass@1。它们主要用于记录本仓库当前复现状态，不应被理解为最终方法结论。

### MATH-500

解码设置：`temperature=0.6`，`top_p=0.95`，`top_k=20`，`max_tokens=3584`，`seed=42`。

| 组别 | 模型 | MATH-500 pass@1 |
|---|---|---:|
| Qwen3-1.7B | 原始模型 | 374/500 = 74.80% |
| Qwen3-1.7B | GRPO | 369/500 = 73.80% |
| Qwen3-1.7B | G2RPO-A | 371/500 = 74.20% |
| Qwen3-1.7B-Base | 原始模型 | 268/500 = 53.60% |
| Qwen3-1.7B-Base | GRPO | 303/500 = 60.60% |
| Qwen3-1.7B-Base | G2RPO-A loss fixed | 300/500 = 60.00% |
| Qwen3-1.7B-Base | G2RPO-A loss old | 301/500 = 60.20% |

主要观察：

- 在 Qwen3-1.7B-Base 上，RL 训练能明显提升 MATH-500，大约提升 6 到 7 个百分点。
- 在 CL200 小数据设置下，修复后的 G2RPO-A 没有超过 vanilla GRPO。
- 当前差异较小，而且只跑了单 seed，需要更大数据和多 seed 才能判断方法优劣。
- `G2RPO-A loss old` 使用的是旧的 loss/mask 行为，不应作为可信方法结果；保留它只是为了对比修复前后的现象。

### 代码能力评测

HumanEval 和 LiveCodeBench 是旁路评测，不是本次数学训练的主要指标。

| 组别 | 模型 | HumanEval pass@1 | LiveCodeBench v1 pass@1 |
|---|---|---:|---:|
| Qwen3-1.7B | 原始模型 | 122/164 = 74.39% | 80/400 = 20.00% |
| Qwen3-1.7B | GRPO | 119/164 = 72.56% | 88/400 = 22.00% |
| Qwen3-1.7B | G2RPO-A | 124/164 = 75.61% | 81/400 = 20.25% |
| Qwen3-1.7B-Base | 原始模型 | 90/164 = 54.88% | 7/400 = 1.75% |
| Qwen3-1.7B-Base | GRPO | 90/164 = 54.88% | 14/400 = 3.50% |
| Qwen3-1.7B-Base | G2RPO-A loss fixed | 91/164 = 55.49% | 12/400 = 3.00% |
| Qwen3-1.7B-Base | G2RPO-A loss old | 84/164 = 51.22% | 15/400 = 3.75% |

## 已知注意事项

- 本仓库是基于不完整发布版和论文补全的复现代码，不是官方完整实现。
- 脚本中含有本地默认路径，例如 `/share/models/Qwen3/...` 和 `/share/users/luhailun/hf_cache`，在其他机器上需要修改。
- `setup.py` 仍继承上游 Open-R1 的包名和结构。
- 当前 G2RPO-A trainer 要求 `use_vllm=True`。
- 如果训练 `code_reward`，需要 `E2B_API_KEY` 和 `e2b-code-interpreter`；数学训练和数学评测不需要 E2B。
- 不要提交 checkpoint、训练数据、日志或评测输出。

## 引用

如果本项目对你有帮助，请引用原论文：

```bibtex
@inproceedings{guo2026g2rpoa,
  title={G2RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance},
  author={Guo, Yongxin and Deng, Wenbo and Cheng, Zhenglin and Tang, Xiaoying},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year={2026}
}
```

## 致谢

本仓库基于 Hugging Face `open-r1` 修改，并使用 TRL、Transformers、Accelerate、DeepSpeed、vLLM 和 math-verify 等项目。
