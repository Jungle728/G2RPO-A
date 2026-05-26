# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


# Replace TRL's rich completion table printer with a per-step JSONL writer so the
# main training log stays small and each step's completions sit in its own file.
import json
import os
from pathlib import Path

import trl.trainer.grpo_trainer as _grpo_mod


def _per_step_completion_writer(prompts, completions, rewards, step):
    out_dir = Path(os.environ.get("STEP_LOG_DIR", "logs/steps"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{int(step):05d}.jsonl"
    with path.open("a") as f:
        for prompt, completion, reward in zip(prompts, completions, rewards):
            f.write(
                json.dumps(
                    {
                        "step": int(step),
                        "prompt": prompt,
                        "completion": completion,
                        "reward": float(reward),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


_grpo_mod.print_prompt_completions_sample = _per_step_completion_writer


logger = logging.getLogger(__name__)



@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset. If `dataset_name` points to an existing local directory
    # produced by `Dataset.save_to_disk`, load it offline; otherwise fall through
    # to HF Hub / cache via `load_dataset`. This lets the curriculum subset
    # (scripts/build_math_cl_subset.py) drop in without touching HF_HOME.
    if os.path.isdir(script_args.dataset_name):
        loaded = load_from_disk(script_args.dataset_name)
        # Normalize to DatasetDict so `dataset[split]` works downstream.
        if not isinstance(loaded, DatasetDict):
            split_name = script_args.dataset_train_split or "train"
            loaded = DatasetDict({split_name: loaded})
        dataset = loaded
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        # Support both code-style (`problem_statement`, e.g. Blancy/verifiable-coding-problems-CoT)
        # and math-style (`problem`, e.g. agentica-org/DeepScaleR-Preview-Dataset, MATH-500) schemas.
        if "problem_statement" in example and example["problem_statement"] is not None:
            user_content = example["problem_statement"]
        elif "problem" in example and example["problem"] is not None:
            user_content = example["problem"]
        elif "question" in example and example["question"] is not None:
            user_content = example["question"]
        else:
            raise KeyError(
                "make_conversation: dataset row has no 'problem_statement', 'problem', or 'question' column."
            )
        prompt.append({"role": "user", "content": user_content})

        out = {"prompt": prompt}

        # accuracy_reward reads the `solution` kwarg. Math datasets (DeepScaleR,
        # MATH-500, NuminaMath) ship the parseable gold in `answer`; their
        # `solution` field, if present, is the long human-written proof which
        # math_verify cannot consistently extract a single answer from. So when
        # the row has an `answer`, treat that as the source of truth and wrap
        # in `$...$` for math_verify's LatexExtractionConfig anchor matching.
        # Code rows (Blancy/verifiable-coding-problems-CoT) have no `answer`
        # column and rely on `verification_info` instead, so this branch does
        # not affect them.
        if "answer" in example and example["answer"] is not None:
            answer = str(example["answer"]).strip()
            if not (answer.startswith("$") and answer.endswith("$")) and "\\boxed" not in answer:
                answer = f"${answer}$"
            out["solution"] = answer

        # G2RPOATrainer._prepare_inputs reads `example["generations"]` to build the
        # guidance prefix (it tokenizes the long-trace and slices to guidance_length).
        # The Blancy code dataset has this column; math datasets do not. Fall back
        # to the human-written `solution` proof, then to `answer`.
        if "generations" not in example or example.get("generations") is None:
            trace = example.get("solution") or example.get("answer")
            if trace is not None:
                out["generations"] = str(trace)
        return out

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    # With a single training process and two visible GPUs, HF Trainer would wrap the
    # policy in DataParallel and occupy the vLLM GPU too. Keep the policy on the
    # process device; vLLM still sees the extra visible GPU and uses it via `auto`.
    if training_args.use_vllm and training_args.n_gpu > 1 and training_args.local_rank in {-1, 0}:
        logger.info(f"Restricting HF Trainer n_gpu from {training_args.n_gpu} to 1 because use_vllm=True")
        training_args._n_gpu = 1

    #############################
    # Initialize the GRPO trainer
    #############################
    if training_args.use_g2rpoa:
        from open_r1.trainer.g2rpoa_trainer import G2RPOATrainer

        trainer_cls = G2RPOATrainer
    else:
        trainer_cls = GRPOTrainer
    logger.info(f"Using trainer class: {trainer_cls.__name__}")
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
