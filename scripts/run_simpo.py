#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

# run_simpo.py
import logging
import random
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import maybe_insert_system_message, is_openai_format
from peft import PeftConfig, PeftModel
from simpo_trainer import SimPOTrainer
from dataclasses import dataclass, field
from typing import Optional, Literal

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


@dataclass
class SimPOConfig(DPOConfig):
    gamma: Optional[float] = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ref_model_init_kwargs: Optional[dict] = field(default=None)
    generate_during_eval: Optional[bool] = field(default=None)
    model_adapter_name: Optional[str] = field(default=None)
    ref_adapter_name: Optional[str] = field(default=None)
    reference_free: Optional[bool] = field(default=False)
    precompute_ref_log_probs: Optional[bool] = field(default=False)

    ref_model = None
    padding_value = None
    max_target_length = None
    dataset_num_proc = None
    callbacks = None
    optimizers = (None, None)
    preprocess_logits_for_metrics = None
    compute_metrics = None
    eval_dataset = None
    model_init = None
    label_pad_token_id = -100
    disable_dropout = True
    truncation_mode = "keep_end"
    label_smoothing = 0
    sync_ref_model = None


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "simpo"],
    auto_insert_empty_system_msg: bool = True,
    change_template=None,
):
    """
    Apply a chat template to an example dictionary based on the specified task.
    
    This function formats dialogue messages contained in the input dictionary according to the
    task type. For tasks "sft" and "generation", it processes the list of messages stored under the
    "messages" key, optionally inserting an empty system message if one is not present. For task "rm",
    it expects both "chosen" and "rejected" keys, applies the chat template to each, and stores the
    results in "text_chosen" and "text_rejected". For task "simpo", the function handles inputs as
    triples of (prompt, chosen, rejected) where the prompt is either provided explicitly or derived by
    taking all but the last turn of the "chosen" list, and then formats and separates the prompt,
    chosen, and rejected responses accordingly (removing any beginning-of-sequence token from the latter
    two if present).
    
    If the optional parameter change_template is set to "mistral", the tokenizer’s chat_template is set
    to a predefined MISTRAL_CHAT_TEMPLATE.
    
    Parameters:
        example (dict): A dictionary containing dialogue data. Depending on the task, required keys are:
            - "sft" or "generation": must include "messages".
            - "rm": must include both "chosen" and "rejected".
            - "simpo": must include either ["chosen", "rejected"] or ["prompt", "chosen", "rejected"].
        tokenizer: An object with chat formatting methods and attributes, including
            apply_chat_template(), chat_template, and bos_token.
        task (Literal["sft", "generation", "rm", "simpo"]): Specifies the formatting task.
        auto_insert_empty_system_msg (bool, optional): If True, inserts an empty system message when none is
            present. Defaults to True.
        change_template (optional): If set to "mistral", updates the tokenizer's chat_template to use the
            MISTRAL_CHAT_TEMPLATE constant.
    
    Returns:
        dict: The updated example dictionary with additional keys:
            - For "sft" and "generation": adds "text".
            - For "rm": adds "text_chosen" and "text_rejected".
            - For "simpo": adds "text_prompt", "text_chosen", and "text_rejected".
    
    Raises:
        ValueError: If required keys for the specified task are missing, if the message format is not valid
            (e.g., not in OpenAI format for "simpo"), or if the task is not supported.
    """
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "simpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(
                example["rejected"]
            ):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False
            )
            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, tokenize=False
            )
            if example["text_chosen"].startswith(tokenizer.bos_token):
                example["text_chosen"] = example["text_chosen"][
                    len(tokenizer.bos_token) :
                ]
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, tokenize=False
            )
            if example["text_rejected"].startswith(tokenizer.bos_token):
                example["text_rejected"] = example["text_rejected"][
                    len(tokenizer.bos_token) :
                ]
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def main():
    """
    Main entry point for training and evaluating a model using the SimPO method.
    
    This function performs the following steps:
    1. Parses command-line arguments for model, data, and training configurations using H4ArgumentParser.
    2. Configures logging, including process-specific log levels and formatter settings.
    3. Checks for an existing checkpoint and sets a random seed for reproducibility.
    4. Loads datasets with specified splits and columns, logs dataset summaries, and applies a chat template to format comparisons.
    5. Renames dataset columns to match expected input names for training.
    6. Initializes the tokenizer and adjusts configurations based on model type (e.g., applying a "mistral" template if applicable).
    7. Sets up model keyword arguments, including device mapping and quantization configurations.
    8. Determines if the model uses an adapter; if so, loads both the base model and the adapter.
    9. Instantiates a SimPOTrainer object with the model, (optional) reference model, training arguments, datasets, tokenizer, and additional configuration parameters.
    10. Executes the training loop, logs training metrics, and saves the training state and model after completion.
    11. Saves the final model, creates a model card, and optionally evaluates the model and pushes it to a hub if specified.
    
    Usage Example:
        if __name__ == "__main__":
            main()
    """
    parser = H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=[
            "messages",
            "chosen",
            "rejected",
            "prompt",
            "completion",
            "label",
            "score_chosen",
            "score_rejected",
        ],
        # seed=training_args.seed,
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = (
        "left"  # Truncate from left to ensure we don't lose labels in final turn
    )
    tokenizer = get_tokenizer(model_args, data_args)

    if "mistral" in model_args.model_name_or_path.lower():
        change_template = "mistral"
    else:
        change_template = None
    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "simpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "change_template": change_template,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {
                "text_prompt": "prompt",
                "text_chosen": "chosen",
                "text_rejected": "rejected",
            }
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(
            f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}"
        )
        logger.info(
            f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}"
        )
        logger.info(
            f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}"
        )

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(
            model_args.model_name_or_path, revision=model_args.model_revision
        )
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=(
                get_kbit_device_map() if quantization_config is not None else None
            ),
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate SimPO trainer
    #########################
    trainer = SimPOTrainer(
        model=model,
        ref_model=ref_model,  # pass in to bypass DPO Trainer check for ref model but is not actually used
        model_init_kwargs=model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
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
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
