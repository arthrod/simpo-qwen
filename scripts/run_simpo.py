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
from simpo_config import SimPOConfig
from dataclasses import dataclass, field
from typing import Optional, Literal
from typing import Literal
logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def validate_system_message_count(text: str) -> bool:
    """Validate that there's exactly one system message per dialogue."""
    system_start_count = text.count("<|im_start|>system\n")
    if system_start_count != 1:
        return False
    
    # Check proper sequence
    parts = text.split("<|im_start|>")
    # First part should be empty, then system, then other messages
    if len(parts) < 2:
        return False
    
    # Second part should be system message
    if not parts[1].startswith("system\n"):
        return False
        
    return True

def validate_dialogue(example):
    """Validate both chosen and rejected dialogues."""
    if not validate_system_message_count(example["text_chosen"]):
        raise ValueError(f"Invalid system message count in chosen dialogue")
    if not validate_system_message_count(example["text_rejected"]):
        raise ValueError(f"Invalid system message count in rejected dialogue")
    return example

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "simpo"],
    auto_insert_empty_system_msg: bool = False,
    change_template = None,
):
    """
    Applies the Qwen chat template to format conversations.
    """
    QWEN_CHAT_TEMPLATE = """{%- if tools %}
{{- '<|im_start|>system\\n' }}
{%- if messages[0]['role'] == 'system' %}
    {{- messages[0]['content'] }}
{%- else %}
    {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
{%- endif %}
{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
{%- for tool in tools %}
    {{- "\\n" }}
    {{- tool | tojson }}
{%- endfor %}
{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
{%- endif %}
{%- endif %}
{%- for message in messages %}
{%- if (message['role'] == "user") or (message['role'] == "system" and not loop.first) or (message['role'] == "assistant" and not message.get('tool_calls')) %}
    {{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}
{%- elif message['role'] == "assistant" %}
    {{- '<|im_start|>' + message['role'] }}
    {%- if message.get('content') %}
        {{- '\\n' + message['content'] }}
    {%- endif %}
    {%- for tool_call in message.get('tool_calls', []) %}
        {%- if tool_call.get('function') %}
            {%- set tool_call = tool_call['function'] %}
        {%- endif %}
        {{- '\\n<tool_call>\\n{"name": "' }}
        {{- tool_call['name'] }}
        {{- '", "arguments": ' }}
        {{- tool_call['arguments'] | tojson }}
        {{- '}\\n</tool_call>' }}
    {%- endfor %}
    {{- '<|im_end|>\\n' }}
{%- elif message['role'] == "tool" %}
    {%- if (loop.index0 == 0) or (messages[loop.index0 - 1]['role'] != "tool") %}
        {{- '<|im_start|>user' }}
    {%- endif %}
    {{- '\\n<tool_response>\\n' }}
    {{- message['content'] }}
    {{- '\\n</tool_response>' }}
    {%- if loop.last or (messages[loop.index0 + 1]['role'] != "tool") %}
        {{- '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

    tokenizer.chat_template = QWEN_CHAT_TEMPLATE

    if task in ["sft", "generation"]:
        messages = example["messages"]
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
            tools=None
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]

            example["text_chosen"] = tokenizer.apply_chat_template(
                chosen_messages, 
                tokenize=False, 
                tools=None
            )
            example["text_rejected"] = tokenizer.apply_chat_template(
                rejected_messages, 
                tokenize=False, 
                tools=None
            )
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )

    elif task == "simpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Only add system message if not present in chosen/rejected
            system_message = {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
            if not any(msg.get("role") == "system" for msg in chosen_messages):
                chosen_messages = [system_message] + list(chosen_messages)
            if not any(msg.get("role") == "system" for msg in rejected_messages):
                rejected_messages = [system_message] + list(rejected_messages)

            # Apply template
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

            # Handle BOS token if needed
            if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
                if example["text_chosen"].startswith(tokenizer.bos_token):
                    example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
                if example["text_rejected"].startswith(tokenizer.bos_token):
                    example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]

        return example

def main():
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
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
        # seed=training_args.seed,
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
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
    logger.info("Dataset columns after applying template:")
    for split in raw_datasets:
        logger.info(f"{split}: {raw_datasets[split].column_names}")

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        logger.info(f"{split}: {raw_datasets[split].column_names}")
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )

    model = model_args.model_name_or_path
    # seems to require internet
    # if is_adapter_model(model, model_args.model_revision) is True:
    #     logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
    #     peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    #     model_kwargs = dict(
    #         revision=model_args.base_model_revision,
    #         trust_remote_code=model_args.trust_remote_code,
    #         use_flash_attention_2=model_args.use_flash_attention_2,
    #         torch_dtype=torch_dtype,
    #         use_cache=False if training_args.gradient_checkpointing else True,
    #         device_map=get_kbit_device_map() if quantization_config is not None else None,
    #         quantization_config=quantization_config,
    #     )
    #     base_model = AutoModelForCausalLM.from_pretrained(
    #         peft_config.base_model_name_or_path,
    #         **model_kwargs,
    #     )
    #     model = PeftModel.from_pretrained(
    #         base_model,
    #         model_args.model_name_or_path,
    #         revision=model_args.model_revision,
    #     )
    #     model_kwargs = None

    training_args.model_init_kwargs = model_kwargs
    #########################
    # Instantiate SimPO trainer
    #########################
    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
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
