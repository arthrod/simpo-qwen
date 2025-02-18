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
    Apply a chat template to a dialogue example based on the specified task.
    
    This function reformats input examples by applying the appropriate chat template through the tokenizer,
    tailored for different training tasks. Depending on the task, it expects the input dictionary to contain
    specific keys and constructs new text fields accordingly.
    
    Parameters:
        example (dict): A dictionary containing dialogue data. Expected keys vary by task:
            - For "sft" or "generation": must contain "messages".
            - For "rm": must contain "chosen" and "rejected".
            - For "simpo": must contain "chosen" and "rejected". Optionally, if "prompt" is provided and is in OpenAI format,
              it will be used as the prompt; otherwise, the prompt is derived from the initial turns of "chosen".
        tokenizer (object): A tokenizer instance that provides the attributes and methods
            `chat_template` and `apply_chat_template`. It is also expected to have a `bos_token` attribute.
        task (Literal["sft", "generation", "rm", "simpo"]): The task type determining how the template is applied.
            - "sft" or "generation": Formats standard dialogue messages.
            - "rm": Formats separate 'chosen' and 'rejected' dialogues.
            - "simpo": Formats dialogues for preference optimization, ensuring messages follow OpenAI format.
        auto_insert_empty_system_msg (bool, optional): If True (default), automatically inserts an empty system
            message when one is missing in the dialogue messages.
        change_template (optional): If set to "mistral", updates the tokenizer's chat template to use MISTRAL_CHAT_TEMPLATE.
    
    Returns:
        dict: The modified example dictionary with additional keys appended:
            - For "sft" and "generation": adds a "text" field.
            - For "rm": adds "text_chosen" and "text_rejected" fields.
            - For "simpo": adds "text_prompt", "text_chosen", and "text_rejected" fields, with the beginning BOS token removed
              from chosen and rejected texts if present.
    
    Raises:
        ValueError: If the example lacks the required keys for a given task, if messages are not in the expected OpenAI format
            for "simpo", or if the provided task is unsupported.
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
    Entrypoint for training and evaluating a language model using the SimPO framework.
    
    This function orchestrates the complete training workflow:
    1. Parses command-line arguments for model, data, and training configurations using H4ArgumentParser.
    2. Sets up logging for both Python’s logging and Hugging Face Transformers to ensure consistent verbosity.
    3. Checks for a previously saved checkpoint and logs the information to potentially resume training.
    4. Sets the random seed for reproducibility.
    5. Loads the required dataset splits with get_datasets, ensuring only the necessary columns are retained, and logs their sizes.
    6. Loads the tokenizer and applies a chat template via apply_chat_template to format input examples appropriately.
    7. Renames dataset columns to match the expected format for the training library.
    8. Logs random samples from the training set for inspection.
    9. Configures model loading parameters, including handling torch data types, quantization, and potential adapter model setups.
    10. Initializes a SimPOTrainer instance with the prepared model, reference model (if applicable), datasets, tokenizer, and additional training parameters.
    11. Runs the training loop, logging and saving training metrics and state.
    12. Saves the final model and creates a model card with metadata upon training completion.
    13. Optionally evaluates the model and, if enabled, pushes the trained model to a model hub.
    
    Returns:
        None
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
