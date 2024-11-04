#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import numpy as np

def build_tokenized_answer(tokenizer, prompt, answer):
    """Tokenizer helper function"""
    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError(
            "Prompt input ids and answer input ids should have the same length."
        )

    response_token_ids_start_idx = len(prompt_input_ids)
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )

def test_full_tokenization():
    print("=== Starting Full Tokenization Test ===")
    
    # Setup
    print("\nInitializing...")
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    tokenizer.model_max_length = 4078
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:2]")
    sample = dataset[0]

    # Process sample
    processed_sample = {
        'prompt': sample['prompt'],
        'chosen': [msg['content'] for msg in sample['chosen'] if msg['role'] == 'assistant'][0],
        'rejected': [msg['content'] for msg in sample['rejected'] if msg['role'] == 'assistant'][0]
    }

    # Constants
    bos_token_id = tokenizer.bos_token_id  # 151643
    eos_token_id = [151645, 151643]  # explicitly set
    max_length = 4078

    print("\nTokenizing with special tokens...")
    
    # Get base tokenization
    chosen_tokens = build_tokenized_answer(tokenizer, processed_sample['prompt'], processed_sample['chosen'])
    rejected_tokens = build_tokenized_answer(tokenizer, processed_sample['prompt'], processed_sample['rejected'])
    
    # Add BOS token
    print("\nAdding BOS token (should be 151643)...")
    for tokens in [chosen_tokens, rejected_tokens]:
        if bos_token_id != tokens["prompt_input_ids"][0]:
            tokens["prompt_input_ids"] = [bos_token_id] + tokens["prompt_input_ids"]
            tokens["prompt_attention_mask"] = [1] + tokens["prompt_attention_mask"]
    
    # Add EOS tokens
    print("\nAdding EOS tokens (should be [151645, 151643])...")
    for tokens in [chosen_tokens, rejected_tokens]:
        if len(tokens["input_ids"]) < 2 or not (tokens["input_ids"][-2:] == eos_token_id):
            tokens["input_ids"].extend(eos_token_id)
            tokens["attention_mask"].extend([1, 1])

    # Print final structure
    print("\nFinal chosen structure:")
    print(f"Prompt starts with BOS: {chosen_tokens['prompt_input_ids'][0] == bos_token_id}")
    print(f"Answer ends with EOS: {chosen_tokens['input_ids'][-2:] == eos_token_id}")
    print(f"Total length: {len(chosen_tokens['prompt_input_ids']) + len(chosen_tokens['input_ids'])}")
    
    print("\nFinal rejected structure:")
    print(f"Prompt starts with BOS: {rejected_tokens['prompt_input_ids'][0] == bos_token_id}")
    print(f"Answer ends with EOS: {rejected_tokens['input_ids'][-2:] == eos_token_id}")
    print(f"Total length: {len(rejected_tokens['prompt_input_ids']) + len(rejected_tokens['input_ids'])}")

if __name__ == "__main__":
    test_full_tokenization()
