#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
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
def create_batch_structure(tokenized_chosen, tokenized_rejected, label_pad_token_id=-100):
    """Create the final batch structure needed by the model"""
    batch = {}
    
    # Create sequence tokens (prompt + answer)
    chosen_sequence_tokens = {
        'input_ids': list(tokenized_chosen['prompt_input_ids']) + list(tokenized_chosen['input_ids']),  # Ensure lists
        'attention_mask': list(tokenized_chosen['prompt_attention_mask']) + list(tokenized_chosen['attention_mask'])
    }
    rejected_sequence_tokens = {
        'input_ids': list(tokenized_rejected['prompt_input_ids']) + list(tokenized_rejected['input_ids']),
        'attention_mask': list(tokenized_rejected['prompt_attention_mask']) + list(tokenized_rejected['attention_mask'])
    }
    
    # Create labels (padding prompt part with label_pad_token_id)
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(tokenized_chosen['prompt_input_ids'])] = [label_pad_token_id] * len(tokenized_chosen['prompt_input_ids'])
    
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(tokenized_rejected['prompt_input_ids'])] = [label_pad_token_id] * len(tokenized_rejected['prompt_input_ids'])
    
    # Add to batch with proper prefixes
    for k, toks in {
        'chosen_': chosen_sequence_tokens,
        'rejected_': rejected_sequence_tokens,
    }.items():
        for type_key, tokens in toks.items():
            batch[f"{k}{type_key}"] = tokens
    
    # Validate no None values
    for k, v in batch.items():
        if None in v:
            raise ValueError(f"Found None values in {k}")
        if not all(isinstance(x, (int, np.int32, np.int64)) for x in v):
            raise ValueError(f"Found non-integer values in {k}")
    
    return batch

def test_full_pipeline():
    print("=== Starting Full Pipeline Test ===")
    
    # Setup
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
    
    # Get base tokenization
    chosen_tokens = build_tokenized_answer(tokenizer, processed_sample['prompt'], processed_sample['chosen'])
    rejected_tokens = build_tokenized_answer(tokenizer, processed_sample['prompt'], processed_sample['rejected'])
    
    # Add BOS to prompt and EOS to answer
    for tokens in [chosen_tokens, rejected_tokens]:
        tokens["prompt_input_ids"] = [151643] + list(tokens["prompt_input_ids"])  # Add BOS
        tokens["prompt_attention_mask"] = [1] + list(tokens["prompt_attention_mask"])
        tokens["input_ids"] = list(tokens["input_ids"]) + [151645, 151643]  # Add EOS
        tokens["attention_mask"] = list(tokens["attention_mask"]) + [1, 1]
    
    # Create batch structure
    batch = create_batch_structure(chosen_tokens, rejected_tokens)
    
    # Print final structure
    print("\nFinal batch structure:")
    for k, v in batch.items():
        print(f"\n{k}:")
        print(f"Length: {len(v)}")
        if 'attention_mask' not in k:  # Only show token values for non-mask fields
            print(f"First five tokens: {v[:5]}")
            print(f"Last five tokens: {v[-5:]}")
            if 'input_ids' in k:
                print(f"Does it start with BOS (151643)? {v[0] == 151643}")
                print(f"Does it end with EOS [151645, 151643]? {v[-2:] == [151645, 151643]}")

def test_template():
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:2]")
    sample = dataset[0]

    # Create messages WITH system message (default behavior)
    chosen_messages = [
        {"role": "user", "content": sample['prompt']},
        {"role": "assistant", "content": sample['chosen'][-1]['content']}
    ]
    
    rejected_messages = [
        {"role": "user", "content": sample['prompt']},
        {"role": "assistant", "content": sample['rejected'][-1]['content']}
    ]

    # Apply chat template with default system message
    chosen_text = tokenizer.apply_chat_template(
        chosen_messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    rejected_text = tokenizer.apply_chat_template(
        rejected_messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    print("\nFormatted chosen text:")
    print(chosen_text)
    print("\nFormatted rejected text:")
    print(rejected_text)
    
    # Tokenize
    chosen_tokens = tokenizer(chosen_text)
    rejected_tokens = tokenizer(rejected_text)

    print("\nChosen tokens:")
    print("First 10:", chosen_tokens.input_ids[:10])
    print("Last 10:", chosen_tokens.input_ids[-10:])
    print("\nRejected tokens:")
    print("First 10:", rejected_tokens.input_ids[:10])
    print("Last 10:", rejected_tokens.input_ids[-10:])

if __name__ == "__main__":
    test_template()