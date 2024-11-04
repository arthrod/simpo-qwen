
#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import numpy as np

def build_tokenized_answer(tokenizer, prompt, answer):
    """Copy of the trainer's build_tokenized_answer but standalone for testing"""
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

def test_tokenization():
    print("=== Starting Tokenization Test ===")
    
    # 1. Load tiny dataset
    print("\nLoading dataset...")
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:2]")
    sample = dataset[0]

    # 2. Process sample
    processed_sample = {
        'prompt': sample['prompt'],
        'chosen': [msg['content'] for msg in sample['chosen'] if msg['role'] == 'assistant'][0],
        'rejected': [msg['content'] for msg in sample['rejected'] if msg['role'] == 'assistant'][0]
    }
    
    # 3. Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    tokenizer.model_max_length = 4078

    # 4. Test build_tokenized_answer
    print("\nTesting build_tokenized_answer...")
    
    # For chosen response
    print("\nTokenizing chosen response...")
    chosen_tokens = build_tokenized_answer(tokenizer, processed_sample['prompt'], processed_sample['chosen'])
    print("\nChosen tokens structure:")
    for k, v in chosen_tokens.items():
        print(f"{k}: length={len(v)}, first few={v[:5]}...")
    
    # For rejected response
    print("\nTokenizing rejected response...")
    rejected_tokens = build_tokenized_answer(tokenizer, processed_sample['prompt'], processed_sample['rejected'])
    print("\nRejected tokens structure:")
    for k, v in rejected_tokens.items():
        print(f"{k}: length={len(v)}, first few={v[:5]}...")

if __name__ == "__main__":
    test_tokenization()
