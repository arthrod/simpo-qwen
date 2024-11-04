#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer
import json

def test_tokenization():
    print("=== Starting Tokenization Test ===")
    
    # 1. Load tiny dataset
    print("\nLoading dataset...")
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:2]")
    sample = dataset[0]
    print("\nSample data structure:")
    print("Keys:", sample.keys())
    print("\nChosen messages:")
    for msg in sample['chosen']:
        print(f"Role: {msg['role']}, Content: {msg['content'][:50]}...")
    print("\nRejected messages:")
    for msg in sample['rejected']:
        print(f"Role: {msg['role']}, Content: {msg['content'][:50]}...")

    # 2. Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    tokenizer.model_max_length = 4078

    # 3. Process sample
    print("\nProcessing sample...")
    processed_sample = {
        'prompt': sample['prompt'],
        'chosen': [msg['content'] for msg in sample['chosen'] if msg['role'] == 'assistant'][0],
        'rejected': [msg['content'] for msg in sample['rejected'] if msg['role'] == 'assistant'][0]
    }
    print("\nProcessed sample:")
    print(json.dumps(processed_sample, indent=2))

    # 4. Test tokenization
    print("\nTokenizing...")
    # Basic tokenization first
    prompt_tokens = tokenizer(processed_sample['prompt'], add_special_tokens=False)
    chosen_tokens = tokenizer(processed_sample['chosen'], add_special_tokens=False)
    rejected_tokens = tokenizer(processed_sample['rejected'], add_special_tokens=False)

    print("\nToken lengths:")
    print(f"Prompt: {len(prompt_tokens['input_ids'])}")
    print(f"Chosen: {len(chosen_tokens['input_ids'])}")
    print(f"Rejected: {len(rejected_tokens['input_ids'])}")
    
    print("\nFirst few tokens of each:")
    print(f"Prompt: {prompt_tokens['input_ids'][:10]}...")
    print(f"Chosen: {chosen_tokens['input_ids'][:10]}...")
    print(f"Rejected: {rejected_tokens['input_ids'][:10]}...")

if __name__ == "__main__":
    test_tokenization()
