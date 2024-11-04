#!/usr/bin/env python3
from transformers import AutoTokenizer
from datasets import load_dataset

def test_simpo_format():
    print("\n=== Testing SimPO Task Format ===")
    
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:1]")
    example = dataset[0]
    
    print("\nOriginal data structure:")
    print("Prompt:", example["prompt"])
    print("\nChosen messages:", [f"{msg['role']}: {msg['content'][:50]}..." for msg in example["chosen"]])
    print("\nRejected messages:", [f"{msg['role']}: {msg['content'][:50]}..." for msg in example["rejected"]])

    # For chosen path - full dialogue
    chosen_messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": example["prompt"]},  # User's question
        example["chosen"][-1]  # Assistant's response
    ]

    # For rejected path - full dialogue
    rejected_messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": example["prompt"]},  # User's question
        example["rejected"][-1]  # Assistant's response
    ]

    # Apply templates
    text_chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    text_rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    print("\n=== What the model sees ===")
    print("\nChosen dialogue:")
    print(text_chosen)
    print("\nRejected dialogue:")
    print(text_rejected)

if __name__ == "__main__":
    test_simpo_format()
