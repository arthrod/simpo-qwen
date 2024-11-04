#!/usr/bin/env python3
from transformers import AutoTokenizer
from datasets import load_dataset

def test_original_simpo_handling():
    print("\n=== Testing Original SimPO Processing ===")
    
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:1]")
    example = dataset[0]
    
    print("\nOriginal example:")
    print(f"Keys: {example.keys()}")
    print(f"Prompt: {example['prompt']}")
    print(f"Chosen: {example['chosen']}")
    print(f"Rejected: {example['rejected']}")

    # Mirror original script logic exactly
    if "prompt" in example and isinstance(example["prompt"], list):  # is_openai_format
        prompt_messages = example["prompt"]
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]
    else:
        prompt_messages = example["chosen"][:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]

    print("\nAfter splitting messages:")
    print(f"Prompt messages: {prompt_messages}")
    print(f"Chosen messages: {chosen_messages}")
    print(f"Rejected messages: {rejected_messages}")

    # Apply template exactly as original
    text_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    text_chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    text_rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    print("\nAfter template application:")
    print("\nPrompt text:")
    print(text_prompt)
    print("\nChosen text:")
    print(text_chosen)
    print("\nRejected text:")
    print(text_rejected)

if __name__ == "__main__":
    test_original_simpo_handling()
