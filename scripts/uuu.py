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

    # Simulate SimPO processing
    prompt_messages = example["prompt"] if "prompt" in example else example["chosen"][:-1]
    chosen_messages = example["chosen"][-1:]
    rejected_messages = example["rejected"][-1:]

    # Add system if needed
    system_message = {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
    if not any(msg.get("role") == "system" for msg in chosen_messages):
        chosen_messages = [system_message] + list(chosen_messages)
    if not any(msg.get("role") == "system" for msg in rejected_messages):
        rejected_messages = [system_message] + list(rejected_messages)

    # Apply templates
    text_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    text_chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    text_rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    print("\n=== What the model sees ===")
    print("\nPrompt:")
    print(text_prompt)
    print("\nChosen response:")
    print(text_chosen)
    print("\nRejected response:")
    print(text_rejected)

if __name__ == "__main__":
    test_simpo_format()
