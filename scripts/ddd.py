#!/usr/bin/env python3
from transformers import AutoTokenizer
from datasets import load_dataset

def test_chat_template():
    print("\n=== Testing Chat Template and System Message ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    
    # Get a real sample
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:1]")
    example = dataset[0]
    
    print("\nOriginal example:")
    print("Chosen messages:", [msg["role"] for msg in example["chosen"]])
    print("Rejected messages:", [msg["role"] for msg in example["rejected"]])
    
    # Test cases
    test_cases = [
        # Case 1: Messages without system
        example["chosen"][-1:],  # Just the last message
        # Case 2: Messages with system already
        [{"role": "system", "content": "test"}, *example["chosen"][-1:]],
    ]
    
    for i, messages in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print("Input messages:", [msg["role"] for msg in messages])
        
        # Add system if needed
        if not any(msg.get("role") == "system" for msg in messages):
            system_message = {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
            messages = [system_message] + list(messages)
            print("Added system message")
            
        # Apply template
        result = tokenizer.apply_chat_template(messages, tokenize=False)
        print("\nTemplate output:")
        print(result[:200] + "..." if len(result) > 200 else result)
        print("\nSystem message count:", result.count("<|im_start|>system\n"))
        
        # Check BOS token
        if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
            if result.startswith(tokenizer.bos_token):
                result = result[len(tokenizer.bos_token):]
                print("Removed BOS token")

if __name__ == "__main__":
    test_chat_template()
