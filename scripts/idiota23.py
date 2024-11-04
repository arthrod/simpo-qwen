#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer
from simpo_trainer import SimPOTrainer
from simpo_config import SimPOConfig
import json

def test_data_pipeline():
    print("=== Starting Data Pipeline Test ===")
    
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

    # 2. Initialize tokenizer and create minimal trainer
    print("\nSetting up trainer with tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    
    args = SimPOConfig(
        output_dir="./test_output",
        max_length=4078,
        max_prompt_length=2048,
        remove_unused_columns=False,
        label_pad_token_id=-100,
        is_encoder_decoder=False  # Add this required parameter
    )

    # Create trainer with just tokenizer
    trainer = SimPOTrainer(
        args=args,
        tokenizer=tokenizer,
        train_dataset=None,
        model=None  # Explicitly set to None
    )

    # 5. Test tokenization using trainer's methods
    print("\nTesting tokenize_row...")
    try:
        # Get the assistant responses only
        sample_processed = {
            'prompt': sample['prompt'],
            'chosen': [msg['content'] for msg in sample['chosen'] if msg['role'] == 'assistant'][0],
            'rejected': [msg['content'] for msg in sample['rejected'] if msg['role'] == 'assistant'][0]
        }
        print("\nProcessed sample:")
        print(json.dumps(sample_processed, indent=2))
        
        tokenized = trainer.tokenize_row(sample_processed)
        print("\nTokenized output keys:", tokenized.keys())
        for k, v in tokenized.items():
            if isinstance(v, (list, torch.Tensor)):
                print(f"{k} length:", len(v))
    except Exception as e:
        print("Error in tokenization:", str(e))
        raise

if __name__ == "__main__":
    test_data_pipeline()
