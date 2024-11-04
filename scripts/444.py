#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer
from simpo_trainer import SimPOTrainer
from simpo_config import SimPOConfig
import json
import torch

def test_data_pipeline():
    print("=== Starting Data Pipeline Test ===")
    
    # 1. Load tiny dataset
    print("\nLoading dataset...")
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:2]")
    sample = dataset[0]
    print("\nSample data structure:")
    print("Keys:", sample.keys())
    print("Chosen structure:", type(sample['chosen']), len(sample['chosen']))
    print("Rejected structure:", type(sample['rejected']), len(sample['rejected']))

    # 2. Initialize tokenizer and create minimal trainer
    print("\nSetting up trainer with tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    
    args = SimPOConfig(
        output_dir="./test_output",
        max_length=4078,
        max_prompt_length=2048,
        remove_unused_columns=False,
        label_pad_token_id=-100
    )

    # Create trainer with just tokenizer
    trainer = SimPOTrainer(
        args=args,
        tokenizer=tokenizer,  # This will be stored as self._tokenizer
        train_dataset=None
    )

    # 5. Test tokenization using trainer's methods
    print("\nTesting tokenize_row...")
    try:
        # Extract content from conversation format
        sample_processed = {
            'prompt': sample['prompt'],
            'chosen': sample['chosen'][-1]['content'],  # Get assistant's response
            'rejected': sample['rejected'][-1]['content']  # Get assistant's response
        }
        print("\nProcessed sample:", sample_processed)
        
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
