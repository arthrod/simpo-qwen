#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import json
import sys
import os

# Import from your existing scripts
#from run_simpo import get_args  # assuming this has your argument parsing
from simpo_trainer import SimPOTrainer
from simpo_config import SimPOConfig

def test_pipeline():
    # Load a small subset of your dataset
    print("Loading dataset...")
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:10]")  # Just 10 examples
    print("Dataset sample:", dataset[0])
    
    # Initialize tokenizer
    print("\n=== Initializing tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    tokenizer.model_max_length = 4078
    
    # Setup config similar to your training script
    args = SimPOConfig(
        output_dir="./test_output",
        max_length=4078,
        bf16=True,
        per_device_train_batch_size=1,  # Small for testing
        gradient_accumulation_steps=1,
        max_steps=1
    )

    print("\n=== Creating trainer ===")
    trainer = SimPOTrainer(
        model="rombodawg/Rombos-LLM-V2.6-Qwen-14b",
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset
    )

    # Test tokenization on one example
    print("\n=== Testing tokenization ===")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    try:
        tokenized = trainer.tokenize_row(sample)
        print("Tokenized batch keys:", tokenized.keys())
        print("Tokenized shapes:")
        for k, v in tokenized.items():
            if isinstance(v, (list, torch.Tensor)):
                print(f"{k}: {len(v)}")
    except Exception as e:
        print("Error in tokenization:", str(e))
        raise

    # Test concatenated_inputs
    print("\n=== Testing concatenated_inputs ===")
    try:
        concatenated = trainer.concatenated_inputs(
            tokenized,
            is_encoder_decoder=False,
            label_pad_token_id=-100,
            padding_value=tokenizer.pad_token_id
        )
        print("Concatenated batch keys:", concatenated.keys())
    except Exception as e:
        print("Error in concatenated_inputs:", str(e))
        raise

    # Test batch preparation
    print("\n=== Testing batch preparation ===")
    try:
        # Get a batch using trainer's data collator
        features = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = trainer.data_collator(features)
        print("Collated batch keys:", batch.keys())
    except Exception as e:
        print("Error in batch preparation:", str(e))
        raise

    # Test forward pass
    print("\n=== Testing forward pass ===")
    try:
        batch = trainer._prepare_inputs(batch)
        loss, metrics = trainer.get_batch_loss_metrics(
            trainer.model,
            batch,
            train_eval="train"
        )
        print("Forward pass successful")
        print("Loss:", loss)
        print("Metrics keys:", metrics.keys())
    except Exception as e:
        print("Error in forward pass:", str(e))
        raise

if __name__ == "__main__":
    test_pipeline()
