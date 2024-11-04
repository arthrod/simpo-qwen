#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.trainer.utils import DPODataCollatorWithPadding
import json
def test_data_pipeline():
    print("=== Starting Data Pipeline Test ===")
    
    # 1. Load tiny dataset
    print("Loading dataset...")
    dataset = load_dataset("arthrod/binarized_trimmed_60percent", split="train[:2]")
    print("\nDataset sample:")
    print(json.dumps(dataset[0], indent=2))

    # 2. Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("rombodawg/Rombos-LLM-V2.6-Qwen-14b")
    tokenizer.model_max_length = 4078
    
    # 3. Create data collator
    print("\nCreating data collator...")
    self._tokenizer = tokenizer
    data_collator = DPODataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        is_encoder_decoder=False
    )

    # 4. Test tokenization
    print("\nTesting tokenization...")
    sample = dataset[0]
    tokenized = tokenize_row(sample, tokenizer)
    print("Tokenized keys:", tokenized.keys())
    for k, v in tokenized.items():
        if isinstance(v, (list, torch.Tensor)):
            print(f"{k} length:", len(v))

    # 5. Test batch creation
    print("\nTesting batch creation...")
    features = [dataset[i] for i in range(2)]
    batch = data_collator(features)
    print("Batch keys:", batch.keys())
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} shape:", v.shape)

    print("\n=== Test Complete ===")

def tokenize_row(feature, tokenizer):
    """Simplified tokenization function"""
    batch = {}
    prompt = feature["prompt"]
    chosen = feature["chosen"][-1]["content"]  # Get the last assistant message
    rejected = feature["rejected"][-1]["content"]  # Get the last assistant message

    # Basic tokenization
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)

    # Create basic batch structure
    batch.update({
        "prompt_input_ids": prompt_tokens["input_ids"],
        "prompt_attention_mask": prompt_tokens["attention_mask"],
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
    })

    return batch

if __name__ == "__main__":
    test_data_pipeline()
