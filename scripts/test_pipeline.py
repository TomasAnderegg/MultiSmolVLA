import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer
from src.pipeline.full_pipeline import VLAPipeline


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}\nDevice: {device}\n{'='*50}\n")

    pipeline = VLAPipeline(device=device)

    B = 1
    inputs = {
        "rgb":     torch.randn(B, 3, 224, 224, device=device),
        "depth":   torch.randn(B, 1, 224, 224, device=device),
        "seg":     torch.randn(B, 1, 224, 224, device=device),
        "thermal": torch.randn(B, 3, 224, 224, device=device),
    }

    task = "pick up the red block\n"
    tokenized = tokenizer(task, padding="max_length", truncation=True, max_length=48, return_tensors="pt")
    batch = {
        "observation.state": torch.randn(B, 7, device=device),
        "observation.language.tokens": tokenized["input_ids"].to(device),
        "observation.language.attention_mask": tokenized["attention_mask"].to(device),
    }

    print("--- Forward pass ---")
    with torch.no_grad():
        actions = pipeline(inputs, batch)

    print(f"\n[FINAL] actions: {actions.shape}")
    print(f"{'='*50}")
    print("✅ Pipeline OK")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()