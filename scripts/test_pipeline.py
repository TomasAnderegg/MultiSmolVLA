import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
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
    batch = {
        "observation.images.top": torch.randn(B, 3, 224, 224, device=device),
        "observation.state": torch.randn(B, 7, device=device),
        "task": ["pick up the red block"],
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