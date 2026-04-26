import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer

from src.pipeline.full_pipeline import VLAPipeline

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rgb_for_test.jpeg")


def load_rgb(path: str, device: str) -> torch.Tensor:
    """Load a JPEG and return (1, 3, 224, 224) float tensor in [0, 1]."""
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0).to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}\nDevice: {device}\n{'='*50}\n")

    try:
        pipeline = VLAPipeline(fourm_checkpoint="EPFL-VILAB/4M-21_B", device=device)
    except Exception as error:
        print(f"\nERROR while creating VLAPipeline:\n{error}")
        return

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    B = 1
    if os.path.exists(IMAGE_PATH):
        print(f"Using real image: {IMAGE_PATH}")
        rgb = load_rgb(IMAGE_PATH, device)
    else:
        print("No real image found, using random noise.")
        rgb = torch.randn(B, 3, 224, 224, device=device)

    # Block1 generates thermal from rgb — no need to provide it here
    inputs = {
        "rgb":   rgb,
        "depth": torch.randn(B, 1, 224, 224, device=device),
        "seg":   torch.randn(B, 1, 224, 224, device=device),
    }

    task = "pick up the red block\n"
    tokenized = tokenizer(task, padding="max_length", truncation=True, max_length=48, return_tensors="pt")
    batch = {
        "observation.state": torch.randn(B, 7, device=device),
        "observation.language.tokens": tokenized["input_ids"].to(device),
        "observation.language.attention_mask": tokenized["attention_mask"].to(device),
    }

    print("--- Forward pass ---")
    try:
        with torch.no_grad():
            actions = pipeline(inputs, batch)
    except Exception as error:
        print(f"\nERROR during forward pass:\n{error}")
        return

    print(f"\n[FINAL] actions: {actions.shape}")
    print(f"{'='*50}")
    print("✅ Pipeline OK")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
