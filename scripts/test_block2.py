import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer

from src.pipeline.block2 import Block2

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rgb_for_test.jpeg")


def load_rgb(path: str, device: str) -> torch.Tensor:
    """Load a JPEG and return (1, 3, 224, 224) float tensor in [0, 1]."""
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # -> (3, 224, 224) in [0, 1]
    ])
    return tf(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        pipeline = Block2(fourm_checkpoint="EPFL-VILAB/4M-21_XL", device=device)
    except ModuleNotFoundError as error:
        print("\nERROR: missing package while importing Block2:")
        print(f"  {error}")
        print("\nPlease install the required packages in your active environment.")
        print("For example: pip install torch transformers fourm")
        return
    except Exception as error:
        print("\nERROR while creating Block2:")
        print(error)
        return

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    B = 1
    if os.path.exists(IMAGE_PATH):
        print(f"Using real image: {IMAGE_PATH}")
        rgb = load_rgb(IMAGE_PATH, device)
    else:
        print("No real image found, using random noise.")
        rgb = torch.randn(B, 3, 224, 224, device=device)

    inputs = {
        "rgb": rgb,
        "depth": torch.randn(B, 1, 224, 224, device=device),
        "seg": torch.randn(B, 1, 224, 224, device=device),
        "thermal": torch.randn(B, 1024, device=device),  # pre-encoded by ImageBind
    }

    task = "pick up the red block\n"
    tokenized = tokenizer(task, padding="max_length", truncation=True, max_length=48, return_tensors="pt")
    batch = {
        "observation.state": torch.randn(B, 7, device=device),
        "observation.language.tokens": tokenized["input_ids"].to(device),
        "observation.language.attention_mask": tokenized["attention_mask"].to(device),
    }

    print("\nRunning Block2 forward pass...")
    try:
        with torch.no_grad():
            actions = pipeline(inputs, batch)
    except Exception as error:
        print("\nERROR during Block2 forward pass:")
        print(error)
        return

    print(f"\nSuccess: actions shape = {actions.shape}")
    print("✅ Block2 pipeline test completed")


if __name__ == "__main__":
    main()
