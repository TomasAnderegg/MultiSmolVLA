import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.pipeline.modality_dropout import ModalityDropout

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")

    dropout = ModalityDropout(
        modalities=["rgb", "depth", "seg", "thermal"],
        p_drop=1.0,          # force dropout sur tout pour le test
        alpha_min=0.0,
        total_epochs=100,
    )
    dropout.train()          # important — pas de corruption en eval()

    B = 1
    inputs = {
        "rgb":     torch.ones(B, 3, 224, 224, device=device),
        "depth":   torch.ones(B, 1, 224, 224, device=device),
        "seg":     torch.ones(B, 1, 224, 224, device=device),
        "thermal": torch.ones(B, 3, 224, 224, device=device),
    }

    print("=" * 50)
    for epoch in [0, 25, 50, 75, 100]:
        print(f"\n--- Epoch {epoch} | alpha={dropout.get_alpha(epoch):.2f} ---")
        outputs = dropout(inputs, epoch=epoch)
        for mod, tensor in outputs.items():
            mean = tensor.mean().item()
            print(f"  {mod}: mean={mean:.4f}  (1.0=clean, 0.0=hard dropout)")
    print("\n" + "=" * 50)
    print("✅ ModalityDropout OK")

if __name__ == "__main__":
    main()