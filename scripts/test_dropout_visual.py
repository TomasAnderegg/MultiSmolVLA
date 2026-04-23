import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from src.pipeline.modality_dropout import ModalityDropout


def load_rgb(path, size=(224, 224)):
    img = Image.open(path).convert("RGB").resize(size)
    return TF.to_tensor(img).unsqueeze(0)  # (1, 3, H, W) float [0,1]


def to_numpy(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_dropout_visual.py <pic_og.png>")
        sys.exit(1)

    img_path = sys.argv[1]
    out_path = os.path.splitext(img_path)[0] + "_dropout_grid.png"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb = load_rgb(img_path).to(device)

    epochs = [0, 25, 50, 75, 100]
    corruption_types = ["gaussian", "blur", "occlusion"]

    cols = 1 + len(epochs)  # original + un par epoch
    rows = len(corruption_types)

    # Couleur par type de corruption
    corruption_colors = {"gaussian": "#e07b39", "blur": "#4a90d9", "occlusion": "#6abf69"}

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3 + 0.6))
    fig.suptitle("ModalityDropout — effet par type de corruption et epoch", fontsize=13, y=1.01)

    for row, corruption in enumerate(corruption_types):
        color = corruption_colors[corruption]
        dropout = ModalityDropout(
            modalities=["rgb"],
            p_drop=1.0,
            alpha_min=0.0,
            total_epochs=100,
            corruption_types=[corruption],
        )
        dropout.train()

        # Colonne 0 : image originale
        ax = axes[row][0]
        ax.imshow(to_numpy(rgb))
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        # Label de ligne (type de corruption) en haut à gauche de l'image
        ax.text(0.02, 0.97, corruption.upper(), transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85))
        if row == 0:
            ax.set_title("Original", fontsize=10, pad=6)

        # Colonnes suivantes : une par epoch
        for col, epoch in enumerate(epochs):
            out = dropout({"rgb": rgb.clone()}, epoch=epoch)
            alpha = dropout.get_alpha(epoch)
            img_np = to_numpy(out["rgb"])
            ax = axes[row][col + 1]
            ax.imshow(img_np)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            # Label alpha sur l'image
            ax.text(0.98, 0.02, f"α={alpha:.2f}", transform=ax.transAxes,
                    fontsize=8, color="white", ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55))
            if row == 0:
                ax.set_title(f"Epoch {epoch}", fontsize=10, pad=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
