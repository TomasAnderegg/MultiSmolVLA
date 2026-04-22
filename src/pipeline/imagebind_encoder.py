import torch
import torch.nn as nn
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


class ImageBindThermalEncoder(nn.Module):
    """
    Encode les images thermales via ImageBind.
    Input  : (B, 3, 224, 224) image thermale RGB synthétique
    Output : (B, 1024)
    """
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        print("[ImageBind] Loading ...")
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("[ImageBind] Loaded ✅")

    def forward(self, thermal_images: torch.Tensor) -> torch.Tensor:
        inputs = {ModalityType.VISION: thermal_images.to(self.device)}
        with torch.no_grad():
            embeddings = self.model(inputs)
        out = embeddings[ModalityType.VISION]   # (B, 1024)
        print(f"[ImageBind] thermal embedding shape: {out.shape}")
        return out