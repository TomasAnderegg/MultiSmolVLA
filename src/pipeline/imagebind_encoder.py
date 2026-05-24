import torch
import torch.nn as nn
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from torchvision.transforms.functional import normalize


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
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        print("[ImageBind] Loaded ✅")

    def forward(self, thermal_images: torch.Tensor) -> torch.Tensor:
        x = normalize(thermal_images.to(self.device),
                      mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711],
                    )
        inputs = {ModalityType.VISION: x}
        with torch.no_grad():
            embeddings = self.model(inputs)
        out = embeddings[ModalityType.VISION]   # (B, 1024)
        return out