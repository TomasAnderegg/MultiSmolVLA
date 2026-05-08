import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("third_party/ThermalGen")
from thermalgen_demo import ThermalGenSIT


class ThermalGenWrapper(nn.Module):
    """
    Wrapper ThermalGen : RGB (B,3,224,224) [0,1] → thermal (B,3,224,224) [0,1]
    """
    def __init__(self, checkpoint: str = "xjh19972/ThermalGen-XL-2", device: str = "cuda"):
        super().__init__()
        self.device = device
        print(f"[ThermalGen] Loading {checkpoint} ...")
        self.model = ThermalGenSIT.from_pretrained(checkpoint).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("[ThermalGen] Loaded ✅")

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb : (B, 3, 224, 224) en [0, 1]
        x = rgb.to(self.device)

        # 224 → 256 (résolution attendue par ThermalGen)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        # [0,1] → [-1,1]
        x = x * 2.0 - 1.0

        # dataset_idx unconditional
        dataset_idx = torch.ones(x.shape[0], dtype=torch.long, device=self.device) * 1000

        thermal = self.model(x, dataset_idx)

        # [-1,1] → [0,1]
        thermal = thermal * 0.5 + 0.5
        thermal = torch.clamp(thermal, 0.0, 1.0)

        # 1 canal → 3 canaux (ImageBind attend RGB)
        if thermal.shape[1] == 1:
            thermal = thermal.repeat(1, 3, 1, 1)

        # 256 → 224
        thermal = F.interpolate(thermal, size=(224, 224), mode="bilinear", align_corners=False)

        return thermal  # (B, 3, 224, 224)
