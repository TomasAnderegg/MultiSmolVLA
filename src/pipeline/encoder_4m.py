import torch
import torch.nn as nn
from fourm.models.fm import FM


class Encoder4M(nn.Module):
    """
    Wrapper autour du vrai encoder 4M-21.
    - RGB, depth, seg : passent directement dans 4M
    - Thermal         : pré-encodé par ImageBind, fourni comme embedding externe
    Input  : dict de tenseurs par modalité
    Output : (B, T, D) séquence de tokens multimodaux
    """
    def __init__(self, checkpoint: str = "apple/4M-21_XL", device: str = "cuda"):
        super().__init__()
        self.device = device
        print(f"[Encoder4M] Loading 4M-21 from {checkpoint} ...")
        self.model = FM.from_pretrained(checkpoint)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("[Encoder4M] Loaded ✅")

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        inputs : {
            "rgb"   : (B, 3, 224, 224),
            "depth" : (B, 1, 224, 224),
            "seg"   : (B, 1, 224, 224),
            "thermal": (B, 1024)  ← embedding ImageBind déjà calculé
        }
        """
        B = inputs["rgb"].shape[0]

        fourm_inputs = {
            "rgb@224":    inputs["rgb"],
            "depth@224":  inputs["depth"].repeat(1, 3, 1, 1),  # 1→3 canaux
            "semseg@224": inputs["seg"].repeat(1, 3, 1, 1),
        }

        with torch.no_grad():
            encoder_out = self.model.encoder(fourm_inputs)

        # encoder_out : dict modality → (B, T_mod, D)
        token_list = list(encoder_out.values())
        tokens = torch.cat(token_list, dim=1)   # (B, T_total, D)

        print(f"[Encoder4M] output shape: {tokens.shape}")
        return tokens