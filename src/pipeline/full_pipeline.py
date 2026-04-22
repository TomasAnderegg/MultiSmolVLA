import torch
import torch.nn as nn
from .encoder_4m import Encoder4M
from .imagebind_encoder import ImageBindThermalEncoder
from .connector import MLPConnector
from .smolvla_wrapper import SmolVLAWrapper


class VLAPipeline(nn.Module):
    """
    Pipeline complet :
    (RGB + depth + seg + thermal) → ImageBind → 4M-21 → MLP → SmolVLA → actions
    """
    def __init__(
        self,
        fourm_checkpoint: str = "apple/4M-21_XL",
        smolvla_checkpoint: str = "lerobot/smolvla_base",
        encoder_dim: int = 1024,
        smolvla_dim: int = 2048,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.thermal_encoder = ImageBindThermalEncoder(device=device)
        self.encoder         = Encoder4M(checkpoint=fourm_checkpoint, device=device)
        self.connector       = MLPConnector(encoder_dim=encoder_dim, smolvla_dim=smolvla_dim)
        self.smolvla         = SmolVLAWrapper(pretrained=smolvla_checkpoint, device=device)

    def forward(self, inputs: dict, batch: dict) -> torch.Tensor:
        """
        inputs : {
            "rgb"    : (B, 3, 224, 224),
            "depth"  : (B, 1, 224, 224),
            "seg"    : (B, 1, 224, 224),
            "thermal": (B, 3, 224, 224),  ← image thermale RGB synthétique
        }
        batch : dict SmolVLA standard (state, task instructions)
        """
        # 1. Thermal → ImageBind embedding
        thermal_emb = self.thermal_encoder(inputs["thermal"])   # (B, 1024)
        inputs["thermal"] = thermal_emb

        # 2. (RGB + depth + seg) + thermal emb → 4M tokens
        tokens = self.encoder(inputs)                           # (B, T, D)

        # 3. Projette vers SmolLM2 token space
        tokens = self.connector(tokens)                         # (B, T, smolvla_dim)

        # 4. SmolVLA génère les actions
        actions = self.smolvla(batch)                           # (B, action_dim)
        return actions