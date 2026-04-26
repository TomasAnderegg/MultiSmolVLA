import torch
import torch.nn as nn
from .imagebind_encoder import ImageBindThermalEncoder
from .smolvla.smolvla_wrapper import SmolVLAWrapper


class VLAPipeline(nn.Module):
    """
    Pipeline complet :
    (RGB + depth + seg + thermal) → ImageBind → 4M-21 → MLP → SmolVLA → actions
    """
    def __init__(
        self,
        smolvla_checkpoint: str = "lerobot/smolvla_libero",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.thermal_encoder = ImageBindThermalEncoder(device=device)
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

        # 2. Pass raw 4M inputs into the SmolVLA wrapper
        smol_batch = dict(batch)
        smol_batch["observation.images.4m"] = inputs

        # 3. SmolVLA génère les actions
        actions = self.smolvla.select_action(smol_batch)
        return actions