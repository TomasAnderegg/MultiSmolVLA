import torch.nn as nn

from .thermal_wrapper import ThermalGenWrapper
from .imagebind_encoder import ImageBindThermalEncoder
from .modality_dropout import ModalityDropout


class Block1(nn.Module):
    """
    Block 1 — Perception multimodale
    Input  : {"rgb": (B,3,224,224), "depth": (B,1,224,224), "seg": (B,1,224,224)}  en [0,1]
    Output : {"rgb": (B,3,224,224), "depth": (B,1,224,224), "seg": (B,1,224,224), "thermal": (B,1024)}
    """

    def __init__(
        self,
        device: str = "cuda",
        p_drop: float = 0.5,
        total_epochs: int = 100,
    ):
        super().__init__()
        self.thermalgen = ThermalGenWrapper(device=device)
        self.dropout    = ModalityDropout(p_drop=p_drop, total_epochs=total_epochs)
        self.imagebind  = ImageBindThermalEncoder(device=device)

    def forward(self, inputs: dict, epoch: int = 0) -> dict:
        """
        inputs : {"rgb", "depth", "seg"} or {"rgb", "depth", "seg", "thermal"}
        epoch  : epoch courante pour le curriculum dropout (ignoré à l'inférence)
        If "thermal" is already present in inputs (pre-computed), ThermalGen is skipped.
        """
        # 1. Use pre-computed thermal if provided, otherwise generate from RGB
        if "thermal" in inputs:
            thermal = inputs["thermal"]                    # (B, 3, 224, 224)
        else:
            thermal = self.thermalgen(inputs["rgb"])       # (B, 3, 224, 224)

        # 2. Assembler les 4 modalités
        inputs = {**inputs, "thermal": thermal}

        # 3. Modality dropout (no-op à l'inférence)
        inputs = self.dropout(inputs, epoch)

        # 4. Thermal → ImageBind embedding
        thermal_emb = self.imagebind(inputs["thermal"])    # (B, 1024)

        return {
            "rgb":     inputs["rgb"],
            "depth":   inputs["depth"],
            "seg":     inputs["seg"],
            "thermal": thermal_emb,
        }
