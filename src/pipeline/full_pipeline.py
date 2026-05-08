import torch
import torch.nn as nn
from .block1 import Block1
from .block2 import Block2


class VLAPipeline(nn.Module):
    """
    Full pipeline:
      Block1: {rgb, depth, seg} → ThermalGen → ModalityDropout → ImageBind → {rgb, depth, seg, thermal (B,1024)}
      Block2: {rgb, depth, seg, thermal} + batch → 4M-21 → MLP → SmolVLA → actions
    """
    def __init__(
        self,
        smolvla_checkpoint: str = "lerobot/smolvla_libero",
        fourm_checkpoint: str = "EPFL-VILAB/4M-21_XL",
        fourm_dim: int = 1024,
        use_4m: bool = True,
        freeze_4m: bool = True,
        freeze_mlp: bool = False,
        p_drop: float = 0.5,
        total_epochs: int = 100,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.block1 = Block1(device=device, p_drop=p_drop, total_epochs=total_epochs)
        self.block2 = Block2(
            smolvla_checkpoint=smolvla_checkpoint,
            fourm_checkpoint=fourm_checkpoint,
            fourm_dim=fourm_dim,
            use_4m=use_4m,
            freeze_4m=freeze_4m,
            freeze_mlp=freeze_mlp,
            device=device,
        )

    def _run_block1(self, inputs: dict, epoch: int) -> dict:
        """Block1: generate synthetic thermal, apply dropout, embed with ImageBind.
        Returns {"rgb", "depth", "seg", "thermal": (B, 1024)}."""
        return self.block1(inputs, epoch=epoch)

    def forward(self, inputs: dict, batch: dict, epoch: int = 0) -> torch.Tensor:
        """Inference: returns actions (B, action_dim).
        inputs : {"rgb": (B,3,224,224), "depth": (B,1,224,224), "seg": (B,1,224,224)}
        """
        inputs = self._run_block1(inputs, epoch)
        return self.block2(inputs, batch)

    def compute_loss(self, inputs: dict, batch: dict, epoch: int = 0) -> dict:
        """Training: returns loss dict from SmolVLA policy.
        inputs : {"rgb": (B,3,224,224), "depth": (B,1,224,224), "seg": (B,1,224,224)}
        """
        inputs = self._run_block1(inputs, epoch)
        return self.block2.compute_loss(inputs, batch)
