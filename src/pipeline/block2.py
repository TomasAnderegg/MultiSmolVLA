import torch
import torch.nn as nn

from .smolvla.smolvla_wrapper import SmolVLAWrapper


class Block2(nn.Module):
    """
    Block 2 — Action.
    Input :
      inputs : {
          "rgb": (B, 3, 224, 224),
          "depth": (B, 1, 224, 224),
          "seg": (B, 1, 224, 224),
          "thermal": (B, 1024),
      }
      batch : dict containing SmolVLA state/language/task info
    Output :
      actions : (B, action_dim)
    """

    def __init__(
        self,
        fourm_checkpoint: str = "EPFL-VILAB/4M-21_XL",
        smolvla_checkpoint: str = "lerobot/smolvla_libero",
        use_4m: bool = True,
        freeze_4m: bool = True,
        freeze_mlp: bool = False,
        fourm_dim: int = 1024,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.smolvla = SmolVLAWrapper(
            pretrained=smolvla_checkpoint,
            use_4m=use_4m,
            freeze_4m=freeze_4m,
            freeze_mlp=freeze_mlp,
            fourm_checkpoint=fourm_checkpoint,
            fourm_dim=fourm_dim,
            device=device,
        )

    def forward(self, inputs: dict, batch: dict) -> torch.Tensor:
        """Inference: returns actions (B, action_dim)."""
        smol_batch = dict(batch)
        smol_batch["observation.images.4m"] = inputs
        actions = self.smolvla.select_action(smol_batch)
        return actions

    def compute_loss(self, inputs: dict, batch: dict) -> dict:
        """Training: returns loss dict from SmolVLA policy."""
        smol_batch = dict(batch)
        smol_batch["observation.images.4m"] = inputs
        loss, loss_dict = self.smolvla.policy.forward(smol_batch)
        return {"loss": loss, **loss_dict}