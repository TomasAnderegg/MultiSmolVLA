import torch
import torch.nn as nn

from .custom_smolvla_policy import CustomSmolVLAPolicy


class SmolVLAWrapper(nn.Module):
    """Local wrapper that exposes SmolVLA as the Block 2 action module."""

    def __init__(
        self,
        pretrained: str = "lerobot/smolvla_libero",
        use_4m: bool = True,
        freeze_4m: bool = True,
        freeze_mlp: bool = False,
        fourm_checkpoint: str = "EPFL-VILAB/4M-21_XL",
        fourm_dim: int = 1024,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        print(f"[CustomSmolVLA] Loading policy from {pretrained} with 4M={use_4m} ...")
        self.policy = CustomSmolVLAPolicy(
            pretrained=pretrained,
            use_4m=use_4m,
            freeze_4m=freeze_4m,
            freeze_mlp=freeze_mlp,
            fourm_checkpoint=fourm_checkpoint,
            fourm_dim=fourm_dim,
            device=device,
        )
        self.policy.to(device)
        self.policy.eval()
        print("[CustomSmolVLA] Loaded ✅")

    def forward(self, batch: dict) -> torch.Tensor:
        return self.policy.forward(batch)

    def select_action(self, batch: dict, noise: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return self.policy.select_action(batch, noise=noise, **kwargs)
