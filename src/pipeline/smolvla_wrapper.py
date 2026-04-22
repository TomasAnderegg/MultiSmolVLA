import torch
import torch.nn as nn
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class SmolVLAWrapper(nn.Module):
    """
    Wrapper autour du vrai SmolVLA.
    Input  : tokens (B, T, D) + batch dict (state, task)
    Output : actions (B, action_dim)
    """
    def __init__(
        self,
        pretrained: str = "lerobot/smolvla_base",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        print(f"[SmolVLA] Loading from {pretrained} ...")
        self.policy = SmolVLAPolicy.from_pretrained(pretrained)
        self.policy.eval()
        print("[SmolVLA] Loaded ✅")

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch : dict standard SmolVLA avec
                "observation.images.top" : (B, 3, 224, 224)
                "observation.state"      : (B, state_dim)
                "task"                   : list[str]
        """
        with torch.no_grad():
            actions = self.policy.select_action(batch)
        print(f"[SmolVLA] actions shape: {actions.shape}")
        return actions