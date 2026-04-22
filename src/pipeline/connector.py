import torch
import torch.nn as nn


class MLPConnector(nn.Module):
    """
    MLP connector style LLaVA-1.5.
    Projette les tokens 4M vers l'espace SmolLM2.
    (B, T, encoder_dim) → (B, T, smolvla_dim)
    """
    def __init__(self, encoder_dim: int = 1024, smolvla_dim: int = 2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, smolvla_dim),
            nn.GELU(),
            nn.Linear(smolvla_dim, smolvla_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        print(f"[Connector] output shape: {out.shape}")
        return out