import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPConnector(nn.Module):
    """
    MLP connector: 4M tokens → SmolLM2 token space.
    (B, N_in, encoder_dim) → (B, n_out_tokens, smolvla_dim)

    n_out_tokens: must match what SigLIP+connector produces (64 for SmolVLM2-500M).
      Without this, SmolLM2 receives a different number of visual tokens than it
      was trained on, which shifts all RoPE text positions and breaks the policy.

    LayerNorm: controls token norms before SmolVLA's ×√hidden_size scaling (~×31).
      Without it, Adam grows magnitudes freely during cosine distillation.
    """
    def __init__(self, encoder_dim: int = 2048, smolvla_dim: int = 960,
                 n_out_tokens: int = 64, use_layer_norm: bool = True):
        super().__init__()
        self.n_out_tokens = n_out_tokens
        layers = [
            nn.Linear(encoder_dim, smolvla_dim),
            nn.GELU(),
            nn.Linear(smolvla_dim, smolvla_dim),
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(smolvla_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_in, encoder_dim)  e.g. (B, 196, 2048)
        x = self.mlp(x)              # (B, N_in, smolvla_dim)
        if self.n_out_tokens is not None and x.shape[1] != self.n_out_tokens:
            # Adaptive average pooling along the token dimension
            # (B, N_in, D) → (B, n_out_tokens, D)
            x = F.adaptive_avg_pool1d(
                x.permute(0, 2, 1),      # (B, D, N_in)
                self.n_out_tokens
            ).permute(0, 2, 1)           # (B, n_out_tokens, D)
        return x