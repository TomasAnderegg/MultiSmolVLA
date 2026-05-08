import os
import sys

import torch
import torch.nn as nn

# Ensure the local 4M package is importable as a top-level `fourm` package.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_fourm_root = os.path.join(_repo_root, "third_party", "ml_4m")
if _fourm_root not in sys.path:
    sys.path.insert(0, _fourm_root)

from fourm.models.fm import FM


class Encoder4M(nn.Module):
    """
    Wrapper around 4M-21, RGB-only encoder.
    Only RGB is passed through 4M for now (depth/thermal go through Block 1).

    Input  : dict with at least "rgb": (B, 3, 224, 224)
    Output : (B, 196, D)  —  196 patch tokens, D = model hidden dim
    """
    # rgb@224 with patch_size=16 → (224/16)^2 = 196 tokens
    NUM_PATCHES = 196

    def __init__(self, checkpoint: str = "EPFL-VILAB/4M-21_XL", device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        print(f"[Encoder4M] Loading 4M-21 from {checkpoint} ({dtype}) ...")
        self.model = FM.from_pretrained(checkpoint)
        self.model.to(device=device, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.output_dim = self.model.dim  # e.g. 768 for B, 1024 for L/XL
        print(f"[Encoder4M] Loaded ✅ (output_dim={self.output_dim})")

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        inputs : {
            "rgb"     : (B, 3, 224, 224),
            "depth"   : ignored for now
            "seg"     : ignored for now
            "thermal" : ignored for now
        }
        Returns: (B, 196, D)
        """
        rgb = inputs["rgb"].to(dtype=self.dtype)
        B = rgb.shape[0]
        device = rgb.device

        # 1. Build the mod_dict expected by 4M:
        #    input_mask=False → all tokens visible (no masking)
        mod_dict = {
            "rgb@224": {
                "tensor": rgb,
                "input_mask": torch.zeros(B, self.NUM_PATCHES, dtype=torch.bool, device=device),
            }
        }

        with torch.no_grad():
            # 2. Apply per-modality encoder embeddings (image → patch tokens)
            encoder_mod_dict = {
                mod: self.model.encoder_embeddings[mod](d)
                for mod, d in mod_dict.items()
                if mod in self.model.encoder_embeddings
            }

            # 3. Concatenate + select tokens (no masking → keep all 196)
            encoder_tokens, encoder_emb, encoder_mask, _ = self.model.forward_mask_encoder(
                encoder_mod_dict, num_encoder_tokens=self.NUM_PATCHES
            )

            # 4. Run transformer encoder
            x = encoder_tokens + encoder_emb
            tokens = self.model.forward_encoder(x, encoder_mask=encoder_mask)  # (B, 196, D)

        return tokens
