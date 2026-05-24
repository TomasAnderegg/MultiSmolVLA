import os
import sys

import torch
import torch.nn as nn

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_fourm_root = os.path.join(_repo_root, "third_party", "ml_4m")
if _fourm_root not in sys.path:
    sys.path.insert(0, _fourm_root)

from fourm.models.fm import FM


class PatchEmbedder(nn.Module):
    """Patchify a single-channel image (depth or seg) and project to 4M hidden dim."""
    PATCH_SIZE = 16

    def __init__(self, out_dim: int = 1024):
        super().__init__()
        in_dim = self.PATCH_SIZE * self.PATCH_SIZE  # 1 channel × 16×16 = 256
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, 224, 224) → (B, 196, out_dim)"""
        B, C, H, W = x.shape
        ps = self.PATCH_SIZE
        # patchify: (B, nh, nw, C, ps, ps) → (B, N, C*ps*ps)
        x = x.reshape(B, C, H // ps, ps, W // ps, ps)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * ps * ps)
        return self.proj(x)


class Encoder4M(nn.Module):
    """
    Wrapper around 4M-21. Encodes RGB (continuous) + optional depth/seg (learnable patch embedders).

    depth and seg are added as residual tokens on top of RGB tokens — same spatial positions,
    no change to token count. The depth/seg embedders are the only trainable parameters here;
    the 4M backbone stays frozen.

    Input  : dict with "rgb": (B, 3, 224, 224), optionally "depth": (B, 1, 224, 224),
             "seg": (B, 1, 224, 224)
    Output : (B, 196, D)  —  196 patch tokens, D = model hidden dim
    """
    NUM_PATCHES = 196

    def __init__(self, checkpoint: str = "EPFL-VILAB/4M-21_XL", device: str = "cuda",
                 dtype: torch.dtype = torch.float32,
                 use_depth: bool = False, use_seg: bool = False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.use_depth = use_depth
        self.use_seg = use_seg

        print(f"[Encoder4M] Loading 4M-21 from {checkpoint} ({dtype}) ...")
        try:
            self.model = FM.from_pretrained(checkpoint, local_files_only=True)
            print("[Encoder4M] Loaded from local cache")
        except Exception:
            print("[Encoder4M] Not in cache — downloading from Hub ...")
            self.model = FM.from_pretrained(checkpoint, local_files_only=False)
        self.model.to(device=device, dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.output_dim = self.model.dim

        if use_depth:
            self.depth_embed = PatchEmbedder(out_dim=self.output_dim)
            print("[Encoder4M] depth patch embedder added ✅")
        if use_seg:
            self.seg_embed = PatchEmbedder(out_dim=self.output_dim)
            print("[Encoder4M] seg patch embedder added ✅")

        print(f"[Encoder4M] Loaded ✅ (output_dim={self.output_dim})")

    def forward(self, inputs: dict) -> torch.Tensor:
        rgb = inputs["rgb"].to(dtype=self.dtype, device=self.device)
        B = rgb.shape[0]
        device = rgb.device

        mod_dict = {
            "rgb@224": {
                "tensor": rgb,
                "input_mask": torch.zeros(B, self.NUM_PATCHES, dtype=torch.bool, device=device),
            }
        }

        with torch.no_grad():
            encoder_mod_dict = {
                mod: self.model.encoder_embeddings[mod](d)
                for mod, d in mod_dict.items()
                if mod in self.model.encoder_embeddings
            }
            encoder_tokens, encoder_emb, encoder_mask, _ = self.model.forward_mask_encoder(
                encoder_mod_dict, num_encoder_tokens=self.NUM_PATCHES
            )
            x = encoder_tokens + encoder_emb
            tokens = self.model.forward_encoder(x, encoder_mask=encoder_mask)  # (B, 196, D)

        # Additive depth/seg tokens (same spatial positions as RGB)
        target_dtype = self.depth_embed.proj[0].weight.dtype if self.use_depth else tokens.dtype
        tokens = tokens.to(dtype=target_dtype)

        if self.use_depth and "depth" in inputs:
            depth = inputs["depth"].to(dtype=target_dtype, device=device)
            tokens = tokens + self.depth_embed(depth)

        if self.use_seg and "seg" in inputs:
            seg = inputs["seg"].to(dtype=target_dtype, device=device)
            tokens = tokens + self.seg_embed(seg)

        return tokens
