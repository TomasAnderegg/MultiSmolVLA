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
        alpha_min: float = 0.0,
        total_epochs: int = 100,
        modalities: list = None,
        corruption_types: list = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.block1 = Block1(
            device=device,
            p_drop=p_drop,
            alpha_min=alpha_min,
            total_epochs=total_epochs,
            modalities=modalities,
            corruption_types=corruption_types,
        )
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

    def compute_distill_loss(self, inputs: dict, epoch: int = 0) -> dict:
        """Stage 1 distillation loss: align 4M+MLP tokens with SigLIP tokens.

        Computes cosine embedding loss between mean-pooled normalized tokens:
          - SigLIP: vision_model(rgb) → last_hidden_state (B, 196, 768)
          - Ours:   fourm_encoder(rgb) → fourm_to_vlm → (B, N, D_vlm)
        Both are mean-pooled and L2-normalized before the cosine loss.
        """
        inputs = self._run_block1(inputs, epoch)
        rgb = inputs["rgb"]  # (B, 3, 224, 224)

        vlm_with_expert = self.block2.smolvla.policy.base_policy.model.vlm_with_expert

        # ── SigLIP tokens (frozen) ────────────────────────────────────────────
        vision_model = vlm_with_expert.get_vlm_model().vision_model
        siglip_dtype = vision_model.dtype
        with torch.no_grad():
            siglip_out = vision_model(pixel_values=rgb.to(dtype=siglip_dtype))
            siglip_tokens = siglip_out.last_hidden_state.float()  # (B, 196, 768)

        # ── 4M + MLP tokens ───────────────────────────────────────────────────
        fourm_encoder = vlm_with_expert.fourm_encoder
        mlp_connector = vlm_with_expert.fourm_to_vlm
        fourm_tokens = fourm_encoder({"rgb": rgb})         # (B, N, D_4m)
        mlp_tokens   = mlp_connector(fourm_tokens).float() # (B, N, D_vlm)

        # ── Mean-pool + L2-normalize ──────────────────────────────────────────
        siglip_mean = torch.nn.functional.normalize(siglip_tokens.mean(dim=1), dim=-1)  # (B, 768)
        mlp_mean    = torch.nn.functional.normalize(mlp_tokens.mean(dim=1),    dim=-1)  # (B, D_vlm)

        # ── Project SigLIP to same dim as MLP output via linear (frozen) ─────
        # Lazy-init projection head stored on the pipeline
        d_siglip = siglip_mean.shape[-1]
        d_mlp    = mlp_mean.shape[-1]
        if not hasattr(self, "_siglip_proj") or self._siglip_proj.weight.shape != (d_mlp, d_siglip):
            self._siglip_proj = torch.nn.Linear(d_siglip, d_mlp, bias=False).to(rgb.device)
            torch.nn.init.eye_(self._siglip_proj.weight[:min(d_mlp, d_siglip), :min(d_mlp, d_siglip)])
            for p in self._siglip_proj.parameters():
                p.requires_grad = False  # frozen — only used to align dims

        siglip_proj = torch.nn.functional.normalize(
            self._siglip_proj(siglip_mean), dim=-1
        )  # (B, D_vlm)

        # ── Cosine embedding loss (target=1 → maximize similarity) ───────────
        target = torch.ones(rgb.shape[0], device=rgb.device)
        loss = torch.nn.functional.cosine_embedding_loss(mlp_mean, siglip_proj, target)

        # ── Also track MSE between token norms (scale alignment) ─────────────
        siglip_norm = siglip_tokens.norm(dim=-1).mean()
        mlp_norm    = mlp_tokens.norm(dim=-1).mean()
        norm_ratio  = (mlp_norm / (siglip_norm + 1e-8)).detach()

        return {
            "loss":        loss,
            "cosine_loss": loss.item(),
            "norm_ratio":  norm_ratio.item(),
            "siglip_norm": siglip_norm.item(),
            "mlp_norm":    mlp_norm.item(),
        }
