import torch
import torch.nn as nn
import torch.nn.functional as F
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
        skip_block1: bool = False,
        use_depth: bool = False,
        use_seg: bool = False,
    ):
        super().__init__()
        self.device = device
        self.skip_block1 = skip_block1
        if not skip_block1:
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
            use_depth=use_depth,
            use_seg=use_seg,
        )

    def _run_block1(self, inputs: dict, epoch: int) -> dict:
        if self.skip_block1:
            return inputs
        return self.block1(inputs, epoch=epoch)

    def forward(self, inputs: dict, batch: dict, epoch: int = 0) -> torch.Tensor:
        inputs = self._run_block1(inputs, epoch)
        return self.block2(inputs, batch)

    def compute_loss(self, inputs: dict, batch: dict, epoch: int = 0) -> dict:
        """Training: returns loss dict from SmolVLA policy.
        inputs : {"rgb": (B,3,224,224), "depth": (B,1,224,224), "seg": (B,1,224,224)}
        """
        inputs = self._run_block1(inputs, epoch)
        return self.block2.compute_loss(inputs, batch)

    def _get_siglip_proj(self, d_siglip: int, d_mlp: int, device) -> torch.nn.Linear:
        if not hasattr(self, "_siglip_proj") or self._siglip_proj.weight.shape != (d_mlp, d_siglip):
            self._siglip_proj = torch.nn.Linear(d_siglip, d_mlp, bias=False).to(device)
            torch.nn.init.zeros_(self._siglip_proj.weight)
            n = min(d_mlp, d_siglip)
            self._siglip_proj.weight.data[:n, :n] = torch.eye(n)
            for p in self._siglip_proj.parameters():
                p.requires_grad = False
        return self._siglip_proj

    def compute_distill_loss(self, inputs: dict, epoch: int = 0) -> dict:
        """Stage 1 distillation loss: per-token cosine alignment of 4M+MLP tokens with SigLIP.

        Aligns each of the 196 spatial tokens individually (not just the mean-pooled global
        representation). This forces per-patch directional alignment, which is what SmolVLA's
        cross-attention actually needs.
        """
        inputs = self._run_block1(inputs, epoch)
        rgb = inputs["rgb"]  # (B, 3, 224, 224)

        vlm_with_expert = self.block2.smolvla.policy.base_policy.model.vlm_with_expert

        # ── SigLIP tokens (frozen) ────────────────────────────────────────────
        vision_model = vlm_with_expert.get_vlm_model().vision_model
        with torch.no_grad():
            siglip_out = vision_model(pixel_values=rgb.to(dtype=vision_model.dtype))
            siglip_tokens = siglip_out.last_hidden_state.float()  # (B, 196, 768)

        # ── 4M + MLP tokens ───────────────────────────────────────────────────
        fourm_tokens = vlm_with_expert.fourm_encoder(inputs)      # (B, 196, D_4m)
        mlp_tokens   = vlm_with_expert.fourm_to_vlm(fourm_tokens).float()  # (B, 196, D_vlm)

        B, N, D_mlp    = mlp_tokens.shape
        _,  _, D_siglip = siglip_tokens.shape

        # ── Project SigLIP 768 → D_vlm per token (frozen near-identity) ──────
        proj = self._get_siglip_proj(D_siglip, D_mlp, rgb.device)
        siglip_proj = proj(siglip_tokens)  # (B, 196, D_mlp)

        # ── Per-token cosine loss ─────────────────────────────────────────────
        mlp_flat  = F.normalize(mlp_tokens.reshape(B * N, D_mlp),   dim=-1)
        sig_flat  = F.normalize(siglip_proj.reshape(B * N, D_mlp),  dim=-1)
        target    = torch.ones(B * N, device=rgb.device)
        loss      = F.cosine_embedding_loss(mlp_flat, sig_flat, target)

        with torch.no_grad():
            mean_cos  = (mlp_flat * sig_flat).sum(dim=-1).mean()
            mlp_norm  = mlp_tokens.norm(dim=-1).mean()
            sig_norm  = siglip_tokens.norm(dim=-1).mean()

        return {
            "loss":                   loss,
            "cosine_loss":            loss.item(),
            "mean_cosine_similarity": mean_cos.item(),
            "mlp_norm":               mlp_norm.item(),
            "siglip_norm":            sig_norm.item(),
        }

    def compute_joint_loss(self, inputs: dict, batch: dict, epoch: int = 0,
                           lambda_distill: float = 0.1) -> dict:
        """Stage 3: L_total = L_action + lambda_distill * L_distill.

        Runs Block1 once and shares the output between the action loss and the
        distillation loss so Block1 is not executed twice.

        - L_action : flow-matching loss from SmolVLA (trains the full policy)
        - L_distill: cosine loss keeping MLP tokens aligned with SigLIP tokens
                     (prevents the MLP from drifting as the action loss optimizes it)
        """
        inputs = self._run_block1(inputs, epoch)
        rgb = inputs["rgb"]  # (B, 3, 224, 224)

        # ── Action loss ───────────────────────────────────────────────────────
        action_dict = self.block2.compute_loss(inputs, batch)
        action_loss = action_dict["loss"]

        # ── Distillation loss (per-token) ─────────────────────────────────────
        vlm_with_expert = self.block2.smolvla.policy.base_policy.model.vlm_with_expert
        vision_model = vlm_with_expert.get_vlm_model().vision_model
        with torch.no_grad():
            siglip_out    = vision_model(pixel_values=rgb.to(dtype=vision_model.dtype))
            siglip_tokens = siglip_out.last_hidden_state.float()  # (B, 196, 768)

        fourm_tokens = vlm_with_expert.fourm_encoder(inputs)
        mlp_tokens   = vlm_with_expert.fourm_to_vlm(fourm_tokens).float()  # (B, 196, D_vlm)

        B, N, D_mlp    = mlp_tokens.shape
        _,  _, D_siglip = siglip_tokens.shape

        proj         = self._get_siglip_proj(D_siglip, D_mlp, rgb.device)
        siglip_proj  = proj(siglip_tokens)
        mlp_flat     = F.normalize(mlp_tokens.reshape(B * N, D_mlp),  dim=-1)
        sig_flat     = F.normalize(siglip_proj.reshape(B * N, D_mlp), dim=-1)
        target       = torch.ones(B * N, device=rgb.device)
        distill_loss = F.cosine_embedding_loss(mlp_flat, sig_flat, target)

        total_loss = action_loss + lambda_distill * distill_loss

        return {
            "loss":         total_loss,
            "action_loss":  action_loss.item(),
            "distill_loss": distill_loss.item(),
            "mlp_norm":     mlp_tokens.norm(dim=-1).mean().item(),
        }
