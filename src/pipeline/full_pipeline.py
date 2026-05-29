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

    def _siglip_features(self, rgb: torch.Tensor) -> torch.Tensor:
        """SigLIP + connector features — the correct distillation target.

        Replicates exactly what prepare_images + embed_image does at inference:
          resize → (512,512), normalize [0,1]→[-1,1],
          vision_model → connector → (B, N_siglip, 960)
        """
        vlm         = self.block2.smolvla.policy.base_policy.model.vlm_with_expert
        resize_hw   = self.block2.smolvla.policy.config.resize_imgs_with_padding
        vision_model = vlm.get_vlm_model().vision_model
        connector    = vlm.get_vlm_model().connector

        rgb_prep = F.interpolate(rgb, size=resize_hw, mode='bilinear', align_corners=False)
        rgb_prep = rgb_prep * 2.0 - 1.0   # [0,1] → [-1,1] as SigLIP expects

        with torch.no_grad():
            hidden = vision_model(
                pixel_values=rgb_prep.to(dtype=vision_model.dtype)
            ).last_hidden_state
            tokens = connector(hidden).float()   # (B, N_siglip, D_vlm)

        return tokens

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
        """Stage 1 distillation: per-token cosine alignment of 4M+MLP vs SigLIP+connector.

        Target = embed_image() output: SigLIP vision_model → SmolVLM connector → (B, N, 960).
        Input normalization matches inference: resize (512,512), [-1,1].
        Token count mismatch handled by linear interpolation.
        """
        inputs = self._run_block1(inputs, epoch)
        rgb = inputs["rgb"]

        vlm = self.block2.smolvla.policy.base_policy.model.vlm_with_expert

        # Correct target: full SigLIP + connector path, inference normalization
        siglip_tokens = self._siglip_features(rgb)     # (B, N_sig, D_vlm)

        # 4M + MLP tokens
        fourm_tokens = vlm.fourm_encoder(inputs)
        mlp_tokens   = vlm.fourm_to_vlm(fourm_tokens).float()   # (B, N_4m, D_vlm)

        B, N_4m, D = mlp_tokens.shape
        _, N_sig, _ = siglip_tokens.shape

        # Align token counts if different (linear interpolation over spatial dim)
        if N_4m != N_sig:
            siglip_tokens = F.interpolate(
                siglip_tokens.permute(0, 2, 1),   # (B, D, N_sig)
                size=N_4m, mode='linear', align_corners=False
            ).permute(0, 2, 1)                    # (B, N_4m, D)

        mlp_flat = F.normalize(mlp_tokens.reshape(B * N_4m, D), dim=-1)
        sig_flat = F.normalize(siglip_tokens.reshape(B * N_4m, D), dim=-1)
        target   = torch.ones(B * N_4m, device=rgb.device)
        loss     = F.cosine_embedding_loss(mlp_flat, sig_flat, target)

        with torch.no_grad():
            mean_cos = (mlp_flat * sig_flat).sum(dim=-1).mean()
            mlp_norm = mlp_tokens.norm(dim=-1).mean()
            sig_norm = siglip_tokens.norm(dim=-1).mean()

        return {
            "loss":                   loss,
            "cosine_loss":            loss.item(),
            "mean_cosine_similarity": mean_cos.item(),
            "mlp_norm":               mlp_norm.item(),
            "siglip_norm":            sig_norm.item(),
            "n_siglip_tokens":        N_sig,
            "n_fourm_tokens":         N_4m,
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

        # ── Distillation loss (per-token, correct target) ─────────────────────
        vlm = self.block2.smolvla.policy.base_policy.model.vlm_with_expert
        siglip_tokens = self._siglip_features(rgb)          # (B, N_sig, D_vlm)
        fourm_tokens  = vlm.fourm_encoder(inputs)
        mlp_tokens    = vlm.fourm_to_vlm(fourm_tokens).float()   # (B, N_4m, D_vlm)

        B, N_4m, D = mlp_tokens.shape
        _, N_sig, _ = siglip_tokens.shape
        if N_4m != N_sig:
            siglip_tokens = F.interpolate(
                siglip_tokens.permute(0, 2, 1),
                size=N_4m, mode='linear', align_corners=False
            ).permute(0, 2, 1)

        mlp_flat     = F.normalize(mlp_tokens.reshape(B * N_4m, D),     dim=-1)
        sig_flat     = F.normalize(siglip_tokens.reshape(B * N_4m, D),  dim=-1)
        target       = torch.ones(B * N_4m, device=rgb.device)
        distill_loss = F.cosine_embedding_loss(mlp_flat, sig_flat, target)

        total_loss = action_loss + lambda_distill * distill_loss

        return {
            "loss":         total_loss,
            "action_loss":  action_loss.item(),
            "distill_loss": distill_loss.item(),
            "mlp_norm":     mlp_tokens.norm(dim=-1).mean().item(),
        }
