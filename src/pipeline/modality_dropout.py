import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ModalityDropout(nn.Module):
    """
    Module de dropout unifié pour les modalités.
    
    alpha = 1.0  →  modalité clean
    alpha = 0.5  →  corruption modérée (soft)
    alpha = 0.0  →  zéro complet (hard dropout)
    
    Le curriculum schedule pilote alpha de 1.0 → 0.0 au fil des epochs.
    """

    def __init__(
        self,
        modalities: list = ["rgb", "depth", "seg", "thermal"],
        p_drop: float = 0.5,
        alpha_min: float = 0.0,
        total_epochs: int = 100,
        corruption_types: list = ["gaussian", "blur", "occlusion"],
    ):
        super().__init__()
        self.modalities = modalities
        self.p_drop = p_drop
        self.alpha_min = alpha_min
        self.total_epochs = total_epochs
        self.corruption_types = corruption_types

    # ------------------------------------------------------------------
    # Curriculum schedule
    # ------------------------------------------------------------------

    def get_alpha(self, epoch: int) -> float:
        """
        alpha décroît linéairement de 1.0 → alpha_min sur total_epochs.
        """
        progress = min(epoch / self.total_epochs, 1.0)
        alpha = 1.0 - progress * (1.0 - self.alpha_min)
        return alpha

    # ------------------------------------------------------------------
    # Corruptions soft
    # ------------------------------------------------------------------

    def gaussian_noise(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """
        intensity ∈ [0, 1] — sigma max = 0.5 * intensity
        """
        sigma = 0.5 * intensity
        noise = torch.randn_like(x) * sigma
        return x + noise

    def motion_blur(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """
        Blur horizontal, kernel_size proportionnel à intensity.
        """
        # kernel_size impair entre 1 et 21
        kernel_size = max(1, int(intensity * 20))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size == 1:
            return x

        B, C, H, W = x.shape

        # Kernel 1D horizontal
        kernel = torch.ones(1, 1, 1, kernel_size, device=x.device) / kernel_size

        # Applique canal par canal
        padding = kernel_size // 2
        x_blurred = F.conv2d(
            x.view(B * C, 1, H, W),
            kernel,
            padding=(0, padding)
        )
        return x_blurred.view(B, C, H, W)

    def occlusion(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """
        Carré noir centré, taille proportionnelle à intensity.
        """
        x = x.clone()
        B, C, H, W = x.shape

        # Taille du carré : 0% → 100% de l'image
        size_h = int(intensity * H)
        size_w = int(intensity * W)

        if size_h == 0 or size_w == 0:
            return x

        # Centre
        h_start = (H - size_h) // 2
        w_start = (W - size_w) // 2

        x[:, :, h_start:h_start + size_h, w_start:w_start + size_w] = 0.0
        return x

    # ------------------------------------------------------------------
    # Corruption unifiée
    # ------------------------------------------------------------------

    def corrupt(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        output = alpha * x + (1 - alpha) * x_corrupted
        Si alpha == 0.0 → hard dropout pur
        """
        intensity = 1.0 - alpha

        if alpha == 0.0:
            return torch.zeros_like(x)

        corruption_type = random.choice(self.corruption_types)

        if corruption_type == "gaussian":
            x_corrupted = self.gaussian_noise(x, intensity)
        elif corruption_type == "blur":
            x_corrupted = self.motion_blur(x, intensity)
        elif corruption_type == "occlusion":
            x_corrupted = self.occlusion(x, intensity)
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

        return alpha * x + (1.0 - alpha) * x_corrupted

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: dict, epoch: int = 0) -> dict:
        """
        inputs : dict modalité → tensor (B, C, H, W)
        epoch  : epoch actuelle pour le curriculum schedule
        """
        # À l'inférence : pas de corruption (comportement standard)
        if not self.training:
            return inputs

        alpha = self.get_alpha(epoch)
        outputs = {}

        for modality, x in inputs.items():
            if random.random() < self.p_drop:
                outputs[modality] = self.corrupt(x, alpha)
                print(f"[ModalityDropout] {modality} corrupted (alpha={alpha:.2f})")
            else:
                outputs[modality] = x

        return outputs