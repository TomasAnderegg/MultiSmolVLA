import os
import sys
import types
from collections import deque
from typing import Optional

import torch
from torch import Tensor, nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LEROBOT_SRC = os.path.join(ROOT, "third_party", "lerobot", "src")
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .configuration_smolvla import LocalSmolVLAConfig
from ..connector import MLPConnector
from ..encoder_4m import Encoder4M


class CustomSmolVLAPolicy(nn.Module):
    """A local SmolVLA policy wrapper that can swap in a 4M encoder + MLP connector."""

    def __init__(
        self,
        config: Optional[SmolVLAConfig] = None,
        pretrained: Optional[str] = None,
        use_4m: Optional[bool] = None,
        freeze_4m: Optional[bool] = None,
        freeze_mlp: Optional[bool] = None,
        fourm_checkpoint: Optional[str] = None,
        fourm_dim: Optional[int] = None,
        device: str = "cuda",
    ):
        super().__init__()
        if pretrained is None and config is None:
            raise ValueError("Either `pretrained` or `config` must be provided.")

        self.config = config
        if pretrained is not None:
            self.base_policy = SmolVLAPolicy.from_pretrained(pretrained)
            if self.config is None:
                self.config = self.base_policy.config
        else:
            assert self.config is not None
            self.base_policy = SmolVLAPolicy(self.config)

        if isinstance(self.config, LocalSmolVLAConfig):
            use_4m = self.config.use_4m if use_4m is None else use_4m
            freeze_4m = self.config.freeze_4m if freeze_4m is None else freeze_4m
            freeze_mlp = self.config.freeze_mlp if freeze_mlp is None else freeze_mlp
            fourm_checkpoint = self.config.fourm_checkpoint if fourm_checkpoint is None else fourm_checkpoint
            fourm_dim = self.config.fourm_dim if fourm_dim is None else fourm_dim

        use_4m = True if use_4m is None else use_4m
        freeze_4m = False if freeze_4m is None else freeze_4m
        freeze_mlp = False if freeze_mlp is None else freeze_mlp
        fourm_checkpoint = "EPFL-VILAB/4M-21_XL" if fourm_checkpoint is None else fourm_checkpoint
        fourm_dim = 1024 if fourm_dim is None else fourm_dim

        if use_4m:
            self._patch_visual_backbone(
                use_4m=use_4m,
                freeze_4m=freeze_4m,
                freeze_mlp=freeze_mlp,
                fourm_checkpoint=fourm_checkpoint,
                fourm_dim=fourm_dim,
                device=device,
            )

        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def _patch_visual_backbone(
        self,
        use_4m: bool,
        freeze_4m: bool,
        freeze_mlp: bool,
        fourm_checkpoint: str,
        fourm_dim: int,
        device: str,
    ) -> None:
        # Patch the already-loaded vlm_with_expert in-place so we don't load SmolVLM2 a second time
        # and so we keep the pretrained lerobot/smolvla_libero weights.
        vlm = self.base_policy.model.vlm_with_expert

        vlm.use_4m = use_4m

        # Attach 4M encoder — nn.Module.__setattr__ registers it as a proper submodule
        vlm.fourm_encoder = Encoder4M(checkpoint=fourm_checkpoint, device=device)
        if freeze_4m:
            for p in vlm.fourm_encoder.parameters():
                p.requires_grad = False

        # MLP connector: use the actual output dim from the loaded encoder (768 for B, 1024 for L/XL)
        target_dim = vlm.config.text_config.hidden_size
        vlm.fourm_to_vlm = MLPConnector(
            encoder_dim=vlm.fourm_encoder.output_dim,
            smolvla_dim=target_dim,
        )
        if freeze_mlp:
            for p in vlm.fourm_to_vlm.parameters():
                p.requires_grad = False

        # Monkey-patch embed_image on this instance, falling back to the original class method
        original_embed_image = vlm.embed_image  # bound to vlm, calls SmolVLMWithExpertModel.embed_image

        def _embed_image(self_vlm, image):
            if self_vlm.use_4m:
                if isinstance(image, dict):
                    tokens = self_vlm.fourm_encoder(image)
                elif image.ndim == 3:
                    tokens = image
                else:
                    return original_embed_image(image)
                tokens = tokens.to(self_vlm.fourm_to_vlm.mlp[0].weight.dtype)
                return self_vlm.fourm_to_vlm(tokens)
            return original_embed_image(image)

        vlm.embed_image = types.MethodType(_embed_image, vlm)

        # prepare_images lives on SmolVLAPolicy (self.base_policy), not on VLAFlowMatching (self.base_policy.model)
        self.base_policy._base_prepare_images = self.base_policy.prepare_images
        self.base_policy.prepare_images = self._prepare_images.__get__(
            self.base_policy,
            type(self.base_policy),
        )

    def _prepare_images(self, batch: dict) -> tuple[list[torch.Tensor | dict], list[torch.Tensor]]:
        image_key = "observation.images.4m"
        if image_key in batch:
            image = batch[image_key]
            if isinstance(image, dict) or image.ndim == 3:
                if isinstance(image, dict):
                    device = next(iter(image.values())).device
                    batch_size = next(iter(image.values())).shape[0]
                else:
                    device = image.device
                    batch_size = image.shape[0]

                image_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                return [image], [image_mask]

        return self._base_prepare_images(batch)

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean") -> dict[str, Tensor]:
        # attention_mask from tokenizers is Long (0/1); smolvlm_with_expert needs bool
        if "observation.language.attention_mask" in batch:
            batch = dict(batch)
            batch["observation.language.attention_mask"] = batch["observation.language.attention_mask"].bool()
        return self.base_policy.forward(batch, noise=noise, time=time, reduction=reduction)

    def select_action(self, batch: dict[str, Tensor], noise: Optional[Tensor] = None, **kwargs) -> Tensor:
        if "observation.language.attention_mask" in batch:
            batch = dict(batch)
            batch["observation.language.attention_mask"] = batch["observation.language.attention_mask"].bool()
        return self.base_policy.select_action(batch, noise=noise, **kwargs)

    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Optional[Tensor] = None, **kwargs) -> Tensor:
        return self.base_policy.predict_action_chunk(batch, noise=noise, **kwargs)

    def get_optim_params(self) -> dict:
        return self.base_policy.get_optim_params()

    def reset(self) -> None:
        self.base_policy.reset()
