from dataclasses import dataclass

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@dataclass
class LocalSmolVLAConfig(SmolVLAConfig):
    """
    Local SmolVLA configuration for the custom 4M + SmolVLA adapter.
    This extends the upstream Lerobot `SmolVLAConfig` with 4M-specific flags.

    Freeze flags (all default to False = trainable):
      freeze_4m           : freeze the 4M encoder
      freeze_mlp          : freeze the MLP connector
      freeze_smolvlm      : freeze the SmolVLM language model
      freeze_action_expert: freeze the action expert
    """

    # 4M settings
    use_4m: bool = True
    fourm_checkpoint: str = "EPFL-VILAB/4M-21_XL"
    fourm_dim: int = 1024

    # Freeze flags
    freeze_4m: bool = False
    freeze_mlp: bool = False
    freeze_smolvlm: bool = False
    freeze_action_expert: bool = False
