#!/usr/bin/env python
"""
Train the full VLA pipeline (Block1: ThermalGen+ImageBind, Block2: 4M+MLP+SmolVLA) on a LIBERO dataset.

Usage examples:

  # Stage 1: train only the MLP connector
  python scripts/train_full_pipeline.py \
      --freeze_thermalgen --freeze_imagebind --freeze_4m --freeze_smolvlm --freeze_action_expert

  # Stage 2: train MLP + action expert
  python scripts/train_full_pipeline.py \
      --freeze_thermalgen --freeze_imagebind --freeze_4m

  # Train everything except ImageBind (usually kept frozen)
  python scripts/train_full_pipeline.py \
      --freeze_imagebind

  # Custom dataset
  python scripts/train_full_pipeline.py \
      --dataset lerobot/libero_object_no_noops \
      --image_key observation.images.top \
      --freeze_imagebind --freeze_4m
"""

import argparse
import contextlib
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "lerobot", "src"))

from src.pipeline.modality_dropout import ModalityDropout

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train full VLA pipeline on LIBERO")

    # Dataset — use --data_dir to load pre-generated parquet shards (recommended),
    # or --dataset to stream from HuggingFace (thermal generated on-the-fly, slower)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to parquet_thermal/ directory (pre-generated all 4 modalities)")
    parser.add_argument("--dataset", type=str, default="lerobot/libero_spatial_no_noops",
                        help="HuggingFace dataset id (fallback when --data_dir is not set)")
    parser.add_argument("--image_key", type=str, default="observation.images.top",
                        help="Dataset key to use as RGB input (only used with --dataset)")

    # Checkpoints
    parser.add_argument("--smolvla_checkpoint", type=str, default="lerobot/smolvla_libero")
    parser.add_argument("--fourm_model", type=str, default=None, choices=["B", "L", "XL"],
                        help="4M-21 variant shortcut (B=768d, L/XL=1024d). "
                             "Overrides --fourm_checkpoint and --fourm_dim when set.")
    parser.add_argument("--fourm_checkpoint", type=str, default="EPFL-VILAB/4M-21_XL")
    parser.add_argument("--fourm_dim", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="checkpoints/full_pipeline")

    # Skip Block1 entirely (no ThermalGen / ImageBind — faster training)
    parser.add_argument("--no_block1", action="store_true",
                        help="Skip Block1 (ThermalGen+ImageBind). Pass RGB directly to 4M.")
    parser.add_argument("--use_depth", action="store_true",
                        help="Feed depth channel into 4M via learnable PatchEmbedder (additive residual).")
    parser.add_argument("--use_seg", action="store_true",
                        help="Feed segmentation channel into 4M via learnable PatchEmbedder (additive residual).")

    # Mixed precision
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 autocast for forward passes (faster on A100).")

    # Freeze flags — Block1
    parser.add_argument("--freeze_thermalgen", action="store_true", help="Freeze ThermalGen (RGB→thermal generator)")
    parser.add_argument("--freeze_imagebind", action="store_true", help="Freeze ImageBind thermal encoder")
    # Freeze flags — Block2
    parser.add_argument("--freeze_4m", action="store_true", help="Freeze the 4M backbone (depth/seg embedders stay trainable)")
    parser.add_argument("--freeze_mlp", action="store_true", help="Freeze the MLP connector")
    parser.add_argument("--freeze_smolvlm", action="store_true", help="Freeze the SmolVLM language model")
    parser.add_argument("--freeze_action_expert", action="store_true", help="Freeze the action expert")

    # Resume from a full VLAPipeline checkpoint (e.g. distillation output)
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a full VLAPipeline .pt checkpoint to resume from. "
                             "Loaded after pipeline construction, before training.")

    # Train action head from scratch (random init) with pretrained SmolVLM2 backbone.
    # Mirrors baseline lerobot approach: load_vlm_weights=True, no pretrained action head.
    parser.add_argument("--from_scratch", action="store_true",
                        help="Ignore --smolvla_checkpoint and initialise the action head randomly. "
                             "The SmolVLM2 VLM backbone is still loaded from HuggingFace (load_vlm_weights=True).")

    # Distillation loss (Stage 1 alignment)
    parser.add_argument("--distill", action="store_true",
                        help="Use feature distillation loss (cosine SigLIP vs 4M+MLP) "
                             "instead of action prediction loss. Recommended for Stage 1.")

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_mlp", type=float, default=1e-3)
    parser.add_argument("--lr_action", type=float, default=1e-6,
                        help="Learning rate for the action head (Stage 3). Very low to prevent "
                             "catastrophic forgetting of gripper timing.")
    parser.add_argument("--joint", action="store_true",
                        help="Stage 3: use joint loss L_action + lambda_distill * L_distill.")
    parser.add_argument("--lambda_distill", type=float, default=0.1,
                        help="Weight of the distillation term in the joint loss (Stage 3).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_lang_tokens", type=int, default=48)
    parser.add_argument("--task_ids", type=int, nargs="+", default=None,
                        help="Filter training data to these task indices (e.g. --task_ids 2). "
                             "Default: all tasks.")
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Number of future action steps per training sample (must match SmolVLA chunk_size)")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="multismolvla")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # LoRA flags
    parser.add_argument("--lora", action="store_true",
                        help="Apply LoRA adapters to the SmolVLM backbone. "
                             "Base weights are frozen; only adapters + MLP are trained. "
                             "Use with --freeze_action_expert to prevent gripper collapse.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # ModalityDropout flags (p_drop, alpha_min, total_epochs, modalities, corruption_types)
    ModalityDropout.add_args(parser)

    return parser.parse_args()


def apply_freeze_flags(pipeline, args):
    # Block1 components (only if Block1 exists)
    if not getattr(args, "no_block1", False):
        if args.freeze_thermalgen:
            for p in pipeline.block1.thermalgen.parameters():
                p.requires_grad = False
            log.info("Frozen: ThermalGen")

        if args.freeze_imagebind:
            for p in pipeline.block1.imagebind.parameters():
                p.requires_grad = False
            log.info("Frozen: ImageBind")

    # Block2 components
    vlm_with_expert = pipeline.block2.smolvla.policy.base_policy.model.vlm_with_expert

    if args.freeze_4m:
        # Only freeze the pretrained 4M backbone; depth/seg PatchEmbedders stay trainable
        for p in vlm_with_expert.fourm_encoder.model.parameters():
            p.requires_grad = False
        log.info("Frozen: 4M backbone (depth/seg embedders remain trainable)")

    if args.freeze_mlp:
        for p in vlm_with_expert.fourm_to_vlm.parameters():
            p.requires_grad = False
        log.info("Frozen: MLP connector")

    if args.freeze_smolvlm:
        for name, p in vlm_with_expert.named_parameters():
            if not name.startswith("fourm_encoder") and not name.startswith("fourm_to_vlm"):
                p.requires_grad = False
        log.info("Frozen: SmolVLM")

    if args.freeze_action_expert:
        model = pipeline.block2.smolvla.policy.base_policy.model
        for p in model.action_in_proj.parameters():
            p.requires_grad = False
        for p in model.action_out_proj.parameters():
            p.requires_grad = False
        for p in model.action_time_mlp_in.parameters():
            p.requires_grad = False
        for p in model.action_time_mlp_out.parameters():
            p.requires_grad = False
        log.info("Frozen: action expert")

    trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pipeline.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def apply_lora(pipeline, args):
    """Wrap the SmolVLM backbone attention layers with LoRA adapters.

    Only the LoRA delta weights are trainable; original weights stay frozen.
    This lets the backbone adapt to new tasks without catastrophic forgetting.
    """
    from peft import LoraConfig, get_peft_model

    vlm_with_expert = pipeline.block2.smolvla.policy.base_policy.model.vlm_with_expert
    vlm_model = vlm_with_expert.get_vlm_model()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    get_peft_model(vlm_model, lora_config)

    lora_params = sum(p.numel() for p in vlm_model.parameters() if p.requires_grad)
    log.info(f"LoRA applied to SmolVLM backbone: r={args.lora_r}, alpha={args.lora_alpha} | "
             f"trainable LoRA params: {lora_params:,}")


def make_batch(sample, args, tokenizer, device, has_precomputed_modalities=False, cli_args=None):
    no_block1 = getattr(cli_args, "no_block1", False) if cli_args is not None else False
    use_depth = getattr(cli_args, "use_depth", False) if cli_args is not None else False
    use_seg   = getattr(cli_args, "use_seg",   False) if cli_args is not None else False

    rgb     = sample["rgb" if has_precomputed_modalities else args.image_key].to(device)
    state   = sample["observation.state"].to(device)
    actions = sample["action"].to(device)
    B = rgb.shape[0]

    lang = sample.get("language_instruction", [""] * B)
    if isinstance(lang, torch.Tensor):
        lang = [l.decode() if isinstance(l, bytes) else str(l) for l in lang]
    tokenized = tokenizer(
        lang,
        padding="max_length",
        truncation=True,
        max_length=args.max_lang_tokens,
        return_tensors="pt",
    )

    if no_block1:
        # Block1 skipped: pass only the modalities the 4M encoder needs
        inputs = {"rgb": rgb}
        if has_precomputed_modalities:
            if use_depth:
                inputs["depth"] = sample["depth"].to(device)
            if use_seg:
                inputs["seg"] = sample["seg"].to(device)
    elif has_precomputed_modalities:
        # Block1 active: thermal drives ImageBind; depth/seg available as residuals
        inputs = {
            "rgb":     rgb,
            "depth":   sample["depth"].to(device),
            "seg":     sample["seg"].to(device),
            "thermal": sample["thermal"].to(device),
        }
    else:
        # HuggingFace dataset: only RGB, Block1 generates thermal on-the-fly
        inputs = {
            "rgb":   rgb,
            "depth": torch.zeros_like(rgb[:, :1]),
            "seg":   torch.zeros_like(rgb[:, :1]),
        }

    batch = {
        "observation.state": state,
        "observation.language.tokens": tokenized["input_ids"].to(device),
        "observation.language.attention_mask": tokenized["attention_mask"].to(device),
        "action": actions,
        "action_is_pad": sample["action_is_pad"].to(device),
    }

    return inputs, batch


_FOURM_VARIANTS = {
    "B":  ("EPFL-VILAB/4M-21_B",  768),
    "L":  ("EPFL-VILAB/4M-21_L",  1024),
    "XL": ("EPFL-VILAB/4M-21_XL", 1024),
}


def main():
    args = parse_args()

    if args.fourm_model is not None:
        args.fourm_checkpoint, args.fourm_dim = _FOURM_VARIANTS[args.fourm_model]
        log.info(f"4M variant: {args.fourm_model} → {args.fourm_checkpoint} (dim={args.fourm_dim})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    if args.data_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from utils.parquet_dataset import ParquetThermalDataset
        log.info(f"Loading pre-generated parquet dataset from: {args.data_dir}")
        dataset = ParquetThermalDataset(args.data_dir, chunk_size=args.chunk_size, task_ids=args.task_ids)
        has_precomputed_modalities = True
    else:
        from lerobot.datasets import LeRobotDataset
        log.info(f"Loading HuggingFace dataset: {args.dataset} (thermal generated on-the-fly)")
        dataset = LeRobotDataset(args.dataset)
        has_precomputed_modalities = False

    log.info(f"Dataset size: {len(dataset)} samples")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    from src.pipeline.full_pipeline import VLAPipeline
    log.info("Building VLAPipeline ...")
    pipeline = VLAPipeline(
        smolvla_checkpoint=args.smolvla_checkpoint,
        fourm_checkpoint=args.fourm_checkpoint,
        fourm_dim=args.fourm_dim,
        use_4m=True,
        freeze_4m=False,
        freeze_mlp=False,
        p_drop=args.p_drop,
        alpha_min=args.alpha_min,
        total_epochs=args.total_epochs,
        modalities=args.modalities,
        corruption_types=args.corruption_types,
        device=device,
        skip_block1=args.no_block1,
        use_depth=args.use_depth,
        use_seg=args.use_seg,
        from_scratch=args.from_scratch,
    )
    if args.resume_checkpoint:
        log.info(f"Resuming from checkpoint: {args.resume_checkpoint}")
        state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        state.pop("_siglip_proj.weight", None)
        pipeline.load_state_dict(state)
        log.info("Checkpoint loaded ✅")

    pipeline.train()

    # Apply freeze flags first so optimizer only tracks trainable params
    apply_freeze_flags(pipeline, args)

    # Cast frozen params to bf16 to free ~14 GB on the V100 (7B frozen params * 2 bytes saved).
    # Safe because frozen params never receive gradient updates.
    if args.bf16:
        n_cast = sum(1 for p in pipeline.parameters() if not p.requires_grad)
        for p in pipeline.parameters():
            if not p.requires_grad:
                p.data = p.data.to(torch.bfloat16)
        log.info(f"Cast {n_cast} frozen param tensors to bf16 "
                 f"(freed ~{sum(p.numel() for p in pipeline.parameters() if not p.requires_grad) * 2 / 2**30:.1f} GB)")

    if args.lora:
        apply_lora(pipeline, args)

    vlm_with_expert = pipeline.block2.smolvla.policy.base_policy.model.vlm_with_expert
    base_model      = pipeline.block2.smolvla.policy.base_policy.model

    # MLP group: connector + depth/seg patch embedders (all at high lr for fast alignment)
    mlp_param_ids = {id(p) for p in vlm_with_expert.fourm_to_vlm.parameters()}
    if hasattr(vlm_with_expert.fourm_encoder, "depth_embed"):
        mlp_param_ids.update(id(p) for p in vlm_with_expert.fourm_encoder.depth_embed.parameters())
    if hasattr(vlm_with_expert.fourm_encoder, "seg_embed"):
        mlp_param_ids.update(id(p) for p in vlm_with_expert.fourm_encoder.seg_embed.parameters())

    action_head_param_ids = set()
    for component_name in ["action_in_proj", "action_out_proj",
                            "action_time_mlp_in", "action_time_mlp_out"]:
        component = getattr(base_model, component_name, None)
        if component is not None:
            action_head_param_ids.update(id(p) for p in component.parameters())

    mlp_trainable         = [p for p in pipeline.parameters()
                              if p.requires_grad and id(p) in mlp_param_ids]
    action_head_trainable = [p for p in pipeline.parameters()
                              if p.requires_grad and id(p) in action_head_param_ids]
    other_params          = [p for p in pipeline.parameters()
                              if p.requires_grad
                              and id(p) not in mlp_param_ids
                              and id(p) not in action_head_param_ids]

    log.info(f"Optimizer groups: "
             f"mlp={sum(p.numel() for p in mlp_trainable):,} params @ lr={args.lr_mlp}  |  "
             f"action_head={sum(p.numel() for p in action_head_trainable):,} params @ lr={args.lr_action}  |  "
             f"other={sum(p.numel() for p in other_params):,} params @ lr={args.lr}")

    optimizer = torch.optim.AdamW([
        {'params': other_params,          'lr': args.lr},
        {'params': mlp_trainable,         'lr': args.lr_mlp},
        {'params': action_head_trainable, 'lr': args.lr_action},
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        log.info(f"WandB run: {wandb.run.url}")

    log.info("Starting training ...")
    step = 0
    running_loss = 0.0
    running_extras = {}  # accumulate distill metrics

    while step < args.steps:
        for sample in dataloader:
            if step >= args.steps:
                break

            inputs, batch = make_batch(sample, args, tokenizer, device,
                                       has_precomputed_modalities, args)

            autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                            if args.bf16 and device == "cuda"
                            else contextlib.nullcontext())

            with autocast_ctx:
                if args.distill:
                    loss_dict = pipeline.compute_distill_loss(inputs, epoch=step)
                elif args.joint:
                    loss_dict = pipeline.compute_joint_loss(
                        inputs, batch, epoch=step, lambda_distill=args.lambda_distill)
                else:
                    loss_dict = pipeline.compute_loss(inputs, batch, epoch=step)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            for k, v in loss_dict.items():
                if k != "loss":
                    running_extras[k] = running_extras.get(k, 0.0) + float(v)
            step += 1

            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                avg_extras = {k: v / args.log_every for k, v in running_extras.items()}

                if args.distill:
                    log.info(f"step={step}/{args.steps}  distill_loss={avg:.4f}"
                             f"  cos_sim={avg_extras.get('mean_cosine_similarity', 0):.4f}"
                             f"  mlp_norm={avg_extras.get('mlp_norm', 0):.1f}"
                             f"  siglip_norm={avg_extras.get('siglip_norm', 0):.1f}")
                elif args.joint:
                    log.info(f"step={step}/{args.steps}  total_loss={avg:.4f}"
                             f"  action={avg_extras.get('action_loss', 0):.4f}"
                             f"  distill={avg_extras.get('distill_loss', 0):.4f}"
                             f"  mlp_norm={avg_extras.get('mlp_norm', 0):.1f}")
                else:
                    log.info(f"step={step}/{args.steps}  loss={avg:.4f}")

                if args.wandb:
                    import wandb
                    log_dict = {"loss": avg, "step": step, **avg_extras}
                    wandb.log(log_dict)

                running_loss = 0.0
                running_extras = {}

            if step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"pipeline_step{step}.pt")
                torch.save(pipeline.state_dict(), ckpt)
                log.info(f"Saved checkpoint: {ckpt}")

    ckpt = os.path.join(args.output_dir, "pipeline_final.pt")
    torch.save(pipeline.state_dict(), ckpt)
    log.info(f"Training done. Final checkpoint: {ckpt}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
