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
import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "lerobot", "src"))

from src.pipeline.modality_dropout import ModalityDropout

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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

    # Freeze flags — Block1
    parser.add_argument("--freeze_thermalgen", action="store_true", help="Freeze ThermalGen (RGB→thermal generator)")
    parser.add_argument("--freeze_imagebind", action="store_true", help="Freeze ImageBind thermal encoder")
    # Freeze flags — Block2
    parser.add_argument("--freeze_4m", action="store_true", help="Freeze the 4M encoder")
    parser.add_argument("--freeze_mlp", action="store_true", help="Freeze the MLP connector")
    parser.add_argument("--freeze_smolvlm", action="store_true", help="Freeze the SmolVLM language model")
    parser.add_argument("--freeze_action_expert", action="store_true", help="Freeze the action expert")

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_mlp", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_lang_tokens", type=int, default=48)
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Number of future action steps per training sample (must match SmolVLA chunk_size)")

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="multismolvla")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # ModalityDropout flags (p_drop, alpha_min, total_epochs, modalities, corruption_types)
    ModalityDropout.add_args(parser)

    return parser.parse_args()


def apply_freeze_flags(pipeline, args):
    # Block1 components
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
        for p in vlm_with_expert.fourm_encoder.parameters():
            p.requires_grad = False
        log.info("Frozen: 4M encoder")

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


def make_batch(sample, args, tokenizer, device, has_precomputed_modalities=False):
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

    if has_precomputed_modalities:
        # All 4 modalities available from pre-generated parquet dataset
        inputs = {
            "rgb":     rgb,
            "depth":   sample["depth"].to(device),    # (B, 1, 224, 224)
            "seg":     sample["seg"].to(device),      # (B, 1, 224, 224)
            "thermal": sample["thermal"].to(device),  # (B, 3, 224, 224) — skips ThermalGen in Block1
        }
    else:
        # HuggingFace dataset: only RGB available, Block1 generates thermal on-the-fly
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
        dataset = ParquetThermalDataset(args.data_dir, chunk_size=args.chunk_size)
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
    )
    pipeline.train()

    # Apply freeze flags first so optimizer only tracks trainable params
    apply_freeze_flags(pipeline, args)

    vlm_with_expert = pipeline.block2.smolvla.policy.base_policy.model.vlm_with_expert
    mlp_params = set(vlm_with_expert.fourm_to_vlm.parameters())

    mlp_trainable = [p for p in mlp_params if p.requires_grad]
    other_params   = [p for p in pipeline.parameters()
                      if p.requires_grad and p not in mlp_params]

    optimizer = torch.optim.AdamW([
        {'params': other_params,  'lr': args.lr},
        {'params': mlp_trainable, 'lr': args.lr_mlp},
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        log.info(f"WandB run: {wandb.run.url}")

    log.info("Starting training ...")
    step = 0
    running_loss = 0.0

    while step < args.steps:
        for sample in dataloader:
            if step >= args.steps:
                break

            inputs, batch = make_batch(sample, args, tokenizer, device, has_precomputed_modalities)

            loss_dict = pipeline.compute_loss(inputs, batch, epoch=step)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                log.info(f"step={step}/{args.steps}  loss={avg:.4f}")
                if args.wandb:
                    import wandb
                    wandb.log({"loss": avg, "step": step})
                running_loss = 0.0

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
