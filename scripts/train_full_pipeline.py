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

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train full VLA pipeline on LIBERO")

    # Dataset
    parser.add_argument("--dataset", type=str, default="lerobot/libero_spatial_no_noops")
    parser.add_argument("--image_key", type=str, default="observation.images.top",
                        help="Dataset key to use as RGB input")

    # Checkpoints
    parser.add_argument("--smolvla_checkpoint", type=str, default="lerobot/smolvla_libero")
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_lang_tokens", type=int, default=48)

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


def make_batch(sample, args, tokenizer, device):
    rgb = sample[args.image_key].to(device)
    state = sample["observation.state"].to(device)
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

    # Block1 generates thermal from rgb — depth/seg are zeros (not in LIBERO)
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
    }

    return inputs, batch


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    from lerobot.datasets import LeRobotDataset
    log.info(f"Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(args.dataset)
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
        device=device,
    )
    pipeline.train()

    apply_freeze_flags(pipeline, args)

    optimizer = torch.optim.AdamW(
        [p for p in pipeline.parameters() if p.requires_grad],
        lr=args.lr,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Starting training ...")
    step = 0
    running_loss = 0.0

    while step < args.steps:
        for sample in dataloader:
            if step >= args.steps:
                break

            inputs, batch = make_batch(sample, args, tokenizer, device)

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
                running_loss = 0.0

            if step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"pipeline_step{step}.pt")
                torch.save(pipeline.state_dict(), ckpt)
                log.info(f"Saved checkpoint: {ckpt}")

    ckpt = os.path.join(args.output_dir, "pipeline_final.pt")
    torch.save(pipeline.state_dict(), ckpt)
    log.info(f"Training done. Final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
