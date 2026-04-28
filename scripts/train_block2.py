#!/usr/bin/env python
"""
Train Block 2 (4M encoder + MLP connector + SmolVLA) on a LIBERO dataset.

Usage examples:

  # Stage 1: train only the MLP connector (everything else frozen)
  python scripts/train_block2.py \
      --freeze_4m --freeze_smolvlm --freeze_action_expert

  # Stage 2: train MLP + action expert + SmolVLM (4M frozen)
  python scripts/train_block2.py \
      --freeze_4m

  # Train everything
  python scripts/train_block2.py

  # Custom dataset / image key
  python scripts/train_block2.py \
      --dataset lerobot/libero_object_no_noops \
      --image_key observation.images.top \
      --freeze_4m --freeze_smolvlm
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
    parser = argparse.ArgumentParser(description="Train Block 2 on LIBERO")

    # Dataset
    parser.add_argument("--dataset", type=str, default="lerobot/libero_spatial_no_noops",
                        help="HuggingFace dataset repo id")
    parser.add_argument("--image_key", type=str, default="observation.images.top",
                        help="Dataset key to use as RGB input to 4M")

    # Checkpoints
    parser.add_argument("--smolvla_checkpoint", type=str, default="lerobot/smolvla_libero")
    parser.add_argument("--output_dir", type=str, default="checkpoints/block2")

    # Model config — freeze flags (passed into LocalSmolVLAConfig)
    parser.add_argument("--freeze_4m", action="store_true", help="Freeze the 4M encoder")
    parser.add_argument("--freeze_mlp", action="store_true", help="Freeze the MLP connector")
    parser.add_argument("--freeze_smolvlm", action="store_true", help="Freeze the SmolVLM language model")
    parser.add_argument("--freeze_action_expert", action="store_true", help="Freeze the action expert")
    parser.add_argument("--fourm_model", type=str, default=None, choices=["B", "L", "XL"],
                        help="4M-21 model variant shortcut (B=768d, L/XL=1024d). "
                             "Overrides --fourm_checkpoint and --fourm_dim when set.")
    parser.add_argument("--fourm_checkpoint", type=str, default="EPFL-VILAB/4M-21_XL")
    parser.add_argument("--fourm_dim", type=int, default=1024)

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_lang_tokens", type=int, default=48)

    # Dummy mode — no dataset needed
    parser.add_argument("--dummy", action="store_true",
                        help="Use random inputs instead of a real dataset (for smoke-testing)")
    parser.add_argument("--dummy_state_dim", type=int, default=7)
    parser.add_argument("--dummy_action_dim", type=int, default=7)
    parser.add_argument("--dummy_action_steps", type=int, default=50,
                        help="Action chunk size (must match SmolVLA chunk_size=50)")

    return parser.parse_args()


def apply_freeze_flags(block2, cfg):
    """Freeze/unfreeze components based on LocalSmolVLAConfig flags."""
    vlm_with_expert = block2.smolvla.policy.base_policy.model.vlm_with_expert

    if cfg.freeze_4m:
        for p in vlm_with_expert.fourm_encoder.parameters():
            p.requires_grad = False
        log.info("Frozen: 4M encoder")

    if cfg.freeze_mlp:
        for p in vlm_with_expert.fourm_to_vlm.parameters():
            p.requires_grad = False
        log.info("Frozen: MLP connector")

    if cfg.freeze_smolvlm:
        for name, p in vlm_with_expert.named_parameters():
            if not name.startswith("fourm_encoder") and not name.startswith("fourm_to_vlm"):
                p.requires_grad = False
        log.info("Frozen: SmolVLM")

    if cfg.freeze_action_expert:
        model = block2.smolvla.policy.base_policy.model
        for p in model.action_in_proj.parameters():
            p.requires_grad = False
        for p in model.action_out_proj.parameters():
            p.requires_grad = False
        for p in model.action_time_mlp_in.parameters():
            p.requires_grad = False
        for p in model.action_time_mlp_out.parameters():
            p.requires_grad = False
        log.info("Frozen: action expert")

    trainable = sum(p.numel() for p in block2.parameters() if p.requires_grad)
    total = sum(p.numel() for p in block2.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def make_dummy_batch(args, tokenizer, device):
    B = args.batch_size
    tokenized = tokenizer(
        ["pick up the red block"] * B,
        padding="max_length",
        truncation=True,
        max_length=args.max_lang_tokens,
        return_tensors="pt",
    )
    inputs = {
        "rgb":     torch.randn(B, 3, 224, 224, device=device),
        "depth":   torch.zeros(B, 1, 224, 224, device=device),
        "seg":     torch.zeros(B, 1, 224, 224, device=device),
        "thermal": torch.zeros(B, 1024, device=device),
    }
    batch = {
        "observation.state":                    torch.randn(B, args.dummy_state_dim, device=device),
        "observation.language.tokens":          tokenized["input_ids"].to(device),
        "observation.language.attention_mask":  tokenized["attention_mask"].to(device),
        "action": torch.randn(B, args.dummy_action_steps, args.dummy_action_dim, device=device),
    }
    return inputs, batch


def make_batch(sample, image_key, tokenizer, max_lang_tokens, device):
    """Convert a LeRobotDataset sample into Block2 inputs + batch dict."""
    rgb = sample[image_key].to(device)          # (B, 3, H, W)
    state = sample["observation.state"].to(device)
    actions = sample["action"].to(device)

    # Tokenize language instruction
    lang = sample.get("language_instruction", [""] * rgb.shape[0])
    if isinstance(lang, torch.Tensor):
        lang = [l.decode() if isinstance(l, bytes) else str(l) for l in lang]
    tokenized = tokenizer(
        lang,
        padding="max_length",
        truncation=True,
        max_length=max_lang_tokens,
        return_tensors="pt",
    )

    inputs = {
        "rgb":     rgb,
        "depth":   torch.zeros_like(rgb[:, :1]),   # not available in LIBERO
        "seg":     torch.zeros_like(rgb[:, :1]),   # not available in LIBERO
        "thermal": torch.zeros(rgb.shape[0], 1024, device=device),  # not available in LIBERO
    }

    batch = {
        "observation.state": state,
        "observation.language.tokens": tokenized["input_ids"].to(device),
        "observation.language.attention_mask": tokenized["attention_mask"].to(device),
        "action": actions,
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    if args.dummy:
        log.info("Dummy mode: using random inputs (no dataset)")
        dataloader = None
    else:
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

    # Build config from CLI flags
    from src.pipeline.smolvla.configuration_smolvla import LocalSmolVLAConfig
    cfg = LocalSmolVLAConfig(
        fourm_checkpoint=args.fourm_checkpoint,
        fourm_dim=args.fourm_dim,
        freeze_4m=args.freeze_4m,
        freeze_mlp=args.freeze_mlp,
        freeze_smolvlm=args.freeze_smolvlm,
        freeze_action_expert=args.freeze_action_expert,
    )
    log.info(f"Config: freeze_4m={cfg.freeze_4m}, freeze_mlp={cfg.freeze_mlp}, "
             f"freeze_smolvlm={cfg.freeze_smolvlm}, freeze_action_expert={cfg.freeze_action_expert}")

    # Build Block2 (freezing handled manually after load so all weights are initialized first)
    from src.pipeline.block2 import Block2
    log.info("Building Block2 ...")
    block2 = Block2(
        fourm_checkpoint=cfg.fourm_checkpoint,
        smolvla_checkpoint=args.smolvla_checkpoint,
        use_4m=cfg.use_4m,
        freeze_4m=False,
        freeze_mlp=False,
        fourm_dim=cfg.fourm_dim,
        device=device,
    )
    block2.train()

    # Apply freeze flags from config
    apply_freeze_flags(block2, cfg)

    # Optimizer — only trainable params
    optimizer = torch.optim.AdamW(
        [p for p in block2.parameters() if p.requires_grad],
        lr=args.lr,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Starting training ...")
    step = 0
    running_loss = 0.0

    def _train_step(inputs, batch):
        nonlocal running_loss, step
        loss_dict = block2.compute_loss(inputs, batch)
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
            ckpt = os.path.join(args.output_dir, f"block2_step{step}.pt")
            torch.save(block2.state_dict(), ckpt)
            log.info(f"Saved checkpoint: {ckpt}")

    if args.dummy:
        while step < args.steps:
            inputs, batch = make_dummy_batch(args, tokenizer, device)
            _train_step(inputs, batch)
    else:
        while step < args.steps:
            for sample in dataloader:
                if step >= args.steps:
                    break
                inputs, batch = make_batch(
                    sample, args.image_key, tokenizer, args.max_lang_tokens, device
                )
                _train_step(inputs, batch)

    # Final save
    ckpt = os.path.join(args.output_dir, "block2_final.pt")
    torch.save(block2.state_dict(), ckpt)
    log.info(f"Training done. Final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
