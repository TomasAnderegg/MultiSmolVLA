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

    # Dataset — use either HF dataset OR local parquet (mutually exclusive)
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace LeRobotDataset repo id (e.g. lerobot/libero_10)")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Path to parquet_thermal dataset dir (e.g. /scratch/libero_thermal/data/train). "
                             "Uses ParquetThermalDataset instead of LeRobotDataset.")
    parser.add_argument("--image_key", type=str, default="observation.images.top",
                        help="Dataset key to use as RGB input to 4M (only for --dataset mode)")

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

    # Distillation mode
    parser.add_argument("--distill", action="store_true",
                        help="Use joint loss: L_action + lambda_distill * L_cosine(4M+MLP, SigLIP+connector). "
                             "Correct target: full embed_image() path, inference normalization.")
    parser.add_argument("--distill_only", action="store_true",
                        help="Distillation loss only — no action loss. "
                             "Only needs RGB images (parquet dataset works). "
                             "Use this for stage 1 before joint training.")
    parser.add_argument("--lambda_distill", type=float, default=0.1,
                        help="Weight of distillation loss in joint training (default 0.1)")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Initialize from an existing checkpoint before training "
                             "(e.g. a previous stage1 run). Keys not found are ignored.")

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


def make_batch_parquet(sample, tokenizer, max_lang_tokens, device):
    """Convert a ParquetThermalDataset sample into Block2 inputs + batch dict.

    ParquetThermalDataset returns: {"rgb": (3,H,W), "depth": (1,H,W), "action": (7,), ...}
    DataLoader batches these to (B, ...).
    """
    B = sample["rgb"].shape[0]
    rgb   = sample["rgb"].to(device)      # (B, 3, H, W)  float [0,1]
    depth = sample.get("depth", torch.zeros(B, 1, rgb.shape[-2], rgb.shape[-1])).to(device)

    # Dummy state and action — parquet has no robot state or action labels
    # Use zero state; actions come from the dataset if available, else zeros
    state   = sample.get("state",  torch.zeros(B, 7, device=device))
    actions = sample.get("action", torch.zeros(B, 50, 7, device=device))
    if isinstance(state,   torch.Tensor): state   = state.to(device)
    if isinstance(actions, torch.Tensor): actions = actions.to(device)
    # Pad action chunk to 50 steps if needed
    if actions.dim() == 2:                  # (B, action_dim)
        actions = actions.unsqueeze(1).expand(-1, 50, -1)
    elif actions.shape[1] < 50:
        pad = torch.zeros(B, 50 - actions.shape[1], actions.shape[2], device=device)
        actions = torch.cat([actions, pad], dim=1)

    lang = ["pick and place the object"] * B   # placeholder — parquet has no lang labels
    tokenized = tokenizer(lang, padding="max_length", truncation=True,
                          max_length=max_lang_tokens, return_tensors="pt")

    inputs = {
        "rgb":     rgb,
        "depth":   depth,
        "seg":     torch.zeros(B, 1, rgb.shape[-2], rgb.shape[-1], device=device),
        "thermal": torch.zeros(B, 1024, device=device),
    }
    batch = {
        "observation.state":                   state,
        "observation.language.tokens":         tokenized["input_ids"].to(device),
        "observation.language.attention_mask": tokenized["attention_mask"].to(device),
        "action": actions,
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
    elif args.parquet is not None:
        from utils.parquet_dataset import ParquetThermalDataset
        log.info(f"Loading parquet dataset from {args.parquet}")
        dataset = ParquetThermalDataset(args.parquet, chunk_size=1)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        log.info(f"Parquet dataset: {len(dataset)} frames")
    elif args.dataset is not None:
        from lerobot.datasets import LeRobotDataset
        log.info(f"Loading HF dataset: {args.dataset}")
        dataset = LeRobotDataset(args.dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        raise ValueError("Provide --dataset, --parquet, or --dummy")

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

    os.makedirs(args.output_dir, exist_ok=True)
    log.info("Starting training ...")
    step = 0
    running_loss = 0.0

    if args.distill_only:
        # Pure distillation — no action loss, no batch needed
        from src.pipeline.full_pipeline import VLAPipeline
        log.info("Building VLAPipeline for distillation-only training ...")
        pipeline = VLAPipeline(
            smolvla_checkpoint=args.smolvla_checkpoint,
            fourm_checkpoint=cfg.fourm_checkpoint,
            fourm_dim=cfg.fourm_dim,
            use_4m=True, freeze_4m=False, freeze_mlp=False,
            device=device, skip_block1=True,
        )
        pipeline.train()
        block2 = pipeline.block2
        apply_freeze_flags(block2, cfg)

        if args.init_checkpoint is not None:
            log.info(f"Loading init checkpoint: {args.init_checkpoint}")
            state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            missing, unexpected = pipeline.load_state_dict(state, strict=False)
            log.info(f"Init checkpoint loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        optimizer = torch.optim.AdamW(
            [p for p in pipeline.parameters() if p.requires_grad], lr=args.lr)

        def _train_step(inputs, batch):
            nonlocal running_loss, step
            loss_dict = pipeline.compute_distill_loss(inputs, epoch=0)
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1
            if step % args.log_every == 0:
                avg  = running_loss / args.log_every
                cos  = loss_dict.get("mean_cosine_similarity", 1 - avg)
                n_sig = loss_dict.get("n_siglip_tokens", "?")
                n_4m  = loss_dict.get("n_fourm_tokens",  "?")
                sig_n = loss_dict.get("siglip_norm", float("nan"))
                mlp_n = loss_dict.get("mlp_norm",    float("nan"))
                log.info(f"step={step}/{args.steps}  distill={avg:.4f}  "
                         f"cos_sim={cos:.4f}  "
                         f"norm_sig={sig_n:.1f}  norm_mlp={mlp_n:.1f}  "
                         f"tokens={n_sig}sig/{n_4m}4m")
                running_loss = 0.0
            if step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"block2_step{step:06d}.pt")
                torch.save(pipeline.state_dict(), ckpt)
                log.info(f"Saved → {ckpt}")

    elif args.distill:
        # Joint mode: need VLAPipeline to access compute_joint_loss
        from src.pipeline.full_pipeline import VLAPipeline
        log.info("Building VLAPipeline (skip_block1=True) for joint distill+action training ...")
        pipeline = VLAPipeline(
            smolvla_checkpoint=args.smolvla_checkpoint,
            fourm_checkpoint=cfg.fourm_checkpoint,
            fourm_dim=cfg.fourm_dim,
            use_4m=True,
            freeze_4m=False,
            freeze_mlp=False,
            device=device,
            skip_block1=True,
        )
        pipeline.train()
        block2 = pipeline.block2
        apply_freeze_flags(block2, cfg)

        if args.init_checkpoint is not None:
            log.info(f"Loading init checkpoint: {args.init_checkpoint}")
            state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            missing, unexpected = pipeline.load_state_dict(state, strict=False)
            log.info(f"Init checkpoint loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        optimizer = torch.optim.AdamW(
            [p for p in pipeline.parameters() if p.requires_grad], lr=args.lr)

        def _train_step(inputs, batch):
            nonlocal running_loss, step
            loss_dict = pipeline.compute_joint_loss(
                inputs, batch, epoch=0, lambda_distill=args.lambda_distill)
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1
            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                cos = loss_dict.get("distill_loss", 0.0)
                act = loss_dict.get("action_loss", 0.0)
                log.info(f"step={step}/{args.steps}  loss={avg:.4f}  "
                         f"action={act:.4f}  distill={cos:.4f}")
                running_loss = 0.0
            if step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"block2_step{step:06d}.pt")
                torch.save(pipeline.state_dict(), ckpt)
                log.info(f"Saved → {ckpt}")

    else:
        # Action-only mode (original behaviour)
        from src.pipeline.block2 import Block2
        log.info("Building Block2 (action loss only) ...")
        block2 = Block2(
            fourm_checkpoint=cfg.fourm_checkpoint,
            smolvla_checkpoint=args.smolvla_checkpoint,
            use_4m=True,
            freeze_4m=False,
            freeze_mlp=False,
            fourm_dim=cfg.fourm_dim,
            device=device,
        )
        block2.train()
        apply_freeze_flags(block2, cfg)

        if args.init_checkpoint is not None:
            log.info(f"Loading init checkpoint: {args.init_checkpoint}")
            state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            missing, unexpected = block2.load_state_dict(state, strict=False)
            log.info(f"Init checkpoint loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        optimizer = torch.optim.AdamW(
            [p for p in block2.parameters() if p.requires_grad], lr=args.lr)

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
                ckpt = os.path.join(args.output_dir, f"block2_step{step:06d}.pt")
                torch.save(block2.state_dict(), ckpt)
                log.info(f"Saved → {ckpt}")

    if args.dummy:
        while step < args.steps:
            inputs, batch = make_dummy_batch(args, tokenizer, device)
            _train_step(inputs, batch)
    else:
        while step < args.steps:
            for sample in dataloader:
                if step >= args.steps:
                    break
                if args.parquet is not None:
                    inputs, batch = make_batch_parquet(
                        sample, tokenizer, args.max_lang_tokens, device)
                else:
                    inputs, batch = make_batch(
                        sample, args.image_key, tokenizer, args.max_lang_tokens, device)
                _train_step(inputs, batch)

    # Final save
    model_to_save = pipeline if (args.distill or args.distill_only) else block2
    ckpt = os.path.join(args.output_dir, "block2_final.pt")
    torch.save(model_to_save.state_dict(), ckpt)
    log.info(f"Training done. Final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
