#!/usr/bin/env python3
"""
Fine-tune 4M-21 on LIBERO parquet data (RGB + depth + seg) with LoRA.

Objective: any-to-any multimodal masked modeling on 3 modalities.
- tok_rgb@224   : RGB   → DiVAE tokenizer   (8192 vocab)
- tok_depth@224 : depth → depth VQ tokenizer (8192 vocab)
- tok_semseg@224: seg   → semseg VQ tokenizer (4096 vocab)

Only LoRA weights in the 4M transformer are trained.
All tokenizers (DiVAE, depth_tok, semseg_tok) stay frozen.
"""

import argparse
import os
import sys
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF

# ── fourm path ────────────────────────────────────────────────────────────────
_repo_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_fourm_root = os.path.join(_repo_root, "third_party", "ml-4m")
if _fourm_root not in sys.path:
    sys.path.insert(0, _fourm_root)

from fourm.models.fm import FM
from fourm.vq.vqvae import DiVAE, VQ
from fourm.models.lora_utils import inject_trainable_LoRA, unfreeze_all_LoRA_layers, get_LoRA_module_names

sys.path.insert(0, _repo_root)
from utils.parquet_dataset import ParquetThermalDataset

NUM_PATCHES = 196
PATCH_GRID  = 14
MODALITIES  = ["tok_rgb@224", "tok_depth@224", "tok_semseg@224"]


# ─────────────────────────────────────────────────────────────────────────────
# Masking helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_batch_mod_dict(rgb_ids, dep_ids, seg_ids, enc_idxs, dec_idx, device):
    """
    Build a batched mod_dict for fm.forward().

    enc_idxs : list of indices into MODALITIES that go to encoder (1 or 2)
    dec_idx  : index into MODALITIES that goes to decoder (the one to predict)
    The remaining modality (if 1→1 mode) is ignored (not in enc, not in dec).

    Returns (mod_dict, num_enc_tokens, num_dec_tokens).
    """
    B = rgb_ids.shape[0]
    all_ids = [rgb_ids, dep_ids, seg_ids]

    num_enc = len(enc_idxs) * NUM_PATCHES
    num_dec = NUM_PATCHES

    mod_dict = {}
    for i, (mod, ids) in enumerate(zip(MODALITIES, all_ids)):
        ids_2d = ids.reshape(B, PATCH_GRID, PATCH_GRID).to(device)

        if i in enc_idxs:
            # Fully visible to encoder, not a decoder target
            input_mask            = torch.zeros(B, NUM_PATCHES, dtype=torch.bool,  device=device)
            target_mask           = torch.ones (B, NUM_PATCHES, dtype=torch.bool,  device=device)
            # decoder_attention_mask not meaningful for encoder tokens, but must exist
            decoder_attention_mask = torch.zeros(B, NUM_PATCHES, dtype=torch.int64, device=device)

        elif i == dec_idx:
            # Not visible to encoder; all 196 tokens are decoder targets
            input_mask             = torch.ones (B, NUM_PATCHES, dtype=torch.bool,  device=device)
            target_mask            = torch.zeros(B, NUM_PATCHES, dtype=torch.bool,  device=device)
            # [N, 0, 0, ...] → cumsum [N,N,N,...] → all tokens attend to all (MaskGIT-style)
            decoder_attention_mask = torch.zeros(B, NUM_PATCHES, dtype=torch.int64, device=device)
            decoder_attention_mask[:, 0] = NUM_PATCHES

        else:
            # Ignored modality: masked from both encoder and decoder
            input_mask             = torch.ones(B, NUM_PATCHES, dtype=torch.bool,  device=device)
            target_mask            = torch.ones(B, NUM_PATCHES, dtype=torch.bool,  device=device)
            decoder_attention_mask = torch.zeros(B, NUM_PATCHES, dtype=torch.int64, device=device)

        mod_dict[mod] = {
            "tensor":                  ids_2d,
            "input_mask":              input_mask,
            "target_mask":             target_mask,
            "decoder_attention_mask":  decoder_attention_mask,
        }

    return mod_dict, num_enc, num_dec


def sample_enc_dec(mode_2to1: bool, rgb_only_target: bool = False):
    """Return (enc_idxs, dec_idx) for a random any→any masking pattern.

    rgb_only_target=True: decoder always predicts RGB (idx=0).
      - 2→1 mode: depth+seg → rgb
      - 1→1 mode: depth→rgb or seg→rgb (random)
    """
    if rgb_only_target:
        dec_idx = 0  # always predict RGB
        if mode_2to1:
            enc_idxs = [1, 2]  # depth + seg → rgb
        else:
            enc_idxs = [random.choice([1, 2])]  # depth→rgb or seg→rgb
        return enc_idxs, dec_idx

    if mode_2to1:
        dec_idx  = random.randint(0, 2)
        enc_idxs = [i for i in range(3) if i != dec_idx]
    else:  # 1→1
        enc_idx  = random.randint(0, 2)
        dec_idx  = random.choice([i for i in range(3) if i != enc_idx])
        enc_idxs = [enc_idx]
    return enc_idxs, dec_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning of 4M-21 on LIBERO")
    p.add_argument("--data_dir",        required=True,  help="Path to parquet_thermal train directory")
    p.add_argument("--output_dir",      default="finetune_4m_libero_lora")
    p.add_argument("--fourm_model",     default="EPFL-VILAB/4M-21_XL")
    p.add_argument("--divae_ckpt",      default="EPFL-VILAB/4M_tokenizers_rgb_16k_224-448")
    p.add_argument("--depth_tok_ckpt",  default="EPFL-VILAB/4M_tokenizers_depth_8k_224-448")
    p.add_argument("--seg_tok_ckpt",    default="EPFL-VILAB/4M_tokenizers_semseg_4k_224-448")
    p.add_argument("--lora_rank",       type=int,   default=8)
    p.add_argument("--lora_scale",      type=float, default=1.0)
    p.add_argument("--lora_target",     default="attention",
                   help="attention | mlp | all  (modules to apply LoRA to)")
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--grad_accum",      type=int,   default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--epochs",          type=int,   default=5)
    p.add_argument("--warmup_steps",    type=int,   default=200)
    p.add_argument("--save_every",      type=int,   default=500,  help="Save checkpoint every N steps")
    p.add_argument("--log_every",       type=int,   default=20,   help="Log to wandb every N steps")
    p.add_argument("--max_steps",       type=int,   default=-1,   help="Stop after N steps (-1 = full)")
    p.add_argument("--resume",          type=str,   default=None, help="Resume from LoRA checkpoint (.pt) — loads LoRA weights, resets optimizer")
    p.add_argument("--mode_2to1_prob",  type=float, default=0.6,
                   help="Probability of using 2→1 masking (vs 1→1).")
    p.add_argument("--rgb_only_target", action="store_true",
                   help="Decoder always predicts RGB (depth+seg→rgb or depth/seg→rgb). "
                        "Use this to focus training on the modality that matters for SmolVLA.")
    p.add_argument("--wandb_project",   default="4m-libero-finetune")
    p.add_argument("--wandb_run_name",  default=None)
    p.add_argument("--wandb_entity",    default=None)
    p.add_argument("--hf_repo",         default=None,
                   help="HuggingFace repo to upload LoRA weights after training (e.g. username/4m_fine_tune_libero)")
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────────
    try:
        import wandb
        run_name = args.wandb_run_name or f"lora_r{args.lora_rank}_{args.lora_target}_lr{args.lr}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
        use_wandb = True
        print(f"[wandb] Run: {run_name}", flush=True)
    except Exception as e:
        print(f"[wandb] Not available or failed to init: {e}", flush=True)
        use_wandb = False

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"[data] Loading parquet dataset from {args.data_dir} ...", flush=True)
    dataset = ParquetThermalDataset(args.data_dir, chunk_size=1)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"[data] {len(dataset)} samples, {len(loader)} batches/epoch", flush=True)

    # ── Load tokenizers (frozen) ───────────────────────────────────────────────
    print("[model] Loading DiVAE (RGB tokenizer) ...", flush=True)
    try:
        divae = DiVAE.from_pretrained(args.divae_ckpt, local_files_only=True)
    except Exception:
        divae = DiVAE.from_pretrained(args.divae_ckpt)
    divae.to(device).eval()
    for p in divae.parameters():
        p.requires_grad = False

    print("[model] Loading depth tokenizer ...", flush=True)
    try:
        depth_tok = VQ.from_pretrained(args.depth_tok_ckpt, local_files_only=True)
    except Exception:
        try:
            depth_tok = VQ.from_pretrained(args.depth_tok_ckpt)
        except Exception:
            depth_tok = DiVAE.from_pretrained(args.depth_tok_ckpt)
    depth_tok.to(device).eval()
    for p in depth_tok.parameters():
        p.requires_grad = False

    print("[model] Loading seg tokenizer ...", flush=True)
    try:
        seg_tok = VQ.from_pretrained(args.seg_tok_ckpt, local_files_only=True)
    except Exception:
        seg_tok = VQ.from_pretrained(args.seg_tok_ckpt)
    seg_tok.to(device).eval()
    for p in seg_tok.parameters():
        p.requires_grad = False

    # ── Load 4M and inject LoRA ───────────────────────────────────────────────
    print(f"[model] Loading 4M from {args.fourm_model} ...", flush=True)
    try:
        fm = FM.from_pretrained(args.fourm_model, local_files_only=True)
    except Exception:
        fm = FM.from_pretrained(args.fourm_model)
    fm.to(device)

    # Freeze all 4M weights first, then inject trainable LoRA
    for p in fm.parameters():
        p.requires_grad = False

    lora_modules = get_LoRA_module_names(args.lora_target)
    inject_trainable_LoRA(fm, rank=args.lora_rank, scale=args.lora_scale,
                          target_replace_modules=lora_modules)
    fm.to(device)   # LoRA layers are created on CPU — move everything to GPU
    unfreeze_all_LoRA_layers(fm)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = fm.load_state_dict(ckpt["lora_state_dict"], strict=False)
        print(f"[model] Resumed LoRA from {args.resume}  "
              f"(step={ckpt.get('step', '?')}, missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    n_total  = sum(p.numel() for p in fm.parameters())
    n_lora   = sum(p.numel() for p in fm.parameters() if p.requires_grad)
    print(f"[model] 4M params: {n_total/1e6:.1f}M total, {n_lora/1e6:.2f}M trainable (LoRA)", flush=True)

    # ── Optimizer + LR scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in fm.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(loader)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.05, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Mixed precision scaler ────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler()

    # ── Training loop ─────────────────────────────────────────────────────────
    fm.train()
    global_step  = 0
    accum_step   = 0
    accum_loss   = 0.0
    t0 = time.time()

    eff_batch = args.batch_size * args.grad_accum
    print(f"\n[train] Starting: {args.epochs} epochs, {total_steps} total steps", flush=True)
    print(f"[train] LoRA rank={args.lora_rank}  target={args.lora_target}  lr={args.lr}", flush=True)
    print(f"[train] batch={args.batch_size}  grad_accum={args.grad_accum}  effective_batch={eff_batch}", flush=True)
    target_str = "RGB only (depth+seg→rgb)" if args.rgb_only_target else "any→any"
    print(f"[train] Masking: {args.mode_2to1_prob*100:.0f}% 2→1, {(1-args.mode_2to1_prob)*100:.0f}% 1→1  target={target_str}\n", flush=True)

    optimizer.zero_grad()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            rgb   = batch["rgb"].to(device)    # (B, 3, 224, 224) [0,1]
            depth = batch["depth"].to(device)  # (B, 1, 224, 224) [0,1]
            seg   = batch["seg"].to(device)    # (B, 1, 224, 224) [0,1]
            B = rgb.shape[0]

            # ── Tokenize (no grad, tokenizers frozen) ─────────────────────────
            with torch.no_grad():
                rgb_m1_1 = rgb * 2.0 - 1.0
                rgb_ids  = divae.tokenize(rgb_m1_1)
                depth_ids = depth_tok.tokenize(depth)
                n_cls    = seg_tok.cls_emb.num_embeddings
                seg_int  = (seg.squeeze(1) * (n_cls - 1)).round().long().clamp(0, n_cls - 1)
                seg_ids  = seg_tok.tokenize(seg_int)

            # ── Any→any masking ────────────────────────────────────────────────
            mode_2to1 = random.random() < args.mode_2to1_prob
            enc_idxs, dec_idx = sample_enc_dec(mode_2to1, rgb_only_target=args.rgb_only_target)
            mod_dict, num_enc, num_dec = build_batch_mod_dict(
                rgb_ids, depth_ids, seg_ids, enc_idxs, dec_idx, device
            )

            # ── Forward (bf16 autocast) + loss ────────────────────────────────
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, mod_loss = fm(mod_dict, num_encoder_tokens=num_enc,
                                    num_decoder_tokens=num_dec, loss_type="mod")
                loss = loss / args.grad_accum

            # ── Backward ──────────────────────────────────────────────────────
            scaler.scale(loss).backward()
            accum_loss += loss.item()
            accum_step += 1

            if accum_step < args.grad_accum:
                continue   # accumulate more gradients before stepping

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in fm.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            loss_val    = accum_loss
            accum_loss  = 0.0
            accum_step  = 0

            epoch_loss  += loss_val
            epoch_steps += 1
            global_step += 1

            # ── Log ───────────────────────────────────────────────────────────
            if global_step % args.log_every == 0:
                elapsed = time.time() - t0
                lr_now  = scheduler.get_last_lr()[0]
                dec_mod  = MODALITIES[dec_idx]
                enc_mods = "+".join(MODALITIES[i].split("@")[0].replace("tok_", "") for i in enc_idxs)
                log_dict = {
                    "train/loss":       loss_val,
                    "train/lr":         lr_now,
                    "train/step":       global_step,
                    "train/epoch":      epoch + epoch_steps / len(loader),
                    "train/mode":       1 if mode_2to1 else 0,
                }
                for mod, ml in mod_loss.items():
                    short = mod.split("@")[0].replace("tok_", "")
                    log_dict[f"train/loss_{short}"] = ml.item()

                if use_wandb:
                    wandb.log(log_dict, step=global_step)

                print(
                    f"[{global_step:5d}/{total_steps}] ep={epoch+1}  "
                    f"loss={loss_val:.4f}  "
                    f"{enc_mods}→{dec_mod.split('@')[0].replace('tok_','')}  "
                    f"lr={lr_now:.2e}  t={elapsed:.0f}s",
                    flush=True,
                )

            # ── Save checkpoint ───────────────────────────────────────────────
            if global_step % args.save_every == 0:
                _save_lora(fm, args, global_step)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n[epoch {epoch+1}/{args.epochs}] avg_loss={avg_loss:.4f}\n", flush=True)
        if use_wandb:
            wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = _save_lora(fm, args, global_step, final=True)

    # ── Upload to HuggingFace ─────────────────────────────────────────────────
    if args.hf_repo:
        _upload_to_hf(final_path, args)

    if use_wandb:
        wandb.finish()
    print("[train] Done.", flush=True)


def _save_lora(fm, args, step, final=False):
    """Save only the LoRA weights (much smaller than full model)."""
    lora_state = {k: v for k, v in fm.state_dict().items() if "lora" in k}
    tag  = "final" if final else f"step{step:06d}"
    path = os.path.join(args.output_dir, f"lora_{tag}.pt")
    torch.save({
        "lora_state_dict": lora_state,
        "step":            step,
        "lora_rank":       args.lora_rank,
        "lora_scale":      args.lora_scale,
        "lora_target":     args.lora_target,
        "fourm_base":      args.fourm_model,
    }, path)
    print(f"[ckpt] Saved LoRA weights → {path}  ({len(lora_state)} tensors)", flush=True)
    return path


def _upload_to_hf(lora_path, args):
    """Upload LoRA checkpoint to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        token = os.environ.get("HF_TOKEN")

        print(f"[hf] Creating/checking repo {args.hf_repo} ...", flush=True)
        create_repo(args.hf_repo, token=token, exist_ok=True, repo_type="model")

        # Upload the final LoRA weights
        print(f"[hf] Uploading {lora_path} → {args.hf_repo} ...", flush=True)
        api.upload_file(
            path_or_fileobj=lora_path,
            path_in_repo="lora_final.pt",
            repo_id=args.hf_repo,
            token=token,
            commit_message=f"LoRA fine-tune on LIBERO — rank={args.lora_rank} target={args.lora_target} epochs={args.epochs}",
        )

        # Upload a small README with the config
        readme = (
            f"# 4M-21 LoRA fine-tuned on LIBERO\n\n"
            f"Base model: `{args.fourm_model}`\n\n"
            f"## LoRA config\n"
            f"- rank: {args.lora_rank}\n"
            f"- scale: {args.lora_scale}\n"
            f"- target: {args.lora_target}\n\n"
            f"## Training\n"
            f"- dataset: LIBERO-10 (RGB + depth + seg, {args.epochs} epochs)\n"
            f"- objective: any→any multimodal masked modeling\n"
            f"- modalities: tok_rgb@224, tok_depth@224, tok_semseg@224\n\n"
            f"## Usage\n"
            f"```python\n"
            f"from src.pipeline.fourm_image_processor_v2 import FourMImageProcessor\n"
            f"processor = FourMImageProcessor(\n"
            f"    lora_checkpoint='path/to/lora_final.pt',\n"
            f"    use_depth=True, use_seg=True,\n"
            f")\n"
            f"```\n"
        )
        readme_path = os.path.join(args.output_dir, "README.md")
        Path(readme_path).write_text(readme)
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.hf_repo,
            token=token,
        )

        print(f"[hf] Upload complete → https://huggingface.co/{args.hf_repo}", flush=True)
    except Exception as e:
        print(f"[hf] Upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
