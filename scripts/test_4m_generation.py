#!/usr/bin/env python3
"""
Test de génération 4M avec CFG (Classifier-Free Guidance).
Compare 4 configs sur depth→RGB :
  - No LoRA, no CFG  (base 4M, domain shift visible)
  - No LoRA, CFG=2   (base 4M amplifié)
  - LoRA, no CFG     (LoRA seul)
  - LoRA, CFG=2      (optimal : LoRA + guidance)

Usage:
    python scripts/test_4m_generation.py \
        --parquet /scratch/libero_thermal/data/train \
        --lora    /scratch/finetune_4m_lora_rgb/lora_step022000.pt \
        --n 4 --maskgit_steps 8 --temperature 3.0 --cfg_scale 2.0
"""
import argparse
import copy
import os
import sys

import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "third_party", "ml-4m"))

from fourm.models.fm import FM
from fourm.vq.vqvae import DiVAE, VQ
from fourm.models.lora_utils import inject_trainable_LoRA, get_LoRA_module_names
from fourm.models.generate import GenerationSampler

NUM_PATCHES = 196
PATCH_GRID  = 14


def build_cond_mod_dict(token_ids_dict, B, device):
    """Encoder-only: input_mask=False (visible), target_mask=True (not a decoder target)."""
    mod_dict = {}
    for mod, ids in token_ids_dict.items():
        mod_dict[mod] = {
            "tensor":                 ids.reshape(B, -1),
            "input_mask":             torch.zeros(B, NUM_PATCHES, dtype=torch.bool, device=device),
            "target_mask":            torch.ones( B, NUM_PATCHES, dtype=torch.bool, device=device),
            "decoder_attention_mask": torch.zeros(B, NUM_PATCHES, dtype=torch.bool, device=device),
        }
    return mod_dict


def build_target_mod_dict(B, device):
    """Empty RGB target: input_mask=True (hidden from encoder), target_mask=False (all to generate)."""
    return {
        "tensor":                 torch.zeros(B, NUM_PATCHES, dtype=torch.long,  device=device),
        "input_mask":             torch.ones( B, NUM_PATCHES, dtype=torch.bool,  device=device),
        "target_mask":            torch.zeros(B, NUM_PATCHES, dtype=torch.bool,  device=device),
        "decoder_attention_mask": torch.zeros(B, NUM_PATCHES, dtype=torch.bool,  device=device),
    }


@torch.no_grad()
def generate_rgb(sampler, cond_ids_dict, B, device,
                 n_steps=8, temperature=3.0, cfg_scale=1.0):
    """
    MaskGIT generation of tok_rgb@224.
    cfg_scale > 1 uses guided_maskgit_step_batched (2x forward passes).
    cfg_scale = 1 uses plain maskgit_step_batched.
    """
    mod_dict = build_cond_mod_dict(cond_ids_dict, B, device)
    mod_dict["tok_rgb@224"] = build_target_mod_dict(B, device)

    cond_mods = list(cond_ids_dict.keys())   # modalities to null-out for CFG

    # Linear token schedule
    tokens_per_step, already = [], 0
    for s in range(n_steps):
        target = int(NUM_PATCHES * (s + 1) / n_steps)
        tokens_per_step.append(max(1, target - already))
        already = target

    for num_select in tokens_per_step:
        if cfg_scale > 1.0:
            sampler.guided_maskgit_step_batched(
                mod_dict, "tok_rgb@224",
                num_select=num_select,
                temperature=temperature,
                top_k=0, top_p=1.0,
                conditioning=cond_mods,
                guidance_scale=cfg_scale,
            )
        else:
            sampler.maskgit_step_batched(
                mod_dict, "tok_rgb@224",
                num_select=num_select,
                temperature=temperature,
                top_k=0, top_p=1.0,
            )

    tok_rgb = mod_dict["tok_rgb@224"]["tensor"]
    return tok_rgb.reshape(B, PATCH_GRID, PATCH_GRID)


def load_fm(fourm_model, lora_path, device):
    try:
        fm = FM.from_pretrained(fourm_model, local_files_only=True)
    except Exception:
        fm = FM.from_pretrained(fourm_model)

    if lora_path:
        ckpt = torch.load(lora_path, map_location="cpu")
        rank   = ckpt.get("lora_rank",   8)
        scale  = ckpt.get("lora_scale",  1.0)
        target = ckpt.get("lora_target", "attention")
        inject_trainable_LoRA(fm, rank=rank, scale=scale,
                              target_replace_modules=get_LoRA_module_names(target))
        fm.to(device)
        fm.load_state_dict(ckpt["lora_state_dict"], strict=False)
        print(f"  LoRA loaded: step={lora_path.split('step')[-1].split('.')[0]}  rank={rank}")
    else:
        fm.to(device)
        print("  No LoRA — base 4M only")

    fm.eval()
    for p in fm.parameters():
        p.requires_grad = False
    return fm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",       required=True)
    parser.add_argument("--lora",          default=None)
    parser.add_argument("--n",             type=int,   default=4)
    parser.add_argument("--maskgit_steps", type=int,   default=8)
    parser.add_argument("--temperature",   type=float, default=3.0,
                        help="Official 4M default for X2RGB: 3.0")
    parser.add_argument("--cfg_scale",     type=float, default=2.0,
                        help="CFG scale (1.0=off, 2.0=official default)")
    parser.add_argument("--divae_steps",   type=int,   default=50)
    parser.add_argument("--out_dir",       default="eval_results/4m_generation_test")
    parser.add_argument("--fourm_model",   default="EPFL-VILAB/4M-21_XL")
    parser.add_argument("--divae_ckpt",    default="EPFL-VILAB/4M_tokenizers_rgb_16k_224-448")
    parser.add_argument("--depth_ckpt",    default="EPFL-VILAB/4M_tokenizers_depth_8k_224-448")
    parser.add_argument("--no_lora_compare", action="store_true",
                        help="Also generate without LoRA for comparison (loads base model twice)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[test] Device: {device}  steps={args.maskgit_steps}  T={args.temperature}  CFG={args.cfg_scale}")

    # ── Tokenizers ────────────────────────────────────────────────────────────
    print("[test] Loading DiVAE ...")
    try:
        divae = DiVAE.from_pretrained(args.divae_ckpt, local_files_only=True)
    except Exception:
        divae = DiVAE.from_pretrained(args.divae_ckpt)
    divae.to(device).eval()

    print("[test] Loading depth tokenizer ...")
    try:
        depth_tok = VQ.from_pretrained(args.depth_ckpt, local_files_only=True)
    except Exception:
        depth_tok = VQ.from_pretrained(args.depth_ckpt)
    depth_tok.to(device).eval()

    # ── 4M model(s) ───────────────────────────────────────────────────────────
    print(f"[test] Loading 4M (with LoRA={args.lora}) ...")
    fm_lora = load_fm(args.fourm_model, args.lora, device)
    sampler_lora = GenerationSampler(fm_lora)

    fm_base, sampler_base = None, None
    if args.no_lora_compare:
        print("[test] Loading base 4M (no LoRA) for comparison ...")
        fm_base = load_fm(args.fourm_model, None, device)
        sampler_base = GenerationSampler(fm_base)

    # ── Data ──────────────────────────────────────────────────────────────────
    from utils.parquet_dataset import ParquetThermalDataset
    ds = ParquetThermalDataset(args.parquet, chunk_size=1)
    print(f"[test] Dataset: {len(ds)} frames")
    indices = [i * (len(ds) // args.n) for i in range(args.n)]

    def psnr(a, b):
        mse = ((a - b) ** 2).mean().item()
        return -10 * np.log10(mse + 1e-8)

    for idx, sample_idx in enumerate(indices):
        item  = ds[sample_idx]
        rgb   = item["rgb"].unsqueeze(0).to(device)
        depth = item["depth"].unsqueeze(0).to(device)
        B = 1

        print(f"\n[test] Sample {idx} (idx {sample_idx})")

        with torch.no_grad():
            rgb_ids   = divae.tokenize(rgb * 2.0 - 1.0)   # (1,14,14)
            depth_ids = depth_tok.tokenize(depth)           # (1,14,14)

            rgb_448 = TF.resize(rgb, [448, 448],
                                interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

            # ── DiVAE passthrough (upper bound) ───────────────────────────────
            rgb_pt = divae.decode_tokens(rgb_ids, timesteps=args.divae_steps)
            rgb_pt = (rgb_pt.clamp(-1, 1) + 1) / 2
            print(f"  DiVAE passthrough:          PSNR={psnr(rgb_448, rgb_pt):.1f} dB")

            # ── LoRA + CFG (main) ─────────────────────────────────────────────
            tok = generate_rgb(sampler_lora, {"tok_depth@224": depth_ids},
                               B, device, args.maskgit_steps, args.temperature, args.cfg_scale)
            rgb_gen = divae.decode_tokens(tok, timesteps=args.divae_steps)
            rgb_gen = (rgb_gen.clamp(-1, 1) + 1) / 2
            label = f"LoRA+CFG({args.cfg_scale})" if args.cfg_scale > 1 else "LoRA"
            print(f"  {label:26s}  PSNR={psnr(rgb_448, rgb_gen):.1f} dB")

            # ── LoRA without CFG ──────────────────────────────────────────────
            tok_nocfg = generate_rgb(sampler_lora, {"tok_depth@224": depth_ids},
                                     B, device, args.maskgit_steps, args.temperature, 1.0)
            rgb_nocfg = divae.decode_tokens(tok_nocfg, timesteps=args.divae_steps)
            rgb_nocfg = (rgb_nocfg.clamp(-1, 1) + 1) / 2
            print(f"  LoRA no CFG:                PSNR={psnr(rgb_448, rgb_nocfg):.1f} dB")

            # ── Base 4M (no LoRA) ─────────────────────────────────────────────
            rgb_base = None
            if sampler_base is not None:
                tok_b = generate_rgb(sampler_base, {"tok_depth@224": depth_ids},
                                     B, device, args.maskgit_steps, args.temperature, args.cfg_scale)
                rgb_base = divae.decode_tokens(tok_b, timesteps=args.divae_steps)
                rgb_base = (rgb_base.clamp(-1, 1) + 1) / 2
                print(f"  Base 4M+CFG (no LoRA):      PSNR={psnr(rgb_448, rgb_base):.1f} dB")

        # ── Save panels ───────────────────────────────────────────────────────
        def t2pil(t, S=224):
            t = t.squeeze(0).clamp(0, 1).cpu()
            t = TF.resize(t, [S, S], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        def depth2pil(d, S=224):
            d = d.squeeze(0).squeeze(0).cpu().numpy()
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)
            return Image.fromarray((d * 255).astype(np.uint8)).convert("RGB").resize((S, S), Image.BILINEAR)

        S = 224
        panels = [t2pil(rgb_448), depth2pil(depth), t2pil(rgb_pt), t2pil(rgb_gen), t2pil(rgb_nocfg)]
        labels = ["GT RGB", "Depth", f"Passthrough\n({psnr(rgb_448,rgb_pt):.0f}dB)",
                  f"LoRA+CFG{args.cfg_scale}\n({psnr(rgb_448,rgb_gen):.0f}dB)",
                  f"LoRA no CFG\n({psnr(rgb_448,rgb_nocfg):.0f}dB)"]
        if rgb_base is not None:
            panels.append(t2pil(rgb_base))
            labels.append(f"Base 4M+CFG\n({psnr(rgb_448,rgb_base):.0f}dB)")

        gap = 4
        n_panels = len(panels)
        canvas = Image.new("RGB", (S * n_panels + gap * (n_panels - 1), S + 24), (20, 20, 20))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        for i, (img, lab) in enumerate(zip(panels, labels)):
            canvas.paste(img, (i * (S + gap), 0))
            for j, line in enumerate(lab.split("\n")):
                draw.text((i * (S + gap) + 4, S + 2 + j * 11), line, fill=(200, 200, 200))

        out_path = os.path.join(args.out_dir, f"compare_{idx:03d}.png")
        canvas.save(out_path)
        print(f"  Saved → {out_path}")

    print("\n[test] Done.")


if __name__ == "__main__":
    main()
