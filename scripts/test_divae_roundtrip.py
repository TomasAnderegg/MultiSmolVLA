#!/usr/bin/env python3
"""
Test minimal : RGB → DiVAE.tokenize() → DiVAE.decode() → RGB
Vérifie que la reconstruction VQ est correcte avant toute intervention du 4M.

Usage:
    python scripts/test_divae_roundtrip.py --image path/to/image.png
    python scripts/test_divae_roundtrip.py --parquet /scratch/libero_thermal/data/train --n 5
"""
import argparse
import os
import sys

import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "ml-4m"))

from fourm.vq.vqvae import DiVAE


def load_image(path: str, size: int = 448) -> torch.Tensor:
    """Load image → (1, 3, H, W) float [0,1]."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0)


def load_from_parquet(data_dir: str, n: int = 5, size: int = 448):
    """Load n RGB frames from parquet dataset, resized to `size`."""
    sys.path.insert(0, _ROOT)
    from utils.parquet_dataset import ParquetThermalDataset
    ds = ParquetThermalDataset(data_dir, chunk_size=1)
    frames = []
    for i in range(min(n, len(ds))):
        rgb = ds[i * (len(ds) // n)]["rgb"]   # (3, 224, 224)
        if size != 224:
            rgb = TF.resize(rgb, [size, size],
                            interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        frames.append(rgb.unsqueeze(0))
    return torch.cat(frames, dim=0)  # (n, 3, size, size)


def save_comparison(rgb_in, rgb_out, token_ids, out_path: str, idx: int):
    """Save side-by-side: input | reconstructed | token heatmap."""
    # Resize to same size for display
    def to_pil(t):
        t = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()
        return Image.fromarray((t * 255).astype(np.uint8))

    img_in  = to_pil(rgb_in)
    img_out = to_pil(rgb_out)

    # Token diversity heatmap: normalize unique token count per row
    tok_np = token_ids.squeeze(0).numpy().astype(np.float32)  # (14, 14)
    tok_norm = (tok_np - tok_np.min()) / (tok_np.max() - tok_np.min() + 1e-6)
    tok_pil = Image.fromarray((tok_norm * 255).astype(np.uint8)).convert("RGB")
    tok_pil = tok_pil.resize((224, 224), Image.NEAREST)

    # Concatenate
    W, H = img_in.size
    canvas = Image.new("RGB", (W * 3 + 8, H))
    canvas.paste(img_in,  (0, 0))
    canvas.paste(img_out, (W + 4, 0))
    canvas.paste(tok_pil, (W * 2 + 8, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    type=str, default=None, help="Single image path")
    parser.add_argument("--parquet",  type=str, default=None, help="Parquet data dir")
    parser.add_argument("--n",        type=int, default=5,    help="Number of frames from parquet")
    parser.add_argument("--size",     type=int, default=448,  help="Input resolution (448=native DiVAE, 224=4M tokens)")
    parser.add_argument("--divae",    default="EPFL-VILAB/4M_tokenizers_rgb_16k_224-448")
    parser.add_argument("--out_dir",  default="eval_results/divae_roundtrip")
    parser.add_argument("--divae_steps", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] Device: {device}")

    # ── Load DiVAE ────────────────────────────────────────────────────────────
    print(f"[test] Loading DiVAE from {args.divae} ...")
    try:
        divae = DiVAE.from_pretrained(args.divae, local_files_only=True)
    except Exception:
        divae = DiVAE.from_pretrained(args.divae)
    divae.to(device).eval()
    print(f"[test] DiVAE loaded  codebook={divae.codebook_size}  patch=16×16  grid=14×14")

    # ── Load images ───────────────────────────────────────────────────────────
    if args.image:
        rgb = load_image(args.image, size=args.size).to(device)
        print(f"[test] Loaded image: {args.image}  shape={tuple(rgb.shape)}")
    elif args.parquet:
        rgb = load_from_parquet(args.parquet, args.n, size=args.size).to(device)
        print(f"[test] Loaded {rgb.shape[0]} frames from parquet  size={args.size}×{args.size}")
    else:
        print("[test] No input specified — using random noise as sanity check")
        rgb = torch.rand(1, 3, 224, 224, device=device)

    B = rgb.shape[0]

    # ── Tokenize (encode → discrete token IDs) ────────────────────────────────
    print(f"\n[test] Step 1: DiVAE.tokenize()  ({B} images)")
    with torch.no_grad():
        rgb_m1_1  = rgb * 2.0 - 1.0
        token_ids = divae.tokenize(rgb_m1_1)   # (B, 14, 14)  long
    print(f"  token_ids: shape={tuple(token_ids.shape)}  dtype={token_ids.dtype}")
    print(f"  vocab usage: {token_ids.unique().numel()} / {divae.codebook_size} tokens used")
    print(f"  token range: [{token_ids.min().item()}, {token_ids.max().item()}]")

    # ── Decode from token IDs (quantized codebook path) ───────────────────────
    print(f"\n[test] Step 2: DiVAE.decode_tokens()  (steps={args.divae_steps})")
    with torch.no_grad():
        rgb_out = divae.decode_tokens(token_ids, timesteps=args.divae_steps)
    rgb_out = (rgb_out.clamp(-1, 1) + 1) / 2   # → [0,1]
    # decode_tokens outputs at native DiVAE resolution (448) — resize to input size
    if rgb_out.shape[-1] != rgb.shape[-1]:
        rgb_out = TF.resize(rgb_out, [rgb.shape[-2], rgb.shape[-1]],
                            interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    print(f"  rgb_out: shape={tuple(rgb_out.shape)}  range=[{rgb_out.min():.3f},{rgb_out.max():.3f}]")

    # ── PSNR / MSE ────────────────────────────────────────────────────────────
    print(f"\n[test] Quality metrics (input vs reconstructed):")
    for b in range(B):
        mse  = ((rgb[b] - rgb_out[b]) ** 2).mean().item()
        psnr = -10 * np.log10(mse + 1e-8)
        print(f"  frame {b}: MSE={mse:.5f}  PSNR={psnr:.1f} dB")

    # ── Save comparison images ────────────────────────────────────────────────
    print(f"\n[test] Saving comparisons to {args.out_dir} ...")
    for b in range(B):
        out_path = os.path.join(args.out_dir, f"roundtrip_{b:03d}.png")
        save_comparison(
            rgb[b:b+1].cpu(),
            rgb_out[b:b+1].cpu(),
            token_ids[b:b+1].cpu(),
            out_path, b,
        )

    print(f"\n[test] Done. Images: input | VQ reconstructed | token map")
    print(f"       If 'VQ reconstructed' is clean → tokenizer OK")
    print(f"       If blurry/noisy → problem in DiVAE itself")


if __name__ == "__main__":
    main()
