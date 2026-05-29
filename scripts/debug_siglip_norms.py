#!/usr/bin/env python3
"""
debug_siglip_norms.py
─────────────────────
Diagnoses why SigLIP token norms show extreme outliers (mean ~776k, median ~182).

Three hypotheses tested:
  A) Real — specific parquet frames cause numerical explosion in SigLIP (bfloat16)
  B) Architectural — extraction layer is wrong (CLS token, pre-LayerNorm, pre-connector)
  C) Pipeline bug — double normalization or wrong input range into vision_model

Checks:
  D1  Per-frame norm scan + NaN/Inf detection
  D2  Input range before/after prepare_images
  D3  Before vs after connector (for outlier frames)
  D4  Per-dimension max absolute value
  D5  Log-scale norm histogram
  D6  bfloat16 vs float32 comparison (outlier frames only)
  D7  Actual RGB images of outlier frames

Usage:
  python scripts/debug_siglip_norms.py \\
      --parquet /scratch/libero_thermal/data/train \\
      --n_frames 200 \\
      --output_dir analysis_results/siglip_debug
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "third_party", "lerobot", "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reused from analyze_encoder_failure.py
# ─────────────────────────────────────────────────────────────────────────────

def load_siglip_encoder(smolvla_checkpoint: str, device: str):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    log.info(f"Loading SmolVLA from {smolvla_checkpoint} ...")
    policy = SmolVLAPolicy.from_pretrained(smolvla_checkpoint)
    policy.eval().to(device)
    vlm = policy.model.vlm_with_expert
    img_keys = list(policy.config.image_features.keys())
    log.info(f"image_features keys: {img_keys}")
    log.info(f"resize_imgs_with_padding: {policy.config.resize_imgs_with_padding}")
    return policy, vlm, img_keys


def load_frames_indexed(parquet_dir: str, n_frames: int, img_size: int = 224):
    """Returns (frames tensor, dataset_indices list)."""
    from utils.parquet_dataset import ParquetThermalDataset
    ds = ParquetThermalDataset(parquet_dir, chunk_size=1)
    log.info(f"Dataset: {len(ds)} frames  →  sampling {n_frames}")
    indices = [i * (len(ds) // n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        item = ds[idx]
        rgb = item["rgb"]
        if rgb.shape[-2:] != (img_size, img_size):
            rgb = F.interpolate(rgb.unsqueeze(0), (img_size, img_size),
                                mode="bilinear", align_corners=False).squeeze(0)
        frames.append(rgb)
    return torch.stack(frames), indices


def _save_json(data: dict, out_dir: str, name: str):
    def _cvt(o):
        if isinstance(o, (np.integer, np.floating)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, bool): return bool(o)
        return o
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump({k: _cvt(v) for k, v in data.items()}, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction — one frame at a time
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_per_frame(policy, vlm, img_key: str,
                      frames: torch.Tensor, device: str):
    """
    Run SigLIP on each frame individually.
    Returns:
      per_frame_stats : list of dicts (has_nan, has_inf, max_norm, mean_norm, n_tokens_above_1000)
      all_feats       : list of (1, N_tok, D) tensors (float32)
      all_pv          : list of (1, 3, H, W) preprocessed pixel tensors (float32)
    """
    per_frame_stats = []
    all_feats       = []
    all_pv          = []

    for i, frame in enumerate(frames):
        frame_4d = frame.unsqueeze(0).to(device)

        # Capture input after prepare_images
        images_list, _ = policy.prepare_images({img_key: frame_4d})
        if not images_list:
            log.warning(f"Frame {i}: prepare_images returned empty")
            per_frame_stats.append({"has_nan": True, "has_inf": False,
                                    "max_norm": float("nan"), "mean_norm": float("nan"),
                                    "n_tokens_above_1000": 0, "pv_min": float("nan"), "pv_max": float("nan")})
            continue
        pv = images_list[0].to(device)

        # Feature extraction
        feat = vlm.embed_image(pv).cpu().float()   # (1, N_tok, D)

        norms = feat.norm(dim=-1).flatten()         # (N_tok,)
        has_nan = bool(torch.isnan(feat).any())
        has_inf = bool(torch.isinf(feat).any())

        # Replace nan/inf with 0 for statistics
        norms_clean = torch.nan_to_num(norms, nan=0.0, posinf=1e9, neginf=0.0)

        per_frame_stats.append({
            "frame_idx":            i,
            "has_nan":              has_nan,
            "has_inf":              has_inf,
            "max_norm":             float(norms_clean.max()),
            "mean_norm":            float(norms_clean.mean()),
            "n_tokens_above_1000":  int((norms_clean > 1000).sum()),
            "pv_min":               float(pv.min().item()),
            "pv_max":               float(pv.max().item()),
        })
        all_feats.append(feat)
        all_pv.append(pv.cpu().float())

        if i == 0:
            log.info(f"Frame 0: feat shape={feat.shape}  "
                     f"max_norm={norms_clean.max():.1f}  "
                     f"pv range=[{pv.min():.3f}, {pv.max():.3f}]")
        if i % 50 == 0:
            log.info(f"  ... frame {i}/{len(frames)}")

    return per_frame_stats, all_feats, all_pv


# ─────────────────────────────────────────────────────────────────────────────
# D1 — Per-frame norm scan + NaN/Inf
# ─────────────────────────────────────────────────────────────────────────────

def run_D1(per_frame_stats: list, ds_indices: list, out_dir: str) -> dict:
    max_norms  = [s["max_norm"]  for s in per_frame_stats]
    mean_norms = [s["mean_norm"] for s in per_frame_stats]
    n_nan = sum(s["has_nan"]  for s in per_frame_stats)
    n_inf = sum(s["has_inf"]  for s in per_frame_stats)
    n_outlier = sum(s["n_tokens_above_1000"] > 0 for s in per_frame_stats)

    log.info(f"D1: NaN frames={n_nan}, Inf frames={n_inf}, "
             f"frames with norm>1000={n_outlier}/{len(per_frame_stats)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("D1 — Per-frame SigLIP Max Norm  (NaN/Inf scan)", fontsize=13, fontweight="bold")

    ax = axes[0]
    colors = ["#D32F2F" if n > 1000 else "#2196F3" for n in max_norms]
    ax.scatter(range(len(max_norms)), max_norms, c=colors, s=15, alpha=0.7)
    ax.axhline(1000, color="orange", ls="--", lw=1, label="1000 threshold")
    ax.set_xlabel("Frame index"); ax.set_ylabel("Max token norm")
    ax.set_title(f"Max norm per frame  ({n_outlier} frames > 1000, red)")
    ax.set_yscale("symlog", linthresh=500)
    ax.legend()

    ax = axes[1]
    ax.scatter(range(len(mean_norms)), mean_norms, c=colors, s=15, alpha=0.7)
    ax.set_xlabel("Frame index"); ax.set_ylabel("Mean token norm")
    ax.set_title(f"Mean norm per frame  (NaN={n_nan}, Inf={n_inf})")
    ax.set_yscale("symlog", linthresh=200)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "D1_per_frame_norms.png"), dpi=150)
    plt.close()

    # Top-5 outlier frames
    sorted_idx = sorted(range(len(max_norms)), key=lambda i: max_norms[i], reverse=True)
    top5 = [{**per_frame_stats[i], "dataset_index": ds_indices[i]} for i in sorted_idx[:5]]
    log.info(f"D1 top-5 outlier frames: {[(t['dataset_index'], t['max_norm']) for t in top5]}")

    result = {
        "n_nan_frames": n_nan, "n_inf_frames": n_inf,
        "n_outlier_frames_norm_above_1000": n_outlier,
        "frac_outlier": n_outlier / max(len(per_frame_stats), 1),
        "max_norm_overall": float(max(max_norms)) if max_norms else 0.0,
        "top5_outliers": top5,
        "hypothesis_A_supported": n_outlier > 0 and n_outlier < len(per_frame_stats) * 0.3,
    }
    _save_json(result, out_dir, "D1_per_frame_norms.json")
    return result, sorted_idx[:5]


# ─────────────────────────────────────────────────────────────────────────────
# D2 — Input range check
# ─────────────────────────────────────────────────────────────────────────────

def run_D2(per_frame_stats: list, frames: torch.Tensor, out_dir: str) -> dict:
    # Raw parquet values
    raw_min = float(frames.min()); raw_max = float(frames.max())

    # After prepare_images (stored in per_frame_stats)
    pv_mins = [s["pv_min"] for s in per_frame_stats if not np.isnan(s["pv_min"])]
    pv_maxs = [s["pv_max"] for s in per_frame_stats if not np.isnan(s["pv_max"])]

    pv_global_min = float(min(pv_mins)) if pv_mins else float("nan")
    pv_global_max = float(max(pv_maxs)) if pv_maxs else float("nan")

    bug_raw = raw_min < -0.01 or raw_max > 1.01
    bug_pv  = pv_global_min < -1.05 or pv_global_max > 1.05

    log.info(f"D2: raw RGB [{raw_min:.4f}, {raw_max:.4f}]  "
             f"after prepare_images [{pv_global_min:.4f}, {pv_global_max:.4f}]  "
             f"bug_raw={bug_raw}  bug_pv={bug_pv}")

    result = {
        "raw_rgb_min": raw_min, "raw_rgb_max": raw_max,
        "pv_min": pv_global_min, "pv_max": pv_global_max,
        "bug_raw_out_of_range": bug_raw,
        "bug_pv_out_of_range":  bug_pv,
        "hypothesis_C_input_range": bug_raw or bug_pv,
    }
    _save_json(result, out_dir, "D2_input_range.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# D3 — Before vs after connector (outlier frames)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_D3(policy, vlm, img_key: str, frames: torch.Tensor,
           outlier_frame_indices: list, device: str, out_dir: str) -> dict:
    """Compare norms before and after the SmolVLM2 connector for outlier frames."""
    vision_model = vlm.get_vlm_model().vision_model
    connector    = vlm.get_vlm_model().connector

    pre_norms, post_norms, labels = [], [], []

    for rank, fi in enumerate(outlier_frame_indices[:5]):
        frame = frames[fi].unsqueeze(0).to(device)
        images_list, _ = policy.prepare_images({img_key: frame})
        if not images_list:
            continue
        pv = images_list[0].to(device)

        # Pre-connector
        hidden = vision_model(
            pixel_values=pv.to(dtype=vision_model.dtype)
        ).last_hidden_state.float()
        pre_norm = float(hidden.norm(dim=-1).max())

        # Post-connector
        post = connector(hidden.to(dtype=next(connector.parameters()).dtype)).float()
        post_norm = float(post.norm(dim=-1).max())

        pre_norms.append(pre_norm)
        post_norms.append(post_norm)
        labels.append(f"frame#{fi}")
        log.info(f"D3 frame#{fi}: pre={pre_norm:.1f}  post={post_norm:.1f}  "
                 f"ratio={post_norm/max(pre_norm, 1e-6):.2f}x")

    if not pre_norms:
        _save_json({"skipped": True}, out_dir, "D3_connector_comparison.json")
        return {}

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("D3 — Max Token Norm: Before vs After Connector (outlier frames)",
                 fontsize=12, fontweight="bold")
    ax.bar(x - 0.2, pre_norms,  0.4, label="Pre-connector  (vision_model output)", color="#FF9800", alpha=0.85)
    ax.bar(x + 0.2, post_norms, 0.4, label="Post-connector (embed_image output)",  color="#2196F3", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Max token norm (log scale)"); ax.set_yscale("symlog", linthresh=200)
    ax.legend(); ax.set_title("Explosion before connector → SigLIP instability (A)\n"
                               "Explosion after connector → connector amplifies (B)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "D3_connector_comparison.png"), dpi=150)
    plt.close()

    explosion_in_connector = any(post > 10 * pre for pre, post in zip(pre_norms, post_norms))
    explosion_pre_connector = any(pre > 1000 for pre in pre_norms)

    result = {
        "pre_norms": pre_norms, "post_norms": post_norms, "frame_labels": labels,
        "explosion_already_before_connector": explosion_pre_connector,
        "connector_amplifies_explosion": explosion_in_connector,
        "hypothesis_A_siglip_instability": explosion_pre_connector,
        "hypothesis_B_connector_amplifies": explosion_in_connector and not explosion_pre_connector,
    }
    _save_json(result, out_dir, "D3_connector_comparison.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# D4 — Per-dimension max absolute value
# ─────────────────────────────────────────────────────────────────────────────

def run_D4(all_feats: list, out_dir: str) -> dict:
    if not all_feats:
        return {}

    stacked = torch.cat(all_feats, dim=0).reshape(-1, all_feats[0].shape[-1])  # (N*T, D)
    stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e9, neginf=-1e9)

    dim_max  = stacked.abs().max(dim=0).values.numpy()   # (D,)
    dim_std  = stacked.float().std(dim=0).numpy()

    sorted_max = np.sort(dim_max)[::-1]
    n_rogue = int((dim_max > 1000).sum())
    top3_dims = np.argsort(dim_max)[::-1][:3].tolist()

    log.info(f"D4: {n_rogue} dimensions with max|value| > 1000  "
             f"top-3 dims={top3_dims}  their max={[float(dim_max[d]) for d in top3_dims]}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("D4 — Per-dimension Max |Value|  (which dims explode?)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(sorted_max, color="#9C27B0", lw=1.5)
    ax.axhline(1000, color="red", ls="--", lw=1, label="1000 threshold")
    ax.set_xlabel("Dimension rank (sorted by max)"); ax.set_ylabel("Max |value|")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_title(f"{n_rogue} / {len(dim_max)} dims exceed 1000")
    ax.legend()

    ax = axes[1]
    ax.bar(range(min(50, len(dim_max))), sorted(dim_max, reverse=True)[:50],
           color="#9C27B0", alpha=0.7)
    ax.set_xlabel("Top-50 dimensions (sorted)"); ax.set_ylabel("Max |value|")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_title("Top-50 most extreme dimensions")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "D4_per_dim_max.png"), dpi=150)
    plt.close()

    # Effective rank with outlier dims included vs excluded
    def eff_rank(s):
        s = np.array(s); s = s[s > 0]
        p = s / s.sum()
        return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

    _, sv_all, _ = np.linalg.svd(stacked.numpy()[:2000], full_matrices=False)
    rank_all = eff_rank(sv_all)

    # SVD without rogue dims
    mask = dim_max < 1000
    stacked_clean = stacked[:, mask]
    if stacked_clean.shape[1] > 1:
        _, sv_clean, _ = np.linalg.svd(stacked_clean.numpy()[:2000], full_matrices=False)
        rank_clean = eff_rank(sv_clean)
    else:
        rank_clean = float("nan")

    log.info(f"D4: effective rank with all dims={rank_all:.1f}, "
             f"without rogue dims={rank_clean:.1f}")

    result = {
        "n_rogue_dims_above_1000":     n_rogue,
        "top3_rogue_dims":             top3_dims,
        "top3_rogue_dim_maxvals":      [float(dim_max[d]) for d in top3_dims],
        "effective_rank_all":          rank_all,
        "effective_rank_clean":        rank_clean,
        "rank_collapse_explained_by_rogue_dims": rank_clean > rank_all * 2,
    }
    _save_json(result, out_dir, "D4_per_dim_max.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# D5 — Log-scale norm histogram
# ─────────────────────────────────────────────────────────────────────────────

def run_D5(all_feats: list, out_dir: str) -> dict:
    if not all_feats:
        return {}

    all_norms = torch.cat([f.norm(dim=-1).flatten() for f in all_feats])
    all_norms = torch.nan_to_num(all_norms, nan=0.0, posinf=1e9).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("D5 — Norm Distribution (log scale reveals outlier structure)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.hist(all_norms, bins=100, color="#2196F3", alpha=0.8)
    ax.set_xlabel("Norm"); ax.set_ylabel("Count")
    ax.set_title(f"Linear scale  (median={np.median(all_norms):.1f}, mean={all_norms.mean():.1f})")

    ax = axes[1]
    log_norms = np.log10(all_norms + 1)
    ax.hist(log_norms, bins=100, color="#F44336", alpha=0.8)
    ax.set_xlabel("log10(norm + 1)"); ax.set_ylabel("Count")
    ax.set_title("Log scale — bimodal = outlier frames, unimodal = uniform noise")

    # Annotate quantiles
    q50  = np.percentile(all_norms, 50)
    q99  = np.percentile(all_norms, 99)
    q999 = np.percentile(all_norms, 99.9)
    q_max = all_norms.max()
    ax.axvline(np.log10(q99  + 1), color="orange", ls="--", lw=1, label=f"p99={q99:.0f}")
    ax.axvline(np.log10(q999 + 1), color="red",    ls="--", lw=1, label=f"p99.9={q999:.0f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "D5_logscale_norms.png"), dpi=150)
    plt.close()

    result = {
        "p50": float(q50), "p95": float(np.percentile(all_norms, 95)),
        "p99": float(q99), "p999": float(q999), "max": float(q_max),
        "frac_above_1000": float((all_norms > 1000).mean()),
    }
    _save_json(result, out_dir, "D5_logscale_norms.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# D6 — bfloat16 vs float32 (outlier frames)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_D6(policy, vlm, img_key: str, frames: torch.Tensor,
           outlier_frame_indices: list, device: str, out_dir: str) -> dict:
    vision_model = vlm.get_vlm_model().vision_model
    connector    = vlm.get_vlm_model().connector
    results = []

    for fi in outlier_frame_indices[:3]:
        frame = frames[fi].unsqueeze(0).to(device)
        images_list, _ = policy.prepare_images({img_key: frame})
        if not images_list:
            continue
        pv = images_list[0].to(device)

        row = {"frame_idx": fi}
        for dtype, label in [(torch.bfloat16, "bfloat16"), (torch.float32, "float32")]:
            try:
                h = vision_model(pixel_values=pv.to(dtype=dtype)).last_hidden_state.float()
                c = connector(h.to(dtype=dtype)).float()
                row[f"{label}_pre_norm"]  = float(h.norm(dim=-1).max())
                row[f"{label}_post_norm"] = float(c.norm(dim=-1).max())
            except Exception as e:
                row[f"{label}_error"] = str(e)
        results.append(row)
        bf = row.get('bfloat16_post_norm', float('nan'))
        fp = row.get('float32_post_norm',  float('nan'))
        log.info(f"D6 frame#{fi}: bf16_post={bf:.1f}  fp32_post={fp:.1f}")

    dtype_causes_explosion = any(
        r.get("bfloat16_post_norm", 0) > 1000 and r.get("float32_post_norm", 0) < 500
        for r in results
    )
    result = {"frames": results, "dtype_causes_explosion": dtype_causes_explosion,
              "hypothesis_A_bfloat16": dtype_causes_explosion}
    _save_json(result, out_dir, "D6_dtype_comparison.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# D7 — Visualize outlier frames
# ─────────────────────────────────────────────────────────────────────────────

def run_D7(frames: torch.Tensor, outlier_frame_indices: list,
           per_frame_stats: list, out_dir: str):
    from PIL import Image as PILImage
    for rank, fi in enumerate(outlier_frame_indices[:5]):
        rgb = frames[fi].permute(1, 2, 0).numpy()
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        img = PILImage.fromarray(rgb)
        stat = per_frame_stats[fi]
        path = os.path.join(out_dir, f"D7_outlier_frame{rank+1}_idx{fi}_norm{stat['max_norm']:.0f}.png")
        img.save(path)
        log.info(f"D7 saved: {path}  max_norm={stat['max_norm']:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(d1, d2, d3, d4, d5, d6, out_dir: str):
    lines = ["=" * 60, "SIGLIP NORM DEBUG — SUMMARY", "=" * 60, ""]

    # Hypothesis A: real instability on specific frames
    hyp_a = (d1.get("hypothesis_A_supported", False) or
             d3.get("hypothesis_A_siglip_instability", False) or
             d6.get("hypothesis_A_bfloat16", False))

    # Hypothesis B: architectural (connector amplifies)
    hyp_b = d3.get("hypothesis_B_connector_amplifies", False)

    # Hypothesis C: input range bug
    hyp_c = d2.get("hypothesis_C_input_range", False)

    lines += [
        f"  Hypothesis A (numerical instability on specific frames):  {'✓ CONFIRMED' if hyp_a else '✗ Not confirmed'}",
        f"  Hypothesis B (connector amplifies edge cases):             {'✓ CONFIRMED' if hyp_b else '✗ Not confirmed'}",
        f"  Hypothesis C (input range bug):                            {'✓ CONFIRMED' if hyp_c else '✗ Not confirmed'}",
        "",
        "─" * 60, "KEY NUMBERS", "─" * 60,
        f"  Frames with norm > 1000:   {d1.get('n_outlier_frames_norm_above_1000', '?')} / {d1.get('n_outlier_frames_norm_above_1000', '?') + 1}",
        f"  NaN frames:                {d1.get('n_nan_frames', '?')}",
        f"  Inf frames:                {d1.get('n_inf_frames', '?')}",
        f"  Input range [raw]:         [{d2.get('raw_rgb_min', '?'):.4f}, {d2.get('raw_rgb_max', '?'):.4f}]",
        f"  Input range [after prep]:  [{d2.get('pv_min', '?'):.4f}, {d2.get('pv_max', '?'):.4f}]",
        f"  Rogue dims (max|v|>1000):  {d4.get('n_rogue_dims_above_1000', '?')}",
        f"  Rank all dims:             {d4.get('effective_rank_all', '?'):.1f}",
        f"  Rank clean dims:           {d4.get('effective_rank_clean', '?'):.1f}",
        f"  p99 norm:                  {d5.get('p99', '?'):.1f}",
        f"  p99.9 norm:                {d5.get('p999', '?'):.1f}",
        f"  max norm:                  {d5.get('max', '?'):.1f}",
        f"  bfloat16 causes explosion: {d6.get('dtype_causes_explosion', '?')}",
        "",
        "─" * 60, "RECOMMENDATION", "─" * 60,
    ]

    if hyp_c:
        lines.append("  → Fix: input range to SigLIP is wrong. Check resize_with_pad output.")
    elif hyp_b:
        lines.append("  → Fix: connector amplifies unstable inputs. Run SigLIP in float32.")
    elif hyp_a and d6.get("dtype_causes_explosion", False):
        lines.append("  → Fix: cast SigLIP to float32 before feature extraction in analysis.")
        lines.append("         Or filter outlier frames from the parquet dataset.")
    elif hyp_a:
        lines.append("  → Fix: a few parquet frames produce extreme SigLIP values.")
        lines.append("         Filter frames with max_norm > 1000 from analysis.")
    else:
        lines.append("  → No clear bug found. The 776k mean may be from a different run/dataset.")

    lines += ["=" * 60]
    report = "\n".join(str(l) for l in lines)
    path = os.path.join(out_dir, "SUMMARY.txt")
    with open(path, "w") as f:
        f.write(report)
    print(report)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",    required=True)
    p.add_argument("--n_frames",   type=int, default=200)
    p.add_argument("--output_dir", default="analysis_results/siglip_debug")
    p.add_argument("--smolvla",    default="lerobot/smolvla_libero")
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    policy, vlm, img_keys = load_siglip_encoder(args.smolvla, device)
    img_key = img_keys[0]

    log.info("Loading frames ...")
    frames, ds_indices = load_frames_indexed(args.parquet, args.n_frames)
    log.info(f"Loaded {len(frames)} frames  shape={frames.shape}")

    log.info("Running per-frame SigLIP extraction (D1 + D2 data) ...")
    per_frame_stats, all_feats, all_pv = extract_per_frame(
        policy, vlm, img_key, frames, device)

    log.info("D1 ...")
    d1, outlier_indices = run_D1(per_frame_stats, ds_indices, args.output_dir)

    log.info("D2 ...")
    d2 = run_D2(per_frame_stats, frames, args.output_dir)

    log.info("D3 (outlier frames: pre vs post connector) ...")
    d3 = run_D3(policy, vlm, img_key, frames, outlier_indices, device, args.output_dir)

    log.info("D4 (per-dimension max) ...")
    d4 = run_D4(all_feats, args.output_dir)

    log.info("D5 (log-scale histogram) ...")
    d5 = run_D5(all_feats, args.output_dir)

    log.info("D6 (bfloat16 vs float32) ...")
    d6 = run_D6(policy, vlm, img_key, frames, outlier_indices, device, args.output_dir)

    log.info("D7 (save outlier images) ...")
    run_D7(frames, outlier_indices, per_frame_stats, args.output_dir)

    write_summary(d1, d2, d3, d4, d5, d6, args.output_dir)
    log.info(f"Done. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
