#!/usr/bin/env python3
"""
analyze_encoder_failure.py
──────────────────────────
Diagnostic pipeline to understand why 4M+MLP failed to replace SigLIP in SmolVLA.

Analyses produced (all optional, all run by default):
  A1  Token norm distributions          — fastest, most likely root cause
  A2  Token count & shape comparison    — structural mismatch detection
  A3  SVD spectral analysis             — effective rank / information capacity
  A4  Per-dimension alignment           — which dimensions are worst aligned
  A5  UMAP of patch embeddings          — geometric / topological structure
  A6  Cosine similarity heatmaps        — token-level alignment matrix
  A7  Attention entropy in SmolLM2      — does the language model still "see" things?
  A8  Action distribution              — gripper collapse signature

Usage:
  python scripts/analyze_encoder_failure.py \
      --parquet     /scratch/libero_thermal/data/train \
      --checkpoint  /scratch/checkpoints/block2_final.pt \
      --n_frames    200 \
      --output_dir  analysis_results/encoder_failure

  # Without checkpoint (random MLP, upper-bound alignment check):
  python scripts/analyze_encoder_failure.py --parquet /scratch/libero_thermal/data/train
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "third_party", "lerobot", "src"))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "ml-4m"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_siglip_encoder(smolvla_checkpoint: str, device: str):
    """Load the original SmolVLA policy (no 4M patching) to get the SigLIP encoder."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    log.info(f"Loading SmolVLA base policy from {smolvla_checkpoint} ...")
    policy = SmolVLAPolicy.from_pretrained(smolvla_checkpoint)
    policy.eval().to(device)
    vlm = policy.model.vlm_with_expert

    # Log the actual image keys this policy expects (from its config.input_features)
    img_keys = list(policy.config.image_features.keys())
    log.info(f"SmolVLA image_features keys: {img_keys}")
    log.info(f"SmolVLA resize_imgs_with_padding: {policy.config.resize_imgs_with_padding}")
    log.info("SmolVLA (SigLIP) loaded ✅")
    return policy, vlm, img_keys


def _resolve_checkpoint(checkpoint_path: str) -> str:
    """Return a local file path — download from HF Hub if checkpoint_path is a repo ID."""
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    # Looks like a HuggingFace repo ID (e.g. "TomasAnderegg/multismolvla-stage1-rgb-pertok")
    if "/" in checkpoint_path and not checkpoint_path.startswith("."):
        from huggingface_hub import hf_hub_download, list_repo_files
        log.info(f"Checkpoint not found locally — downloading from HF Hub: {checkpoint_path}")
        # Find the .pt file in the repo
        try:
            files = list(list_repo_files(checkpoint_path))
            pt_files = [f for f in files if f.endswith(".pt")]
            if not pt_files:
                raise FileNotFoundError(f"No .pt file found in repo {checkpoint_path}. Files: {files}")
            pt_file = sorted(pt_files)[-1]  # take the last one (e.g. final > stepXXX)
            log.info(f"Downloading {pt_file} from {checkpoint_path} ...")
            local_path = hf_hub_download(repo_id=checkpoint_path, filename=pt_file)
            log.info(f"Downloaded → {local_path}")
            return local_path
        except Exception as e:
            raise FileNotFoundError(f"Could not download checkpoint from {checkpoint_path}: {e}")
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def load_fourm_encoder(fourm_checkpoint: str, fourm_dim: int,
                       checkpoint_path: str | None, device: str):
    """Load 4M encoder + MLP connector (with optional trained weights).

    checkpoint_path can be:
      - a local path: /scratch/checkpoints/block2/block2_final.pt
      - a HF repo ID: TomasAnderegg/multismolvla-stage1-rgb-pertok
      - None: MLP uses random weights
    """
    from src.pipeline.encoder_4m import Encoder4M
    from src.pipeline.connector import MLPConnector

    log.info(f"Loading 4M encoder from {fourm_checkpoint} ...")
    enc = Encoder4M(checkpoint=fourm_checkpoint, device=device).eval()
    enc_out_dim = enc.output_dim  # read actual dim from the loaded encoder (e.g. 2048 for 4M-21-XL)
    log.info(f"4M encoder output_dim = {enc_out_dim}")

    mlp = None  # instantiated after we know the target dim from the checkpoint

    if checkpoint_path is not None:
        local_path = _resolve_checkpoint(checkpoint_path)
        log.info(f"Loading trained weights from {local_path} ...")
        state = torch.load(local_path, map_location="cpu", weights_only=False)
        log.info(f"Checkpoint keys (first 5): {list(state.keys())[:5]}")

        # Extract MLP state dict, trying several possible key prefixes
        mlp_state = None
        for prefix in [
            "smolvla.policy.base_policy.model.vlm_with_expert.fourm_to_vlm.",
            "block2.smolvla.policy.base_policy.model.vlm_with_expert.fourm_to_vlm.",
            "fourm_to_vlm.",
        ]:
            candidate = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if candidate:
                mlp_state = candidate
                log.info(f"MLP keys found with prefix '{prefix}' ({len(mlp_state)} tensors)")
                break

        if mlp_state and "mlp.0.weight" in mlp_state:
            # Read actual dims from the saved weights — never assume
            w0 = mlp_state["mlp.0.weight"]   # (smolvla_dim, encoder_dim)
            encoder_dim_ckpt = w0.shape[1]
            smolvla_dim_ckpt = w0.shape[0]
            log.info(f"Checkpoint MLP dims: encoder_dim={encoder_dim_ckpt}, smolvla_dim={smolvla_dim_ckpt}")
            use_ln = "mlp.3.weight" in mlp_state  # LayerNorm present only if key exists
            log.info(f"use_layer_norm={use_ln}")
            mlp = MLPConnector(encoder_dim=encoder_dim_ckpt,
                               smolvla_dim=smolvla_dim_ckpt,
                               use_layer_norm=use_ln).to(device).eval()
            missing, unexpected = mlp.load_state_dict(mlp_state, strict=True)
            log.info(f"MLP weights loaded ✅")
        else:
            log.warning("No fourm_to_vlm keys found in checkpoint — using random MLP weights")
            log.info(f"All checkpoint keys: {list(state.keys())[:20]}")
    else:
        log.warning("No checkpoint provided — MLP has random weights (untrained baseline)")

    if mlp is None:
        # Fallback: use enc output_dim → args fourm_dim as smolvla target
        smolvla_dim = fourm_dim if fourm_dim != enc_out_dim else 960
        mlp = MLPConnector(encoder_dim=enc_out_dim, smolvla_dim=smolvla_dim).to(device).eval()
        log.info(f"MLPConnector instantiated with random weights: {enc_out_dim}→{smolvla_dim}")

    return enc, mlp


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_siglip_features(policy, vlm, img_keys: list[str],
                             rgb_frames: torch.Tensor, device: str) -> torch.Tensor:
    """
    Extract SigLIP features for a batch of RGB frames.
    rgb_frames : (N, 3, H, W) float [0, 1]
    Returns    : (N, N_tokens, D_hidden)  projected into SmolLM2 space (BEFORE ×√D scaling)

    NOTE: embed_prefix() applies an additional ×√2048 scaling after embed_image().
    We capture features before that scaling — the relative comparison between SigLIP
    and 4M+MLP is still valid since both encoders go through the same scaling.
    """
    # Use the first image key from the policy config (handles any dataset key name)
    img_key = img_keys[0]
    log.info(f"SigLIP extraction using image key: '{img_key}'")

    all_feats = []
    for i in range(len(rgb_frames)):
        frame = rgb_frames[i].unsqueeze(0).to(device)
        batch = {img_key: frame}
        images_list, masks_list = policy.prepare_images(batch)
        if not images_list:
            log.warning(f"Frame {i}: prepare_images returned empty — skipping")
            continue
        pv = images_list[0].to(device)
        # embed_image: SigLIP vision_model → connector → (B, N_tokens, 2048)
        feat = vlm.embed_image(pv)
        if i == 0:
            log.info(f"SigLIP feature shape: {feat.shape}  "
                     f"(N_tokens={feat.shape[1]}, D={feat.shape[2]})")
        all_feats.append(feat.cpu().float())
    return torch.cat(all_feats, dim=0)  # (N, N_tokens, D)


@torch.no_grad()
def extract_fourm_features(enc, mlp, rgb_frames: torch.Tensor, device: str) -> torch.Tensor:
    """
    Extract 4M+MLP features for a batch of RGB frames.
    rgb_frames : (N, 3, H, W) float [0, 1]
    Returns    : (N, 196, 2048)
    """
    all_feats = []
    for i in range(len(rgb_frames)):
        frame = rgb_frames[i].unsqueeze(0).to(device)
        inputs = {"rgb": frame}
        tokens = enc(inputs)                         # (1, 196, 1024)
        tokens = tokens.to(mlp.mlp[0].weight.dtype)
        proj = mlp(tokens)                           # (1, 196, 2048)
        all_feats.append(proj.cpu())
    return torch.cat(all_feats, dim=0)  # (N, 196, 2048)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_frames(parquet_dir: str, n_frames: int, img_size: int = 224) -> torch.Tensor:
    """Load N evenly-spaced RGB frames from the parquet dataset."""
    from utils.parquet_dataset import ParquetThermalDataset
    ds = ParquetThermalDataset(parquet_dir, chunk_size=1)
    log.info(f"Dataset: {len(ds)} frames  →  sampling {n_frames}")
    indices = [i * (len(ds) // n_frames) for i in range(n_frames)]
    frames = []
    for idx in indices:
        item = ds[idx]
        rgb = item["rgb"]   # (3, H, W) float [0,1]
        if rgb.shape[-2:] != (img_size, img_size):
            rgb = F.interpolate(rgb.unsqueeze(0), (img_size, img_size),
                                mode="bilinear", align_corners=False).squeeze(0)
        frames.append(rgb)
    return torch.stack(frames)   # (N, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# A1 — Token norm distributions
# ─────────────────────────────────────────────────────────────────────────────

def analysis_token_norms(sig_feats: torch.Tensor, fourm_feats: torch.Tensor,
                         out_dir: str) -> dict:
    """
    Compare per-token L2 norms between SigLIP and 4M+MLP.
    sig_feats   : (N, N_sig, D)
    fourm_feats : (N, 196, D)

    WHY: SmolLM2 self-attention score = QKᵀ/√d. If K (visual tokens) have very different
    norms than what SmolLM2 was trained on with SigLIP, the attention temperature shifts,
    causing collapse or uniform attention.
    """
    sig_norms   = sig_feats.norm(dim=-1).flatten().numpy()     # (N * N_sig,)
    fourm_norms = fourm_feats.norm(dim=-1).flatten().numpy()   # (N * 196,)

    # Clip at p99 to avoid a single outlier collapsing the histogram
    sig_p99   = float(np.percentile(sig_norms,   99))
    fourm_p99 = float(np.percentile(fourm_norms, 99))
    sig_clip   = sig_norms[sig_norms     <= sig_p99]
    fourm_clip = fourm_norms[fourm_norms <= fourm_p99]
    x_max = max(sig_p99, fourm_p99)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("A1 — Token Norm Distributions", fontsize=14, fontweight="bold")

    ax = axes[0]
    bins = np.linspace(0, x_max, 80)
    ax.hist(sig_clip,   bins=bins, alpha=0.7, color="#2196F3", label=f"SigLIP (median={np.median(sig_norms):.1f})")
    ax.hist(fourm_clip, bins=bins, alpha=0.7, color="#F44336", label=f"4M+MLP (median={np.median(fourm_norms):.1f})")
    ax.set_xlabel("L2 norm of token embedding")
    ax.set_ylabel("Count")
    pct_outliers_sig = 100 * (1 - len(sig_clip)/len(sig_norms))
    ax.set_title(f"Norm histogram (clipped at p99,  {pct_outliers_sig:.1f}% SigLIP outliers removed)")
    ax.legend()

    ax = axes[1]
    stats = {
        "SigLIP":  {"mean": float(sig_norms.mean()),   "std": float(sig_norms.std()),
                    "median": float(np.median(sig_norms)), "p95": float(np.percentile(sig_norms, 95))},
        "4M+MLP":  {"mean": float(fourm_norms.mean()), "std": float(fourm_norms.std()),
                    "median": float(np.median(fourm_norms)), "p95": float(np.percentile(fourm_norms, 95))},
    }
    ratio = stats["4M+MLP"]["mean"] / max(stats["SigLIP"]["mean"], 1e-8)
    categories = ["mean", "std", "median", "p95"]
    sig_vals   = [stats["SigLIP"][c]  for c in categories]
    fourm_vals = [stats["4M+MLP"][c] for c in categories]
    x = np.arange(len(categories))
    ax.bar(x - 0.2, sig_vals,   0.4, label="SigLIP",  color="#2196F3", alpha=0.8)
    ax.bar(x + 0.2, fourm_vals, 0.4, label="4M+MLP",  color="#F44336", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_title(f"Norm statistics  (4M/SigLIP ratio = {ratio:.2f}x)")
    ax.legend()

    _annotate_danger(axes[1], ratio < 0.5 or ratio > 2.0,
                     f"Norm ratio {ratio:.2f}x {'→ ATTENTION DISTORTION LIKELY' if (ratio < 0.5 or ratio > 2.0) else '→ acceptable'}")

    plt.tight_layout()
    path = os.path.join(out_dir, "A1_token_norms.png")
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"A1 saved → {path}")

    # Use median for ratio — robust to outliers
    ratio_median = float(np.median(fourm_norms)) / max(float(np.median(sig_norms)), 1e-8)
    result = {**stats,
              "norm_ratio_4m_over_siglip": ratio,
              "norm_ratio_median": ratio_median,
              "flag_distortion": bool(ratio_median < 0.5 or ratio_median > 2.0)}
    _save_json(result, out_dir, "A1_token_norms.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A2 — Token count & shape
# ─────────────────────────────────────────────────────────────────────────────

def analysis_token_count(sig_feats: torch.Tensor, fourm_feats: torch.Tensor,
                         out_dir: str) -> dict:
    """
    Report token counts for both encoders.

    WHY: SmolLM2 processes visual tokens by prepending them to the text sequence.
    If SigLIP produces 729 tokens and 4M produces 196, the sequence length changes,
    breaking positional encodings (RoPE positions) for ALL subsequent text tokens.
    Even if features are perfectly aligned per-token, wrong positions destroy the
    temporal attention structure the policy was trained on.
    """
    n_sig, n_fourm = sig_feats.shape[1], fourm_feats.shape[1]
    d_sig, d_fourm = sig_feats.shape[2], fourm_feats.shape[2]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("A2 — Token Count & Dimensionality", fontsize=14, fontweight="bold")

    data = {"SigLIP": [n_sig, d_sig], "4M+MLP": [n_fourm, d_fourm]}
    x = np.arange(2)
    ax.bar(x - 0.2, [n_sig, n_fourm], 0.4,
           color=["#2196F3", "#F44336"], alpha=0.8,
           label=["SigLIP N_tokens", "4M+MLP N_tokens"])
    ax.set_xticks(x); ax.set_xticklabels(["N tokens", "Hidden dim"])
    # hidden dim on secondary axis
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [d_sig, d_fourm], 0.4,
            color=["#2196F3", "#F44336"], alpha=0.4, hatch="//")
    ax.set_title(f"SigLIP: {n_sig} tokens × {d_sig}d    |    4M+MLP: {n_fourm} tokens × {d_fourm}d")

    mismatch = n_sig != n_fourm
    _annotate_danger(ax, mismatch,
                     f"TOKEN COUNT MISMATCH: {n_sig} vs {n_fourm}  ← RoPE positions broken for ALL text tokens"
                     if mismatch else f"Token counts match ({n_sig})")

    plt.tight_layout()
    path = os.path.join(out_dir, "A2_token_count.png")
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"A2 saved → {path}")

    result = {"siglip_n_tokens": n_sig, "fourm_n_tokens": n_fourm,
              "siglip_dim": d_sig, "fourm_dim": d_fourm,
              "token_count_mismatch": mismatch,
              "token_count_ratio": n_sig / max(n_fourm, 1)}
    _save_json(result, out_dir, "A2_token_count.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A3 — SVD spectral analysis (effective rank)
# ─────────────────────────────────────────────────────────────────────────────

def analysis_svd(sig_feats: torch.Tensor, fourm_feats: torch.Tensor, out_dir: str) -> dict:
    """
    Compute SVD of the token matrices and compare singular value spectra.

    WHY: SigLIP was trained to extract diverse, high-rank representations.
    If 4M+MLP collapses to a low-rank subspace (few dominant singular values),
    the policy cannot extract the fine-grained spatial cues it needs for manipulation.
    Effective rank measures how many 'independent dimensions' carry information.
    """
    # Reshape to (N*N_tokens, D) and compute SVD
    sig_mat   = sig_feats.reshape(-1, sig_feats.shape[-1]).float()
    fourm_mat = fourm_feats.reshape(-1, fourm_feats.shape[-1]).float()

    # Center the matrices
    sig_mat   = sig_mat   - sig_mat.mean(0, keepdim=True)
    fourm_mat = fourm_mat - fourm_mat.mean(0, keepdim=True)

    # Top-100 singular values (efficient)
    k = min(100, sig_mat.shape[0], sig_mat.shape[1],
            fourm_mat.shape[0], fourm_mat.shape[1])
    _, s_sig,  _ = torch.linalg.svd(sig_mat[:2000],   full_matrices=False)
    _, s_fourm,_ = torch.linalg.svd(fourm_mat[:2000], full_matrices=False)
    s_sig   = s_sig[:k].numpy()
    s_fourm = s_fourm[:k].numpy()

    def effective_rank(s):
        p = s / s.sum()
        return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    er_sig   = effective_rank(s_sig)
    er_fourm = effective_rank(s_fourm)

    # Explained variance
    var_sig   = (s_sig**2).cumsum()   / (s_sig**2).sum()
    var_fourm = (s_fourm**2).cumsum() / (s_fourm**2).sum()
    d90_sig   = int(np.searchsorted(var_sig,   0.9)) + 1
    d90_fourm = int(np.searchsorted(var_fourm, 0.9)) + 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("A3 — SVD Spectral Analysis (Effective Rank)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(s_sig   / s_sig[0],   label=f"SigLIP  (eff.rank={er_sig:.1f})",   color="#2196F3", lw=2)
    ax.plot(s_fourm / s_fourm[0], label=f"4M+MLP  (eff.rank={er_fourm:.1f})", color="#F44336", lw=2)
    ax.set_xlabel("Singular value rank"); ax.set_ylabel("Normalized singular value")
    ax.set_title("Singular value decay"); ax.legend(); ax.set_yscale("log")

    ax = axes[1]
    ax.plot(range(1, k+1), var_sig,   label=f"SigLIP  (90% @ {d90_sig} dims)",   color="#2196F3", lw=2)
    ax.plot(range(1, k+1), var_fourm, label=f"4M+MLP  (90% @ {d90_fourm} dims)", color="#F44336", lw=2)
    ax.axhline(0.9, color="gray", ls="--", lw=1)
    ax.set_xlabel("Number of components"); ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Explained variance curve"); ax.legend()

    ax = axes[2]
    bars = ax.bar(["SigLIP", "4M+MLP"], [er_sig, er_fourm],
                  color=["#2196F3", "#F44336"], alpha=0.85)
    ax.bar_label(bars, fmt="%.1f"); ax.set_ylabel("Effective rank")
    ax.set_title(f"Effective rank  (ratio = {er_fourm/max(er_sig,1e-6):.2f}x)")
    _annotate_danger(ax, er_fourm < 0.5 * er_sig,
                     f"RANK COLLAPSE: 4M+MLP rank {er_fourm:.1f} vs SigLIP {er_sig:.1f}"
                     if er_fourm < 0.5 * er_sig else "Rank comparable")

    plt.tight_layout()
    path = os.path.join(out_dir, "A3_svd_spectrum.png")
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"A3 saved → {path}")

    result = {"siglip_effective_rank": er_sig, "fourm_effective_rank": er_fourm,
              "siglip_dims_for_90pct_var": d90_sig, "fourm_dims_for_90pct_var": d90_fourm,
              "rank_ratio": er_fourm / max(er_sig, 1e-6)}
    _save_json(result, out_dir, "A3_svd_spectrum.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A4 — Per-dimension alignment
# ─────────────────────────────────────────────────────────────────────────────

def analysis_per_dim_alignment(sig_feats: torch.Tensor, fourm_feats: torch.Tensor,
                                out_dir: str) -> dict:
    """
    For each of the D=2048 dimensions, compute how well SigLIP and 4M+MLP agree.
    Uses Pearson correlation across (N * N_tokens) samples per dimension.

    WHY: Global cosine similarity of 0.78 masks which specific dimensions are well
    aligned vs completely wrong. If the dimensions that SmolLM2 relies most heavily
    on (via Q projection weights) are precisely the misaligned ones, the policy fails
    even with high average cosine similarity.
    """
    N_sig = sig_feats.shape[1]
    N_4m  = fourm_feats.shape[1]
    D     = sig_feats.shape[2]

    # Use only overlapping token positions (min of both counts)
    N_tok = min(N_sig, N_4m)
    sig_2d   = sig_feats[:, :N_tok, :].reshape(-1, D).float()   # (N*N_tok, D)
    fourm_2d = fourm_feats[:, :N_tok, :].reshape(-1, D).float()

    # Pearson correlation per dimension
    sig_c   = sig_2d   - sig_2d.mean(0, keepdim=True)
    fourm_c = fourm_2d - fourm_2d.mean(0, keepdim=True)
    num   = (sig_c * fourm_c).mean(0)
    denom = sig_c.std(0) * fourm_c.std(0) + 1e-8
    corr  = (num / denom).numpy()   # (D,)

    # Global cosine per token
    cos_per_token = F.cosine_similarity(
        sig_feats[:, :N_tok, :].reshape(-1, D),
        fourm_feats[:, :N_tok, :].reshape(-1, D), dim=-1).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("A4 — Per-dimension Alignment (Pearson correlation)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(corr, bins=80, color="#9C27B0", alpha=0.8)
    ax.axvline(corr.mean(), color="red", ls="--", lw=2, label=f"mean={corr.mean():.3f}")
    ax.set_xlabel("Pearson r per dimension"); ax.set_ylabel("# dimensions")
    ax.set_title(f"Correlation histogram  (D={D})")
    ax.legend()
    frac_bad = float((corr < 0.3).mean())
    _annotate_danger(ax, frac_bad > 0.2,
                     f"{frac_bad*100:.0f}% dims have r<0.3 (severely misaligned)")

    ax = axes[1]
    sorted_corr = np.sort(corr)
    ax.plot(sorted_corr, color="#9C27B0", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axhline(0.3, color="orange", ls="--", lw=1, label="r=0.3 threshold")
    ax.fill_between(range(len(sorted_corr)), sorted_corr, 0,
                    where=sorted_corr < 0, alpha=0.3, color="red", label="Anti-correlated dims")
    ax.set_xlabel("Dimension rank (sorted by corr)"); ax.set_ylabel("Pearson r")
    ax.set_title("Sorted correlation profile"); ax.legend()

    ax = axes[2]
    ax.hist(cos_per_token, bins=80, color="#FF9800", alpha=0.8)
    mean_cos = cos_per_token.mean()
    ax.axvline(mean_cos, color="red", ls="--", lw=2, label=f"mean={mean_cos:.3f}")
    ax.set_xlabel("Cosine similarity per token"); ax.set_ylabel("Count")
    ax.set_title(f"Token-level cosine similarity distribution")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "A4_per_dim_alignment.png")
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"A4 saved → {path}")

    result = {"mean_pearson_r": float(corr.mean()), "std_pearson_r": float(corr.std()),
              "frac_dims_below_0.3": frac_bad, "frac_dims_negative": float((corr < 0).mean()),
              "mean_cosine_similarity_per_token": float(mean_cos),
              "n_tokens_compared": N_tok}
    _save_json(result, out_dir, "A4_per_dim_alignment.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A5 — UMAP visualization
# ─────────────────────────────────────────────────────────────────────────────

def analysis_umap(sig_feats: torch.Tensor, fourm_feats: torch.Tensor,
                  frames: torch.Tensor, out_dir: str) -> dict:
    """
    UMAP projection of patch tokens from both encoders, colored by source.

    WHY: Global cosine similarity can be high while the manifold geometry (topology,
    cluster structure) is completely different. SmolLM2 processes tokens in context —
    if nearby tokens in SigLIP space map to distant tokens in 4M space, the
    attention over visual context is disrupted even if individual vectors are close.
    """
    try:
        import umap
    except ImportError:
        log.warning("umap-learn not installed. Running with PCA fallback.")
        umap = None

    N_sig  = sig_feats.shape[1]
    N_4m   = fourm_feats.shape[1]
    D      = sig_feats.shape[2]
    N_frames = min(50, sig_feats.shape[0])  # limit for speed

    # Sample tokens
    sig_sample   = sig_feats[:N_frames].reshape(-1, D).float().numpy()
    fourm_sample = fourm_feats[:N_frames].reshape(-1, D).float().numpy()

    # Sub-sample if too many tokens
    max_pts = 3000
    if len(sig_sample) > max_pts:
        idx = np.random.choice(len(sig_sample), max_pts, replace=False)
        sig_sample = sig_sample[idx]
    if len(fourm_sample) > max_pts:
        idx = np.random.choice(len(fourm_sample), max_pts, replace=False)
        fourm_sample = fourm_sample[idx]

    combined = np.concatenate([sig_sample, fourm_sample], axis=0)
    labels   = np.array([0] * len(sig_sample) + [1] * len(fourm_sample))

    if umap is not None:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        emb = reducer.fit_transform(combined)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        emb = pca.fit_transform(combined)

    emb_sig   = emb[labels == 0]
    emb_fourm = emb[labels == 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    method = "UMAP" if umap is not None else "PCA"
    fig.suptitle(f"A5 — {method} of Patch Tokens", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.scatter(emb_sig[:, 0],   emb_sig[:, 1],   c="#2196F3", alpha=0.3, s=5, label=f"SigLIP ({len(emb_sig)})")
    ax.scatter(emb_fourm[:, 0], emb_fourm[:, 1], c="#F44336", alpha=0.3, s=5, label=f"4M+MLP ({len(emb_fourm)})")
    ax.set_title(f"{method} — Source overlay"); ax.legend(markerscale=3)
    ax.set_xlabel(f"{method}1"); ax.set_ylabel(f"{method}2")

    # Compute overlap metric: fraction of 4M points within SigLIP convex hull area
    ax2 = axes[1]
    # Color by spatial token position (patch index)
    tok_idx_sig   = np.tile(np.arange(len(sig_sample)),   1)[:len(sig_sample)]
    tok_idx_fourm = np.tile(np.arange(len(fourm_sample)), 1)[:len(fourm_sample)]
    # Normalize by respective N_tokens
    tok_frac_sig   = tok_idx_sig   / max(N_sig - 1, 1)
    tok_frac_fourm = tok_idx_fourm / max(N_4m - 1, 1)

    sc1 = ax2.scatter(emb_sig[:, 0],   emb_sig[:, 1],   c=tok_frac_sig,
                      cmap="Blues", alpha=0.4, s=5, marker="o")
    sc2 = ax2.scatter(emb_fourm[:, 0], emb_fourm[:, 1], c=tok_frac_fourm,
                      cmap="Reds",  alpha=0.4, s=5, marker="^")
    ax2.set_title("Colored by spatial token position\n(gradient = position 0→N)")
    ax2.set_xlabel(f"{method}1"); ax2.set_ylabel(f"{method}2")

    plt.tight_layout()
    path = os.path.join(out_dir, f"A5_umap{'_pca' if umap is None else ''}.png")
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"A5 saved → {path}")

    # Overlap metric: do the two clouds overlap?
    from scipy.spatial import ConvexHull
    overlap_est = "n/a"
    try:
        hull_sig = ConvexHull(emb_sig[:500])
        # Count 4M points inside SigLIP hull (approximate via bounding box)
        bbox_min = emb_sig.min(0); bbox_max = emb_sig.max(0)
        inside = ((emb_fourm[:, 0] > bbox_min[0]) & (emb_fourm[:, 0] < bbox_max[0]) &
                  (emb_fourm[:, 1] > bbox_min[1]) & (emb_fourm[:, 1] < bbox_max[1]))
        overlap_est = float(inside.mean())
    except Exception:
        pass

    result = {"method": method, "overlap_bbox_estimate": overlap_est,
              "n_siglip_tokens": len(emb_sig), "n_fourm_tokens": len(emb_fourm)}
    _save_json(result, out_dir, "A5_umap.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A6 — Cosine similarity heatmap across token positions
# ─────────────────────────────────────────────────────────────────────────────

def analysis_cosine_heatmap(sig_feats: torch.Tensor, fourm_feats: torch.Tensor,
                             out_dir: str) -> dict:
    """
    For each spatial token position, compute mean cosine similarity between
    the SigLIP token at that position and ALL 4M+MLP tokens.

    WHY: The MLP connector maps each 4M token independently. But even with high
    average cosine similarity, if token i in 4M maps to a position that semantically
    corresponds to token j in SigLIP (spatial misalignment), the policy is confused
    about WHICH object/region is at which position. This is especially critical for
    manipulation tasks where spatial precision matters.
    """
    N_tok = min(sig_feats.shape[1], fourm_feats.shape[1])
    N_frames = min(30, sig_feats.shape[0])

    sig_s   = sig_feats[:N_frames, :N_tok, :].float()    # (N, N_tok, D)
    fourm_s = fourm_feats[:N_frames, :N_tok, :].float()

    # (N, N_tok_sig, N_tok_fourm) cosine similarity matrix
    sig_n   = F.normalize(sig_s,   dim=-1)
    fourm_n = F.normalize(fourm_s, dim=-1)
    cosine_matrix = torch.einsum("nid,njd->nij", sig_n, fourm_n)  # (N, N_tok, N_tok)
    mean_matrix   = cosine_matrix.mean(0).numpy()  # (N_tok, N_tok)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("A6 — Cross-encoder Cosine Similarity Heatmap", fontsize=14, fontweight="bold")

    # Full cosine matrix
    ax = axes[0]
    im = ax.imshow(mean_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("4M+MLP token index"); ax.set_ylabel("SigLIP token index")
    ax.set_title(f"Mean cosine sim  [SigLIP × 4M]  ({N_tok}×{N_tok})")

    # Diagonal = same position alignment
    diag = np.diag(mean_matrix)
    ax = axes[1]
    ax.plot(diag, color="#9C27B0", lw=2)
    ax.axhline(diag.mean(), color="red", ls="--", lw=1, label=f"mean={diag.mean():.3f}")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Token position"); ax.set_ylabel("Cosine similarity")
    ax.set_title("Same-position cosine (diagonal)")
    ax.legend()

    # Best match per SigLIP token
    best_match_sim  = mean_matrix.max(axis=1)   # best cos for each SigLIP token
    best_match_idx  = mean_matrix.argmax(axis=1)
    perfect_spatial = int((best_match_idx == np.arange(N_tok)).sum())
    ax = axes[2]
    ax.scatter(range(N_tok), best_match_idx, c=best_match_sim, cmap="RdYlGn",
               vmin=-1, vmax=1, s=20)
    ax.plot([0, N_tok], [0, N_tok], "k--", lw=1, alpha=0.5, label="Perfect spatial match")
    ax.set_xlabel("SigLIP token index"); ax.set_ylabel("Best-matching 4M token index")
    ax.set_title(f"Spatial matching  ({perfect_spatial}/{N_tok} exact matches)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "A6_cosine_heatmap.png")
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"A6 saved → {path}")

    result = {"mean_diagonal_cosine": float(diag.mean()),
              "std_diagonal_cosine": float(diag.std()),
              "mean_best_match_cosine": float(best_match_sim.mean()),
              "perfect_spatial_matches": perfect_spatial,
              "frac_perfect_spatial": perfect_spatial / max(N_tok, 1)}
    _save_json(result, out_dir, "A6_cosine_heatmap.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A7 — Attention entropy analysis
# ─────────────────────────────────────────────────────────────────────────────

def analysis_attention_entropy(policy_base, vlm, enc, mlp, frames: torch.Tensor,
                                device: str, out_dir: str) -> dict:
    """
    Hook SmolLM2 self-attention, run forward with SigLIP vs 4M+MLP tokens,
    and compare attention entropy on visual token positions.

    WHY: SmolLM2 was trained to attend to SigLIP tokens with specific patterns.
    If 4M tokens cause uniform attention (entropy → max) or single-token attention
    (entropy → 0), the language model is effectively blind — regardless of how
    aligned the feature vectors appear in isolation.

    Entropy: H = -Σ p log p  normalized to [0,1] by dividing by log(N).
    Low entropy = attention collapse (model looks at 1 token).
    High entropy = uniform attention (model looks at nothing specific).
    """
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    from transformers import AutoTokenizer

    # Dummy task description to build a proper language batch
    task = "put the object in the basket"
    try:
        tok = AutoTokenizer.from_pretrained("lerobot/smolvla_libero")
    except Exception:
        log.warning("A7 skipped: could not load tokenizer")
        return {"skipped": True}

    enc_out = tok(task, return_tensors="pt", padding="max_length",
                  max_length=48, truncation=True)
    lang_tokens = enc_out.input_ids.to(device)
    lang_mask   = enc_out.attention_mask.bool().to(device)

    attention_weights = {"siglip": [], "fourm": []}

    def _make_hook(key):
        def hook(module, inp, out):
            # out may be (attn_output, attn_weights) or just attn_output
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attention_weights[key].append(out[1].detach().cpu())
        return hook

    # Find self-attention modules in SmolLM2
    smollm = vlm.get_vlm_model() if hasattr(vlm, "get_vlm_model") else vlm.language_model
    hooks = []
    for name, mod in smollm.named_modules():
        if any(t in type(mod).__name__.lower() for t in ("attention", "selfattn", "multihead")):
            hooks.append(mod.register_forward_hook(_make_hook("siglip")))

    try:
        with torch.no_grad():
            # SigLIP path (1 frame only for speed)
            frame = frames[0].unsqueeze(0).to(device)
            siglip_vis = vlm.embed_image(
                policy_base.prepare_images({"observation.images.image": frame})[0][0].to(device)
            )  # (1, N_sig, D)
    except Exception as e:
        log.warning(f"A7 SigLIP forward failed: {e}")
        for h in hooks: h.remove()
        return {"skipped": True, "error": str(e)}
    finally:
        for h in hooks: h.remove()

    hooks = []
    for name, mod in smollm.named_modules():
        if any(t in type(mod).__name__.lower() for t in ("attention", "selfattn", "multihead")):
            hooks.append(mod.register_forward_hook(_make_hook("fourm")))
    try:
        with torch.no_grad():
            fourm_tokens = enc({"rgb": frame})
            fourm_tokens = fourm_tokens.to(mlp.mlp[0].weight.dtype)
            fourm_vis = mlp(fourm_tokens)
    except Exception as e:
        log.warning(f"A7 4M forward failed: {e}")
        for h in hooks: h.remove()
        return {"skipped": True, "error": str(e)}
    finally:
        for h in hooks: h.remove()

    # If no attention weights were captured, attention is fused into output
    if not attention_weights["siglip"] or not attention_weights["fourm"]:
        log.warning("A7: no attention weights captured (attention_output_attentions may be False). "
                    "This analysis requires output_attentions=True in the model config.")
        _save_json({"skipped": True, "reason": "attention_weights_not_accessible"}, out_dir, "A7_attention_entropy.json")
        return {"skipped": True}

    def entropy_stats(attn_list):
        entropies = []
        for w in attn_list:  # (B, heads, seq, seq)
            eps = 1e-9
            h = -(w * (w + eps).log()).sum(-1)          # (B, heads, seq)
            h_norm = h / np.log(w.shape[-1] + eps)     # normalize to [0,1]
            entropies.append(h_norm.mean().item())
        return np.array(entropies)

    ent_sig   = entropy_stats(attention_weights["siglip"])
    ent_fourm = entropy_stats(attention_weights["fourm"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("A7 — Attention Entropy in SmolLM2", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(ent_sig,   color="#2196F3", lw=2, label=f"SigLIP (mean={ent_sig.mean():.3f})")
    ax.plot(ent_fourm, color="#F44336", lw=2, label=f"4M+MLP (mean={ent_fourm.mean():.3f})")
    ax.set_xlabel("Layer index"); ax.set_ylabel("Normalized attention entropy")
    ax.set_title("Attention entropy per layer"); ax.legend()
    ax.axhline(0.0, color="gray", ls=":", lw=1, label="collapse (0)")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="uniform (1)")

    ax = axes[1]
    ax.bar(["SigLIP", "4M+MLP"],
           [ent_sig.mean(), ent_fourm.mean()],
           color=["#2196F3", "#F44336"], alpha=0.85)
    ax.set_ylabel("Mean normalized entropy")
    ax.set_title("Overall attention entropy comparison")
    delta = abs(ent_sig.mean() - ent_fourm.mean())
    _annotate_danger(ax, delta > 0.15,
                     f"Entropy differs by {delta:.3f} — attention pattern disrupted"
                     if delta > 0.15 else "Entropy comparable")

    plt.tight_layout()
    path = os.path.join(out_dir, "A7_attention_entropy.png")
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"A7 saved → {path}")

    result = {"siglip_mean_entropy": float(ent_sig.mean()),
              "fourm_mean_entropy": float(ent_fourm.mean()),
              "entropy_delta": float(delta),
              "flag_disrupted": bool(delta > 0.15)}
    _save_json(result, out_dir, "A7_attention_entropy.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# A8 — Temporal drift
# ─────────────────────────────────────────────────────────────────────────────

def analysis_temporal_drift(sig_feats: torch.Tensor, fourm_feats: torch.Tensor,
                             out_dir: str) -> dict:
    """
    Measure how much embeddings change between consecutive frames.
    Uses the dataset order as a proxy for temporal ordering.

    WHY: SmolLM2 maintains temporal context across steps in a trajectory.
    If 4M features jump drastically between consecutive frames (high temporal variance)
    while SigLIP features change smoothly, the policy sees an incoherent temporal
    signal and produces erratic / repetitive actions.
    """
    N = min(sig_feats.shape[0] - 1, 100)
    sig_mean   = sig_feats[:N+1].mean(dim=1)   # (N+1, D) — mean over tokens
    fourm_mean = fourm_feats[:N+1].mean(dim=1)

    sig_delta   = (sig_mean[1:] - sig_mean[:-1]).norm(dim=-1).numpy()    # (N,)
    fourm_delta = (fourm_mean[1:] - fourm_mean[:-1]).norm(dim=-1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("A8 — Temporal Embedding Drift", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(sig_delta,   color="#2196F3", alpha=0.7, lw=1, label=f"SigLIP (mean={sig_delta.mean():.3f})")
    ax.plot(fourm_delta, color="#F44336", alpha=0.7, lw=1, label=f"4M+MLP (mean={fourm_delta.mean():.3f})")
    ax.set_xlabel("Frame step"); ax.set_ylabel("||f(t+1) - f(t)||  (mean over tokens)")
    ax.set_title("Frame-to-frame embedding change"); ax.legend()

    ax = axes[1]
    ratio_drift = fourm_delta.mean() / max(sig_delta.mean(), 1e-8)
    ax.bar(["SigLIP", "4M+MLP"],
           [sig_delta.mean(), fourm_delta.mean()],
           color=["#2196F3", "#F44336"], alpha=0.85,
           yerr=[sig_delta.std(), fourm_delta.std()], capsize=6)
    ax.set_ylabel("Mean Δ between frames"); ax.set_title(f"Temporal drift (ratio={ratio_drift:.2f}x)")
    _annotate_danger(ax, ratio_drift > 2.0,
                     f"4M drift {ratio_drift:.1f}x HIGHER → temporal incoherence"
                     if ratio_drift > 2.0 else "Drift comparable")

    plt.tight_layout()
    path = os.path.join(out_dir, "A8_temporal_drift.png")
    plt.savefig(path, dpi=150); plt.close()
    log.info(f"A8 saved → {path}")

    result = {"siglip_mean_drift": float(sig_delta.mean()), "fourm_mean_drift": float(fourm_delta.mean()),
              "drift_ratio_4m_over_siglip": float(ratio_drift), "flag_incoherent": bool(ratio_drift > 2.0)}
    _save_json(result, out_dir, "A8_temporal_drift.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(results: dict, out_dir: str):
    lines = ["=" * 60, "ENCODER FAILURE ANALYSIS — SUMMARY", "=" * 60, ""]

    checks = {
        "A1 Token norms":        ("A1", "flag_distortion",         "Attention distortion likely"),
        "A2 Token count":        ("A2", "token_count_mismatch",    "RoPE position shift"),
        "A3 SVD rank":           ("A3", lambda r: r.get("rank_ratio", 1) < 0.5, "Rank collapse"),
        "A4 Dim alignment":      ("A4", lambda r: r.get("frac_dims_below_0.3", 0) > 0.2, "Severe dim misalignment"),
        "A7 Attention entropy":  ("A7", "flag_disrupted",          "Attention disrupted"),
        "A8 Temporal drift":     ("A8", "flag_incoherent",         "Temporal incoherence"),
    }

    flags_fired = []
    for label, (key, flag, msg) in checks.items():
        r = results.get(key, {})
        if r.get("skipped"):
            lines.append(f"  {label:30s}  SKIPPED")
            continue
        fired = flag(r) if callable(flag) else r.get(flag, False)
        status = "⚠  PROBLEM DETECTED" if fired else "✓  OK"
        lines.append(f"  {label:30s}  {status}" + (f"  ← {msg}" if fired else ""))
        if fired:
            flags_fired.append(msg)

    lines += ["", "─" * 60, "KEY NUMBERS", "─" * 60]
    for key in ["A1", "A2", "A3", "A4", "A6", "A7", "A8"]:
        r = results.get(key, {})
        if r and not r.get("skipped"):
            lines.append(f"\n  [{key}]")
            for k, v in r.items():
                if not k.startswith("flag"):
                    lines.append(f"    {k}: {v}")

    lines += ["", "=" * 60,
              f"  PROBLEMS DETECTED: {len(flags_fired)}",
              *[f"    • {f}" for f in flags_fired], "=" * 60]

    report = "\n".join(lines)
    path = os.path.join(out_dir, "SUMMARY.txt")
    with open(path, "w") as f:
        f.write(report)
    print(report)
    log.info(f"Summary → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_danger(ax, condition: bool, msg: str):
    color = "#D32F2F" if condition else "#388E3C"
    ax.annotate(msg, xy=(0.5, 0.02), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))


def _save_json(data: dict, out_dir: str, name: str):
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, bool): return bool(obj)
        return obj
    clean = {k: _convert(v) for k, v in data.items()}
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(clean, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",         required=True,  help="Path to libero_thermal train parquet dir")
    p.add_argument("--checkpoint",      default=None,   help="Path to block2_final.pt (trained MLP weights)")
    p.add_argument("--smolvla",         default="lerobot/smolvla_libero")
    p.add_argument("--fourm_model",     default="EPFL-VILAB/4M-21_XL")
    p.add_argument("--fourm_dim",       type=int, default=1024)
    p.add_argument("--n_frames",        type=int, default=200)
    p.add_argument("--output_dir",      default="analysis_results/encoder_failure")
    p.add_argument("--device",          default=None)
    p.add_argument("--skip",            nargs="*", default=[],
                   help="Analyses to skip, e.g. --skip A5 A7  (A7 most likely to fail)")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    frames = load_frames(args.parquet, args.n_frames)
    log.info(f"Loaded {len(frames)} frames  {frames.shape}")

    # ── Load models ──────────────────────────────────────────────────────────
    policy_base, vlm, img_keys = load_siglip_encoder(args.smolvla, device)
    enc, mlp = load_fourm_encoder(args.fourm_model, args.fourm_dim, args.checkpoint, device)

    # ── Extract features ─────────────────────────────────────────────────────
    log.info("Extracting SigLIP features ...")
    sig_feats = extract_siglip_features(policy_base, vlm, img_keys, frames, device)
    log.info(f"SigLIP features: {sig_feats.shape}")

    log.info("Extracting 4M+MLP features ...")
    fourm_feats = extract_fourm_features(enc, mlp, frames, device)
    log.info(f"4M+MLP features: {fourm_feats.shape}")

    # ── Run analyses ─────────────────────────────────────────────────────────
    skip = set(args.skip)
    results = {}

    if "A1" not in skip:
        log.info("Running A1: token norm distributions ...")
        results["A1"] = analysis_token_norms(sig_feats, fourm_feats, args.output_dir)

    if "A2" not in skip:
        log.info("Running A2: token count & shape ...")
        results["A2"] = analysis_token_count(sig_feats, fourm_feats, args.output_dir)

    if "A3" not in skip:
        log.info("Running A3: SVD spectrum ...")
        results["A3"] = analysis_svd(sig_feats, fourm_feats, args.output_dir)

    if "A4" not in skip:
        log.info("Running A4: per-dimension alignment ...")
        results["A4"] = analysis_per_dim_alignment(sig_feats, fourm_feats, args.output_dir)

    if "A5" not in skip:
        log.info("Running A5: UMAP ...")
        results["A5"] = analysis_umap(sig_feats, fourm_feats, frames, args.output_dir)

    if "A6" not in skip:
        log.info("Running A6: cosine heatmap ...")
        results["A6"] = analysis_cosine_heatmap(sig_feats, fourm_feats, args.output_dir)

    if "A7" not in skip:
        log.info("Running A7: attention entropy ...")
        results["A7"] = analysis_attention_entropy(
            policy_base, vlm, enc, mlp, frames, device, args.output_dir)

    if "A8" not in skip:
        log.info("Running A8: temporal drift ...")
        results["A8"] = analysis_temporal_drift(sig_feats, fourm_feats, args.output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(results, args.output_dir)
    log.info(f"All results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
