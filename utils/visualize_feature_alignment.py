"""
Feature alignment analysis: SigLIP (vanilla SmolVLA) vs 4M+MLP (our pipeline).

Extracts visual tokens at the point they enter SmolVLM2's transformer for both
pipelines on the same LIBERO images, then computes:
  1. Token norm distributions
  2. Cosine similarity (token-to-token, per image mean)
  3. CKA score (Centered Kernel Alignment)
  4. t-SNE / UMAP 2D visualization

Usage:
    python utils/visualize_feature_alignment.py \
        --data_dir /scratch/izar/garate/data/parquet_thermal \
        --n_images 100 \
        --output_dir eval_results/feature_alignment \
        --checkpoint /scratch/izar/garate/checkpoints/florian/pipeline_final.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "lerobot" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "ThermalGen"))

from utils.parquet_dataset import ParquetThermalDataset
from src.pipeline.full_pipeline import VLAPipeline


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Feature alignment visualisation")
    p.add_argument("--data_dir", type=str,
                   default="/scratch/izar/garate/data/parquet_thermal")
    p.add_argument("--n_images", type=int, default=100,
                   help="Number of images to compare")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to our pipeline .pt checkpoint (None = base weights)")
    p.add_argument("--smolvla_checkpoint", type=str, default="lerobot/smolvla_libero")
    p.add_argument("--fourm_checkpoint", type=str, default="EPFL-VILAB/4M-21_XL")
    p.add_argument("--fourm_dim", type=int, default=1024)
    p.add_argument("--output_dir", type=str,
                   default="eval_results/feature_alignment")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ── CKA ──────────────────────────────────────────────────────────────────────

def centering(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def rbf_kernel(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    sq_dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    if sigma is None:
        sigma = np.sqrt(np.median(sq_dists[sq_dists > 0]))
    return np.exp(-sq_dists / (2 * sigma ** 2))


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between matrices X (N, D1) and Y (N, D2)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XXT = X @ X.T
    YYT = Y @ Y.T
    KX = centering(XXT)
    KY = centering(YYT)
    return float(np.sum(KX * KY) / (np.linalg.norm(KX, "fro") * np.linalg.norm(KY, "fro") + 1e-8))


# ── Token extraction hooks ────────────────────────────────────────────────────

class TokenHook:
    """Captures the output of a module on the next forward pass."""
    def __init__(self):
        self.tokens = None
        self._handle = None

    def attach(self, module: torch.nn.Module):
        def _hook(mod, inp, out):
            self.tokens = out.detach().cpu()
        self._handle = module.register_forward_hook(_hook)

    def remove(self):
        if self._handle:
            self._handle.remove()


def get_vlm_with_expert(pipeline: VLAPipeline):
    """Return the SmolVLMWithExpertModel from our pipeline."""
    return pipeline.block2.smolvla.policy.base_policy.model.vlm_with_expert


def get_siglip_vision_model(pipeline: VLAPipeline):
    """Return the SigLIP vision model: vlm_with_expert.vlm.model.vision_model."""
    vlm_with_expert = get_vlm_with_expert(pipeline)
    return vlm_with_expert.get_vlm_model().vision_model


def get_mlp_connector(pipeline: VLAPipeline):
    """Return the MLP connector (fourm_to_vlm) from our pipeline."""
    vlm_with_expert = get_vlm_with_expert(pipeline)
    return vlm_with_expert.fourm_to_vlm


# ── Data loading ──────────────────────────────────────────────────────────────

def load_images(data_dir: str, n_images: int, device: str) -> torch.Tensor:
    """Load N RGB images from the parquet dataset."""
    dataset = ParquetThermalDataset(data_dir)
    images = []
    indices = np.linspace(0, len(dataset) - 1, n_images, dtype=int)
    for i in indices:
        sample = dataset[i]
        images.append(sample["rgb"])  # (3, 224, 224) [0,1]
    return torch.stack(images).to(device)  # (N, 3, 224, 224)


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_siglip_tokens(policy, images: torch.Tensor, batch_size: int = 8) -> np.ndarray:
    """Extract SigLIP visual tokens from vanilla SmolVLA."""
    hook, _ = get_siglip_hook(policy)
    all_tokens = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        # SmolVLA expects pixel_values in a specific format
        # We call the vision model directly
        try:
            vision_model = policy.model.model.vision_model
            out = vision_model(pixel_values=batch)
            tokens = out.last_hidden_state  # (B, N_patches, D)
        except Exception:
            # Use the hook instead
            _ = hook.tokens  # consume previous
            # Dummy forward to trigger hook
            tokens = hook.tokens
        all_tokens.append(tokens.detach().cpu().numpy() if isinstance(tokens, torch.Tensor)
                         else hook.tokens.numpy())
    hook.remove()
    return np.concatenate(all_tokens, axis=0)  # (N, N_patches, D)


@torch.no_grad()
def extract_mlp_tokens(pipeline: VLAPipeline, images: torch.Tensor,
                       batch_size: int = 8) -> np.ndarray:
    """Extract 4M+MLP tokens by calling encoder and connector directly (bypasses action queue)."""
    vlm_with_expert = get_vlm_with_expert(pipeline)
    fourm_encoder   = vlm_with_expert.fourm_encoder
    mlp_connector   = vlm_with_expert.fourm_to_vlm
    block1          = pipeline.block1

    all_tokens = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        b = len(batch)

        # Block1: ThermalGen → ImageBind (only rgb needed, depth/seg are zeros)
        inputs = {
            "rgb":   batch,
            "depth": torch.zeros(b, 1, 224, 224, device=batch.device),
            "seg":   torch.zeros(b, 1, 224, 224, device=batch.device),
        }
        block1_out = block1(inputs, epoch=0)  # adds thermal embedding, applies dropout (no-op in eval)
        rgb = block1_out["rgb"]               # (B, 3, 224, 224)

        # 4M encoding → MLP projection
        fourm_tokens = fourm_encoder(rgb)               # (B, N_tokens, D_4m)
        mlp_out      = mlp_connector(fourm_tokens)      # (B, N_tokens, D_vlm)
        all_tokens.append(mlp_out.float().cpu().numpy())

    return np.concatenate(all_tokens, axis=0)  # (N, N_tokens, D_vlm)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_norm_distribution(siglip_tokens: np.ndarray, mlp_tokens: np.ndarray,
                           output_dir: str):
    """Plot L2 norm distribution of tokens."""
    siglip_norms = np.linalg.norm(siglip_tokens.reshape(-1, siglip_tokens.shape[-1]), axis=-1)
    mlp_norms    = np.linalg.norm(mlp_tokens.reshape(-1, mlp_tokens.shape[-1]),    axis=-1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(siglip_norms, bins=60, alpha=0.6, color="steelblue",  label=f"SigLIP  (mean={siglip_norms.mean():.2f})")
    ax.hist(mlp_norms,    bins=60, alpha=0.6, color="tomato",     label=f"4M+MLP  (mean={mlp_norms.mean():.2f})")
    ax.set_xlabel("Token L2 norm")
    ax.set_ylabel("Count")
    ax.set_title("Token norm distribution: SigLIP vs 4M+MLP")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "norm_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_cosine_similarity(siglip_tokens: np.ndarray, mlp_tokens: np.ndarray,
                           output_dir: str):
    """Plot per-image mean cosine similarity between corresponding tokens."""
    N = min(len(siglip_tokens), len(mlp_tokens))
    T = min(siglip_tokens.shape[1], mlp_tokens.shape[1])
    D = min(siglip_tokens.shape[2], mlp_tokens.shape[2])
    s = siglip_tokens[:N, :T, :D]
    m = mlp_tokens[:N, :T, :D]

    # Normalize
    s_norm = s / (np.linalg.norm(s, axis=-1, keepdims=True) + 1e-8)
    m_norm = m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-8)

    # Per-image mean cosine similarity
    cos_sims = (s_norm * m_norm).sum(axis=-1).mean(axis=-1)  # (N,)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cos_sims, bins=40, color="mediumpurple", alpha=0.8)
    ax.axvline(cos_sims.mean(), color="black", linestyle="--",
               label=f"Mean = {cos_sims.mean():.3f}")
    ax.set_xlabel("Mean cosine similarity per image")
    ax.set_ylabel("Count")
    ax.set_title("Cosine similarity: SigLIP tokens vs 4M+MLP tokens")
    ax.set_xlim(-1, 1)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "cosine_similarity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"  Mean cosine similarity: {cos_sims.mean():.4f} ± {cos_sims.std():.4f}")
    return float(cos_sims.mean())


def plot_tsne(siglip_tokens: np.ndarray, mlp_tokens: np.ndarray, output_dir: str):
    """2D t-SNE projection of SigLIP vs 4M+MLP tokens (mean-pooled per image)."""
    from sklearn.manifold import TSNE

    # Mean-pool across tokens → one vector per image
    siglip_vecs = siglip_tokens.mean(axis=1)  # (N, D)
    mlp_vecs    = mlp_tokens.mean(axis=1)     # (N, D)

    N = min(len(siglip_vecs), len(mlp_vecs))
    siglip_vecs = siglip_vecs[:N]
    mlp_vecs    = mlp_vecs[:N]

    # Reduce to same dim if different
    D = min(siglip_vecs.shape[-1], mlp_vecs.shape[-1])
    all_vecs = np.concatenate([siglip_vecs[:, :D], mlp_vecs[:, :D]], axis=0)

    print(f"Running t-SNE on {len(all_vecs)} samples...")
    perplexity = min(30, len(all_vecs) // 2 - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    proj = tsne.fit_transform(all_vecs)

    siglip_proj = proj[:N]
    mlp_proj    = proj[N:]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(siglip_proj[:, 0], siglip_proj[:, 1],
               c="steelblue", alpha=0.6, s=20, label="SigLIP (vanilla SmolVLA)")
    ax.scatter(mlp_proj[:, 0],    mlp_proj[:, 1],
               c="tomato",    alpha=0.6, s=20, label="4M+MLP (our pipeline)")
    ax.set_title("t-SNE: visual token space comparison")
    ax.legend()
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(output_dir, "tsne.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load images ──────────────────────────────────────────────────────────
    print(f"Loading {args.n_images} images from {args.data_dir}...")
    images = load_images(args.data_dir, args.n_images, device)
    print(f"  Image batch: {images.shape}")

    # ── Load our pipeline ────────────────────────────────────────────────────
    print("Loading our VLAPipeline (4M + MLP + SmolVLA)...")
    pipeline = VLAPipeline(
        smolvla_checkpoint=args.smolvla_checkpoint,
        fourm_checkpoint=args.fourm_checkpoint,
        fourm_dim=args.fourm_dim,
        use_4m=True, freeze_4m=True, freeze_mlp=False,
        device=device,
    )
    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        pipeline.load_state_dict(state)
    pipeline.to(device)
    pipeline.eval()

    # ── Extract SigLIP tokens ─────────────────────────────────────────────────
    print("Extracting SigLIP tokens (vanilla SigLIP vision encoder)...")
    vision_model = get_siglip_vision_model(pipeline)
    siglip_tokens_list = []
    bs = 8
    siglip_dtype = vision_model.dtype
    with torch.no_grad():
        for i in range(0, len(images), bs):
            batch = images[i:i+bs].to(dtype=siglip_dtype)
            out = vision_model(pixel_values=batch)
            siglip_tokens_list.append(out.last_hidden_state.float().cpu().numpy())
    siglip_tokens = np.concatenate(siglip_tokens_list, axis=0)
    print(f"  SigLIP tokens: {siglip_tokens.shape}")

    # ── Extract 4M+MLP tokens ─────────────────────────────────────────────────
    print("Extracting 4M+MLP tokens (our pipeline)...")
    mlp_tokens = extract_mlp_tokens(pipeline, images, batch_size=4)
    print(f"  4M+MLP tokens: {mlp_tokens.shape}")

    # ── CKA score ─────────────────────────────────────────────────────────────
    print("Computing CKA score...")
    N = min(len(siglip_tokens), len(mlp_tokens))
    siglip_pooled = siglip_tokens[:N].mean(axis=1)
    mlp_pooled    = mlp_tokens[:N].mean(axis=1)
    D = min(siglip_pooled.shape[-1], mlp_pooled.shape[-1])
    cka_score = cka(siglip_pooled[:, :D], mlp_pooled[:, :D])
    print(f"  CKA score: {cka_score:.4f}  (1.0 = identical structure, 0.0 = no alignment)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_norm_distribution(siglip_tokens, mlp_tokens, args.output_dir)
    mean_cos = plot_cosine_similarity(siglip_tokens, mlp_tokens, args.output_dir)
    plot_tsne(siglip_tokens, mlp_tokens, args.output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "n_images":              N,
        "siglip_token_shape":    list(siglip_tokens.shape),
        "mlp_token_shape":       list(mlp_tokens.shape),
        "siglip_norm_mean":      float(np.linalg.norm(siglip_tokens.reshape(-1, siglip_tokens.shape[-1]), axis=-1).mean()),
        "mlp_norm_mean":         float(np.linalg.norm(mlp_tokens.reshape(-1, mlp_tokens.shape[-1]),       axis=-1).mean()),
        "mean_cosine_similarity": mean_cos,
        "cka_score":             cka_score,
    }

    import json
    summary_path = os.path.join(args.output_dir, "alignment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
