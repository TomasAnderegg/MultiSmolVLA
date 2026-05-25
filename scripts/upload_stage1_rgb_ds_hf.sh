#!/bin/bash
# Upload Stage1 RGB+depth+seg per-token distillation checkpoint to HuggingFace
set -e

export HF_HOME=/scratch/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/hf_cache/hub
export PYTHONPATH=/home/garate/MultiSmolVLA:/home/garate/MultiSmolVLA/third_party/lerobot/src:$PYTHONPATH

CKPT="/scratch/checkpoints/stage1_rgb_ds/pipeline_final.pt"
REPO="TomasAnderegg/multismolvla-stage1-rgb-ds-pertok"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

echo "Uploading $CKPT → $REPO ..."
echo "File size: $(du -sh $CKPT | cut -f1)"

python - <<EOF
from huggingface_hub import HfApi
import os

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

repo_id = "$REPO"
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
print(f"Repo ready: https://huggingface.co/{repo_id}")

api.upload_file(
    path_or_fileobj="$CKPT",
    path_in_repo="pipeline_final.pt",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Stage1 RGB+depth+seg per-token distillation — 20k steps, cos_sim=0.779",
)

readme = """---
license: apache-2.0
tags:
  - robotics
  - manipulation
  - smolvla
  - libero
---

# MultiSmolVLA — Stage 1 RGB+depth+seg per-token distillation

MLP connector trained to align 4M-21 (XL) visual tokens with SigLIP token space,
with PatchEmbedder residuals for depth and segmentation modalities.

## Training details
- **Loss**: per-token cosine embedding loss (B×196 tokens flattened)
- **Encoder**: 4M-21 XL (frozen), RGB + depth + seg via PatchEmbedder residuals
- **Steps**: 20 000
- **Batch size**: 16
- **LR MLP**: 1e-3
- **Final cos_sim**: 0.779
- **Final mlp_norm**: 23.3 (siglip_norm: 36) ✅ stable
- **Dataset**: TomasAnderegg/libero_10_thermal (LIBERO-10, 3 tasks)

## Architecture
Depth and segmentation are injected as additive residuals to 4M RGB tokens via
learned PatchEmbedders (patch_size=16, (B,1,224,224) → (B,196,D)):
```python
tokens = fourm_rgb_tokens + depth_embed(depth) + seg_embed(seg)
```

## Usage
```python
import torch
from src.pipeline.full_pipeline import VLAPipeline

pipeline = VLAPipeline(use_4m=True, skip_block1=True, use_depth=True, use_seg=True, device="cuda")
state = torch.load("pipeline_final.pt", map_location="cpu")
pipeline.load_state_dict(state, strict=False)
```
"""
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Add model card",
)

print(f"Done! https://huggingface.co/{repo_id}")
EOF
