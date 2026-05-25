#!/bin/bash
# Upload Stage1 RGB per-token distillation checkpoint to HuggingFace
# Run as a RunAI job (see command below) or interactive session with HF_TOKEN set.
#
# RunAI submit:
#   runai submit upload-stage1-rgb \
#     -p course-ee-559-project01-garate \
#     --image registry.rcp.epfl.ch/ee-559-garate/multismolvla:latest \
#     --run-as-uid 214984 --run-as-gid 30264 \
#     --node-pools a100-40g --gpu 0 \
#     --pvc home:/home/garate \
#     --pvc course-ee-559-project01-scratch:/scratch \
#     -e HF_TOKEN=<your_token> \
#     --command -- bash /home/garate/MultiSmolVLA/scripts/upload_stage1_rgb_hf.sh

set -e

export HF_HOME=/scratch/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/hf_cache/hub
export PYTHONPATH=/home/garate/MultiSmolVLA:/home/garate/MultiSmolVLA/third_party/lerobot/src:$PYTHONPATH

CKPT="/scratch/checkpoints/stage1_rgb/pipeline_final.pt"
REPO="TomasAnderegg/multismolvla-stage1-rgb-pertok"

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
    commit_message="Stage1 RGB per-token distillation — 20k steps, cos_sim=0.784",
)

# Upload a model card
readme = """---
license: apache-2.0
tags:
  - robotics
  - manipulation
  - smolvla
  - libero
---

# MultiSmolVLA — Stage 1 RGB per-token distillation

MLP connector trained to align 4M-21 (XL) visual tokens with SigLIP token space,
using per-token cosine embedding loss over all B×196 tokens.

## Training details
- **Loss**: per-token cosine embedding loss (B×196 tokens flattened)
- **Encoder**: 4M-21 XL (frozen), RGB only, no Block1
- **Steps**: 20 000
- **Batch size**: 16
- **LR MLP**: 1e-3
- **Final cos_sim**: 0.784
- **Final mlp_norm**: 59.6 (siglip_norm: 36)
- **Dataset**: TomasAnderegg/libero_10_thermal (LIBERO-10, 3 tasks)

## Evaluation on LIBERO-10
- **Success rate**: 0% (all 10 tasks)
- **Conclusion**: cos_sim=0.784 is insufficient; structural incompatibility between
  4M and SigLIP spaces cannot be bridged by a 2-layer MLP without end-to-end fine-tuning.

## Usage
```python
import torch
from src.pipeline.full_pipeline import VLAPipeline

pipeline = VLAPipeline(use_4m=True, skip_block1=True, device="cuda")
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
