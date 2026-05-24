#!/bin/bash
# Upload Stage 1 checkpoint (distill_mlp/pipeline_step35000.pt) to HuggingFace
# Usage: bash scripts/upload_stage1_hf.sh
# Requires: huggingface-cli login (or HF_TOKEN set)

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

CKPT="/scratch/izar/$USER/checkpoints/distill_mlp/pipeline_step35000.pt"
REPO="TomasAnderegg/multismolvla-stage1-distill"

echo "Uploading $CKPT to $REPO ..."

/scratch/izar/$USER/envs/multismolvla/bin/python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$REPO", repo_type="model", private=True, exist_ok=True)
print("Repo ready. Starting upload (27 GB) — this will take ~30 min on IZAR...")
api.upload_file(
    path_or_fileobj="$CKPT",
    path_in_repo="pipeline_step35000.pt",
    repo_id="$REPO",
    repo_type="model",
)
print("Done! https://huggingface.co/$REPO")
EOF
