#!/bin/bash
# Upload distillation checkpoint to HuggingFace
# Run from an interactive job: bash scripts/upload_distill_checkpoint_hf.sh

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

CKPT="/scratch/izar/$USER/checkpoints/stage2_finetune/pipeline_final.pt"
REPO="TomasAnderegg/multismolvla-stage2-finetune"

echo "Uploading $CKPT to $REPO ..."

/scratch/izar/$USER/envs/multismolvla/bin/python - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("$REPO", repo_type="model", private=True, exist_ok=True)
print("Repo ready. Starting upload (27GB, this will take a while)...")
api.upload_file(
    path_or_fileobj="$CKPT",
    path_in_repo="pipeline_final.pt",
    repo_id="$REPO",
    repo_type="model",
)
print("Done! https://huggingface.co/$REPO")
EOF
