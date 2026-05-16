#!/bin/bash
#SBATCH --job-name=upload_ckpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/upload_ckpt_%j.out

CKPT_DIR="/scratch/izar/garate/checkpoints/full_pipeline_alignment_mlp"
REPO_ID="TomasAnderegg/multismolvla-mlp-alignment"

echo "Upload started at $(date)"

/scratch/izar/garate/envs/multismolvla/bin/python - <<EOF
from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    repo_id="$REPO_ID",
    repo_type="model",
    private=True,
    exist_ok=True,
)
print("Repo ready.")

api.upload_file(
    path_or_fileobj="$CKPT_DIR/pipeline_final.pt",
    path_in_repo="pipeline_final.pt",
    repo_id="$REPO_ID",
    repo_type="model",
)
print("Done!")
EOF

echo "Upload finished at $(date)"
