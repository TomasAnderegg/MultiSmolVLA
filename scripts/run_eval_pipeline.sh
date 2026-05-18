#!/bin/bash
#SBATCH --job-name=eval_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/eval_%j.out

mkdir -p /home/garate/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

echo "Job started at $(date)"
nvidia-smi

# ── configure ────────────────────────────────────────────────────────────────
CHECKPOINT_DIR="/scratch/izar/$USER/checkpoints/florian"
CHECKPOINT="$CHECKPOINT_DIR/pipeline_final.pt"
TASK="libero_spatial"  # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
N_EPISODES=10
OUTPUT_DIR="/scratch/izar/$USER/eval_logs/multismolvla_${TASK}"
# ─────────────────────────────────────────────────────────────────────────────

# Download Florian's checkpoint from HuggingFace if not already present
if [ ! -f "$CHECKPOINT" ]; then
    echo "Downloading checkpoint from HuggingFace..."
    python - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="ftanguy/MultiSmolVLA-Robust",
    filename="pipeline_final.pt",
    repo_type="model",
    local_dir="$CHECKPOINT_DIR",
)
print("Download complete.")
EOF
fi

mkdir -p "$OUTPUT_DIR"

python scripts/eval_pipeline.py \
    --checkpoint    "$CHECKPOINT"             \
    --fourm_checkpoint EPFL-VILAB/4M-21_XL   \
    --fourm_dim     1024                      \
    --task          "$TASK"                   \
    --task_ids      0                         \
    --n_episodes    "$N_EPISODES"             \
    --batch_size    1                         \
    --output_dir    "$OUTPUT_DIR"

echo "Job finished at $(date)"
