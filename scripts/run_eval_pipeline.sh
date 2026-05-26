#!/bin/bash
#SBATCH --job-name=eval_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/home/apapadat/MultiSmolVLA/logs/eval_%j.out

mkdir -p /home/apapadat/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate lerobot

cd /home/apapadat/MultiSmolVLA

export PYTHONPATH="/scratch/izar/apapadat/LIBERO:$PYTHONPATH"
export ROBOSUITE_LOG_FILE="/scratch/izar/$USER/robosuite.log"
export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

echo "Job started at $(date)"
nvidia-smi

# ── configure ────────────────────────────────────────────────────────────────
HF_REPO="TomasAnderegg/multismolvla-stage2-finetune-v2"
HF_FILENAME="pipeline_step30000.pt"
CHECKPOINT="/scratch/izar/$USER/checkpoints/multismolvla-stage2-finetune-v2/${HF_FILENAME}"
TASK="libero_10"
TASK_IDS="0"       # single task — change to run more (e.g. "0 1 2")
N_EPISODES=7
OUTPUT_DIR="/home/apapadat/MultiSmolVLA/eval_results/stage2_finetune_v2_task${TASK_IDS}"

# ─────────────────────────────────────────────────────────────────────────────

# Download checkpoint from HuggingFace if not already present
if [ ! -f "$CHECKPOINT" ]; then
    echo "Downloading checkpoint from HuggingFace (28 GB — this may take a while)..."
    python - <<EOF
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id="${HF_REPO}",
    filename="${HF_FILENAME}",
    repo_type="model",
    local_dir=os.path.dirname("${CHECKPOINT}"),
)
print("Download complete.")
EOF
fi

mkdir -p "$OUTPUT_DIR"

python scripts/eval_pipeline.py \
    --checkpoint        "$CHECKPOINT"              \
    --fourm_checkpoint  EPFL-VILAB/4M-21_XL        \
    --fourm_dim         1024                       \
    --task              "$TASK"                    \
    --task_ids          $TASK_IDS                  \
    --n_episodes        "$N_EPISODES"              \
    --batch_size        1                          \
    --max_episodes_rendered 3                      \
    --output_dir        "$OUTPUT_DIR"

echo "Job finished at $(date)"
