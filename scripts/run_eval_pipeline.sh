#!/bin/bash
#SBATCH --job-name=eval_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/eval_%j.out

mkdir -p /home/garate/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

export PYTHONPATH="/scratch/izar/garate/LIBERO:$PYTHONPATH"
export ROBOSUITE_LOG_FILE="/scratch/izar/$USER/robosuite.log"
export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

echo "Job started at $(date)"
nvidia-smi

# ── configure ────────────────────────────────────────────────────────────────
CHECKPOINT="/scratch/izar/$USER/checkpoints/stage2_finetune_v2/pipeline_step95000.pt"
TASK="libero_10"
N_EPISODES=5
OUTPUT_DIR="/home/garate/MultiSmolVLA/eval_results/multismolvla_${TASK}_stage2v2"

# ─────────────────────────────────────────────────────────────────────────────

# Download Florian's checkpoint from HuggingFace if not already present
if [ -n "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT" ]; then
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

CHECKPOINT_ARG=()
[[ -n "$CHECKPOINT" ]] && CHECKPOINT_ARG=(--checkpoint "$CHECKPOINT")

python scripts/eval_pipeline.py \
    "${CHECKPOINT_ARG[@]}"                    \
    --fourm_checkpoint EPFL-VILAB/4M-21_XL   \
    --fourm_dim     1024                      \
    --task          "$TASK"                   \
    --n_episodes    "$N_EPISODES"             \
    --batch_size    1                         \
    --output_dir    "$OUTPUT_DIR"

echo "Job finished at $(date)"
