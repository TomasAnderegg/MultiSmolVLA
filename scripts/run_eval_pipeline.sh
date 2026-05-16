#!/bin/bash
#SBATCH --job-name=eval_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_pipeline_%j.out

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate lerobot

cd /scratch/izar/marferna/MultiSmolVLA

echo "Job started at $(date)"
nvidia-smi

echo "── HF cache check ──────────────────────────────────────────────────────"
ls ~/.cache/huggingface/hub/ 2>/dev/null | grep 4M || echo "WARNING: 4M model not found in HF cache — will attempt download"
echo "────────────────────────────────────────────────────────────────────────"

# ── configure ────────────────────────────────────────────────────────────────
CHECKPOINT=""          # set to e.g. "checkpoints/full_pipeline/pipeline_final.pt", or leave empty for random weights
TASK="libero_spatial"  # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
N_EPISODES=1
OUTPUT_DIR="./eval_logs/multismolvla_${TASK}"
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_ARG=()
[[ -n "$CHECKPOINT" ]] && CHECKPOINT_ARG=(--checkpoint "$CHECKPOINT")

python scripts/eval_pipeline.py \
    "${CHECKPOINT_ARG[@]}"          \
    --fourm_checkpoint EPFL-VILAB/4M-21_B \
    --fourm_dim 768 \
    --task          "$TASK"         \
    --task_ids      0               \
    --n_episodes    "$N_EPISODES"   \
    --batch_size    1               \
    --output_dir    "$OUTPUT_DIR"

echo "Job finished at $(date)"
