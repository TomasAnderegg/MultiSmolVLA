#!/bin/bash
#SBATCH --job-name=stage3_joint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/stage3_%j.out

# Stage 3: joint loss (L_action + 0.1 * L_distill) with differentiated learning rates.
#
# Why: Stage 2 collapsed the gripper to always-open (-1) because the flow-matching
# loss converged to the mode of the training data (55.7% open). The action head
# "forgot" the gripper timing from the original SmolVLA weights.
#
# Fix:
#   - Resume from Stage 1 checkpoint (MLP aligned, action head = original SmolVLA)
#   - lr_action = 1e-6  →  action head barely changes (preserves gripper behavior)
#   - lr_mlp    = 5e-5  →  MLP adapts to produce useful visual tokens for actions
#   - lr        = 5e-6  →  SmolVLM backbone (normal)
#   - L_distill keeps MLP aligned with SigLIP while L_action guides toward LIBERO tasks

mkdir -p /home/garate/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
OUTPUT_DIR="/scratch/izar/$USER/checkpoints/stage3_joint"
# Start from Stage 1: MLP well-aligned, action head untouched
STAGE1_CKPT="/scratch/izar/$USER/checkpoints/distill_mlp/pipeline_final.pt"
mkdir -p "$OUTPUT_DIR"

echo "Job started at $(date)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir           "$DATA_DIR"      \
    --output_dir         "$OUTPUT_DIR"    \
    --resume_checkpoint  "$STAGE1_CKPT"  \
    --freeze_thermalgen                   \
    --freeze_imagebind                    \
    --freeze_4m                           \
    --joint                               \
    --lambda_distill     0.1             \
    --alpha_min         0.0               \
    --total_epochs      150000            \
    --batch_size        4                 \
    --steps             100000            \
    --lr                5e-6              \
    --lr_mlp            5e-5              \
    --lr_action         1e-6              \
    --num_workers       4                 \
    --log_every         100               \
    --save_every        5000              \
    --wandb                               \
    --wandb_project     multismolvla      \
    --wandb_run_name    stage3_joint

echo "Job finished at $(date)"
