#!/bin/bash
#SBATCH --job-name=stage2_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/stage2_%j.out

mkdir -p /home/garate/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
OUTPUT_DIR="/scratch/izar/$USER/checkpoints/stage2_finetune"
DISTILL_CKPT="/scratch/izar/$USER/checkpoints/distill_mlp/pipeline_step35000.pt"
mkdir -p "$OUTPUT_DIR"

echo "Job started at $(date)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir           "$DATA_DIR"      \
    --output_dir         "$OUTPUT_DIR"    \
    --resume_checkpoint  "$DISTILL_CKPT"  \
    --freeze_thermalgen                  \
    --freeze_imagebind                   \
    --freeze_4m                          \
    --alpha_min         0.0              \
    --total_epochs      50000            \
    --batch_size        4                \
    --steps             50000            \
    --lr                1e-5             \
    --lr_mlp            1e-4             \
    --num_workers       4                \
    --log_every         100              \
    --save_every        5000             \
    --wandb                              \
    --wandb_project     multismolvla     \
    --wandb_run_name    stage2_finetune

echo "Job finished at $(date)"
