#!/bin/bash
#SBATCH --job-name=test_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/test_training_%j.out

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate lerobot

cd /home/apapadat/MultiSmolVLA

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"

echo "Job started at $(date)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir        "$DATA_DIR"     \
    --freeze_thermalgen               \
    --freeze_imagebind                \
    --freeze_4m                       \
    --freeze_smolvlm                  \
    --freeze_action_expert            \
    --batch_size      2               \
    --steps           5               \
    --num_workers     0               \
    --log_every       1               \
    --wandb                           \
    --wandb_project   multismolvla    \
    --wandb_run_name  smoke_test

echo "Job finished at $(date)"
