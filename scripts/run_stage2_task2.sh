#!/bin/bash
#SBATCH --job-name=stage2_task2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/apapadat/MultiSmolVLA/logs/stage2_task2_%j.out

# End-to-end training: MLP + action head unfrozen, task 2 only (libero_10).
# ThermalGen, ImageBind, 4M remain frozen.
# Task 2: "put the yellow and white mug in the microwave and close it"

mkdir -p /home/apapadat/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate lerobot

cd /home/apapadat/MultiSmolVLA

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
OUTPUT_DIR="/scratch/izar/$USER/checkpoints/stage2_task2_e2e"
RESUME_CKPT="/scratch/izar/$USER/checkpoints/multismolvla-stage2-finetune-v2/pipeline_step30000.pt"
mkdir -p "$OUTPUT_DIR"

echo "Job started at $(date)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir           "$DATA_DIR"      \
    --output_dir         "$OUTPUT_DIR"    \
    --resume_checkpoint  "$RESUME_CKPT"  \
    --freeze_thermalgen                   \
    --freeze_imagebind                    \
    --freeze_4m                           \
    --task_ids           2                \
    --batch_size         64               \
    --steps              20000            \
    --lr                 5e-6             \
    --lr_mlp             5e-5             \
    --lr_action          1e-4             \
    --num_workers        8                \
    --log_every          100              \
    --save_every         4000             \
    --wandb                               \
    --wandb_project      multismolvla     \
    --wandb_run_name     stage2_task2_e2e

echo "Job finished at $(date)"
