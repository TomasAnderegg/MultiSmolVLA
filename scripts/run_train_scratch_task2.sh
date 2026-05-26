#!/bin/bash
#SBATCH --job-name=scratch_task2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/apapadat/MultiSmolVLA/logs/scratch_task2_%j.out

# From-scratch training: random action head, pretrained SmolVLM2 backbone.
# Mirrors baseline lerobot approach (load_vlm_weights=True, no pretrained action head).
# MLP + action head unfrozen. ThermalGen, ImageBind, 4M frozen. Task 2 only (libero_10).

mkdir -p /home/apapadat/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate lerobot

cd /home/apapadat/MultiSmolVLA

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
OUTPUT_DIR="/scratch/izar/$USER/checkpoints/scratch_task2_e2e"
mkdir -p "$OUTPUT_DIR"

echo "Job started at $(date)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir           "$DATA_DIR"      \
    --output_dir         "$OUTPUT_DIR"    \
    --from_scratch                        \
    --freeze_thermalgen                   \
    --freeze_imagebind                    \
    --freeze_4m                           \
    --task_ids           2                \
    --bf16                                \
    --batch_size         4                \
    --steps              20000            \
    --lr                 5e-6             \
    --lr_mlp             5e-5             \
    --lr_action          1e-4             \
    --num_workers        8                \
    --log_every          100              \
    --save_every         4000             \
    --wandb                               \
    --wandb_project      multismolvla     \
    --wandb_run_name     scratch_task2_e2e \
    --p_drop             0.0

echo "Job finished at $(date)"
