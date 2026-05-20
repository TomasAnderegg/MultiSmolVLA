#!/bin/bash
#SBATCH --job-name=distill_mlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/distill_%j.out

mkdir -p /home/garate/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
OUTPUT_DIR="/scratch/izar/$USER/checkpoints/distill_mlp"
mkdir -p "$OUTPUT_DIR"

echo "Job started at $(date)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir          "$DATA_DIR"   \
    --output_dir        "$OUTPUT_DIR" \
    --distill                         \
    --freeze_thermalgen               \
    --freeze_imagebind                \
    --freeze_4m                       \
    --freeze_smolvlm                  \
    --freeze_action_expert            \
    --alpha_min         1.0           \
    --batch_size        8             \
    --steps             50000         \
    --lr_mlp            1e-3          \
    --num_workers       4             \
    --log_every         100           \
    --save_every        5000          \
    --wandb                           \
    --wandb_project     multismolvla  \
    --wandb_run_name    distill_stage1

echo "Job finished at $(date)"
