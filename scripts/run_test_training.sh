#!/bin/bash
#SBATCH --job-name=test_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/train_%j.out

mkdir -p logs

eval "$(conda shell.bash hook)"
#conda activate lerobot
conda activate /scratch/izar/$USER/envs/multismolvla
#cd /home/apapadat/MultiSmolVLA
cd /home/garate/MultiSmolVLA

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
OUTPUT_DIR="/scratch/izar/$USER/checkpoints/full_pipeline_alignment_mlp"
mkdir -p "$OUTPUT_DIR"

echo "Job started at $(date)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir        "$DATA_DIR"     \
    --output_dir      "$OUTPUT_DIR"   \
    --freeze_thermalgen               \
    --freeze_imagebind                \
    --freeze_4m                       \
    --freeze_smolvlm                  \
    --freeze_action_expert            \
    --alpha_min       1.0             \
    --batch_size      8               \
    --steps           10000           \
    --num_workers     4               \
    --lr              1e-3            \
    --log_every       100             \
    --save_every      2000            \
    --wandb                           \
    --wandb_project   multismolvla    \
    --wandb_run_name  stage1_mlp_alignment

echo "Job finished at $(date)"
