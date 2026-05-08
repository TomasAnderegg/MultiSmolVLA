#!/bin/bash
#SBATCH --job-name=multismolvla
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out

# -------------------------------------------------------------------------
# Usage:
#   sbatch scripts/run_training.sh
#
# Prerequisites:
#   1. Conda env created at /scratch/izar/$USER/envs/multismolvla  (see SETUP_IZAR.md)
#   2. Parquet thermal dataset available at DATA_DIR below
#   3. mkdir -p logs  (done automatically below)
# -------------------------------------------------------------------------

mkdir -p logs checkpoints/full_pipeline

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/$USER/MultiSmolVLA

# Path to the pre-generated parquet_thermal dataset
DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"

# Download dataset from HuggingFace if not already present
if [ ! "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "Dataset not found at $DATA_DIR, downloading from HuggingFace..."
    DATA_DIR_THERMAL=$DATA_DIR python data/download_thermal_dataset.py
else
    echo "Dataset found at $DATA_DIR ($(ls $DATA_DIR | wc -l) shards)"
fi

echo "Job started at $(date)"
echo "Node: $(hostname)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir        "$DATA_DIR"          \
    --freeze_thermalgen                    \
    --freeze_imagebind                     \
    --freeze_4m                            \
    --steps           20000                \
    --batch_size      4                    \
    --lr              1e-5                 \
    --lr_mlp          1e-4                 \
    --save_every      1000                 \
    --log_every       50                   \
    --output_dir      checkpoints/full_pipeline

echo "Job finished at $(date)"
