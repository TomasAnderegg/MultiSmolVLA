#!/bin/bash
#SBATCH --job-name=test_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_pipeline_%j.out

nvidia-smi

eval "$(conda shell.bash hook)"
conda activate lerobot

cd /home/apapadat/MultiSmolVLA

python scripts/test_pipeline.py --fourm_model XL
