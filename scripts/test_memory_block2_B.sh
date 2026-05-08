#!/bin/bash
#SBATCH --job-name=test_mem_b2_B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/test_memory_block2_B_%j.out

eval "$(conda shell.bash hook)"
conda activate lerobot

cd /home/apapadat/MultiSmolVLA

python scripts/test_block2.py --fourm_model B
