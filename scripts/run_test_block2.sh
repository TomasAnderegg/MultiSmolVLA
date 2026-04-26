#!/bin/bash
#SBATCH --job-name=test_block2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_block2_%j.out

eval "$(conda shell.bash hook)"
conda activate lerobot

cd /home/apapadat/MultiSmolVLA

python scripts/test_block2.py
