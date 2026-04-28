#!/bin/bash
#SBATCH --job-name=test_block2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_block2_%j.out

nvidia-smi

eval "$(conda shell.bash hook)"
conda activate $SCRATCH/envs/multismolvla

cd /home/garate/MultiSmolVLA

python scripts/test_pipeline.py
