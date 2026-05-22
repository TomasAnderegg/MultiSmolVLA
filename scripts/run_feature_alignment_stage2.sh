#!/bin/bash
#SBATCH --job-name=feat_align_s2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/feat_align_stage2_%j.out

mkdir -p /home/garate/MultiSmolVLA/logs
mkdir -p /home/garate/MultiSmolVLA/assets/feature_alignment_stage2

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

echo "Job started at $(date)"

python utils/visualize_feature_alignment.py \
    --data_dir   /scratch/izar/$USER/data/parquet_thermal \
    --n_images   100 \
    --checkpoint /scratch/izar/$USER/checkpoints/stage2_finetune/pipeline_final.pt \
    --output_dir /home/garate/MultiSmolVLA/assets/feature_alignment_stage2

echo "Job finished at $(date)"
