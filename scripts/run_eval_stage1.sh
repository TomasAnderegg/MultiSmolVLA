#!/bin/bash
#SBATCH --job-name=eval_stage1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/eval_stage1_%j.out

mkdir -p /home/garate/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

export PYTHONPATH="/scratch/izar/garate/LIBERO:$PYTHONPATH"
export ROBOSUITE_LOG_FILE="/scratch/izar/$USER/robosuite.log"
export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

echo "Job started at $(date)"
nvidia-smi

CHECKPOINT="/scratch/izar/$USER/checkpoints/distill_mlp/pipeline_step35000.pt"
OUTPUT_DIR="/home/garate/MultiSmolVLA/eval_results/stage1_libero_10"
mkdir -p "$OUTPUT_DIR"

python scripts/eval_pipeline.py \
    --checkpoint    "$CHECKPOINT"            \
    --fourm_checkpoint EPFL-VILAB/4M-21_XL  \
    --fourm_dim     1024                     \
    --task          libero_10                \
    --n_episodes    5                        \
    --batch_size    1                        \
    --output_dir    "$OUTPUT_DIR"

echo "Job finished at $(date)"
