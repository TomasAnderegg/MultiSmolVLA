#!/bin/bash
#SBATCH --job-name=thermal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3          # 4 jobs paralleles (modifier selon le nb de shards)
#SBATCH --output=logs/thermal_%A_%a.out

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

echo "Job $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
nvidia-smi

python utils/thermal_pipeline.py \
    --job_id $SLURM_ARRAY_TASK_ID \
    --n_jobs $SLURM_ARRAY_TASK_COUNT \
    --batch_size 16
