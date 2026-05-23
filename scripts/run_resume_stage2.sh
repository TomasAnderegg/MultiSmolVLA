#!/bin/bash
#SBATCH --job-name=stage2_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/stage2_resume_%j.out

# Resume Stage 2 v2 from step 95000 to complete the final 5000 steps.
# Then optionally continue with Stage 3 (joint distill + action loss).

mkdir -p /home/garate/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
OUTPUT_DIR="/scratch/izar/$USER/checkpoints/stage2_finetune_v2"
RESUME_CKPT="/scratch/izar/$USER/checkpoints/stage2_finetune_v2/pipeline_step95000.pt"
mkdir -p "$OUTPUT_DIR"

echo "Job started at $(date)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir           "$DATA_DIR"      \
    --output_dir         "$OUTPUT_DIR"    \
    --resume_checkpoint  "$RESUME_CKPT"   \
    --freeze_thermalgen                   \
    --freeze_imagebind                    \
    --freeze_4m                           \
    --alpha_min         0.0               \
    --total_epochs      150000            \
    --batch_size        4                 \
    --steps             5000              \
    --lr                5e-6              \
    --lr_mlp            5e-5              \
    --num_workers       4                 \
    --log_every         100               \
    --save_every        5000              \
    --wandb                               \
    --wandb_project     multismolvla      \
    --wandb_run_name    stage2_finetune_v2_resume

echo "Job finished at $(date)"
