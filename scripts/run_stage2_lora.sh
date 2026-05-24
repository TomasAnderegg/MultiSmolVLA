#!/bin/bash
#SBATCH --job-name=stage2_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/stage2_lora_%j.out

# Stage 2 LoRA: fine-tune SmolVLM backbone with LoRA + train MLP with L_action.
#
# Why LoRA prevents gripper collapse:
#   - Original backbone weights are frozen (only small delta adapters train)
#   - Action head is completely frozen (--freeze_action_expert)
#   - Only LoRA adapters (r=16) + MLP learn from L_action
#   - Even with 55.7% gripper-open data, the frozen action head cannot collapse
#
# Compared to Stage 1b (distill only):
#   - Stage 1b: MLP aligned with SigLIP, no task-specific adaptation
#   - Stage 2 LoRA: backbone adapts to LIBERO tasks via LoRA + modality dropout robustness

mkdir -p /home/garate/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
OUTPUT_DIR="/scratch/izar/$USER/checkpoints/stage2_lora"
STAGE1_CKPT="/scratch/izar/$USER/checkpoints/distill_mlp/pipeline_step35000.pt"
mkdir -p "$OUTPUT_DIR"

echo "Job started at $(date)"
nvidia-smi

python scripts/train_full_pipeline.py \
    --data_dir           "$DATA_DIR"      \
    --output_dir         "$OUTPUT_DIR"    \
    --resume_checkpoint  "$STAGE1_CKPT"  \
    --freeze_thermalgen                   \
    --freeze_imagebind                    \
    --freeze_4m                           \
    --freeze_action_expert                \
    --lora                                \
    --lora_r             16              \
    --lora_alpha         32              \
    --lora_dropout       0.05            \
    --p_drop             0.3             \
    --alpha_min          0.0             \
    --total_epochs       50000           \
    --batch_size         4               \
    --steps              50000           \
    --lr                 5e-5            \
    --lr_mlp             5e-5            \
    --num_workers        4               \
    --log_every          100             \
    --save_every         5000            \
    --wandb                              \
    --wandb_project      multismolvla    \
    --wandb_run_name     stage2_lora

echo "Job finished at $(date)"
