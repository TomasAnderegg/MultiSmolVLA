#!/bin/bash
#SBATCH --job-name=stage1b_dropout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/stage1b_%j.out

# Stage 1b: MLP-only training with modality dropout.
#
# Goal: make the MLP robust to missing sensors (depth, seg, thermal).
# Strategy: freeze everything except the MLP, train with L_distill only.
# With p_drop=0.5, half of batches have random modalities zeroed out.
# The MLP learns: "even without depth/seg, produce tokens close to SigLIP".
# No L_action => no gripper collapse possible.

mkdir -p /home/garate/MultiSmolVLA/logs

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

cd /home/garate/MultiSmolVLA

export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

DATA_DIR="/scratch/izar/$USER/data/parquet_thermal"
OUTPUT_DIR="/scratch/izar/$USER/checkpoints/stage1b_dropout"
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
    --freeze_smolvlm                      \
    --freeze_action_expert                \
    --distill                             \
    --p_drop             0.5             \
    --alpha_min          0.0             \
    --total_epochs       50000           \
    --batch_size         4               \
    --steps              50000           \
    --lr                 0.0             \
    --lr_mlp             5e-5            \
    --num_workers        4               \
    --log_every          100             \
    --save_every         5000            \
    --wandb                              \
    --wandb_project      multismolvla    \
    --wandb_run_name     stage1b_dropout

echo "Job finished at $(date)"
