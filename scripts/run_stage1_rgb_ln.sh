#!/bin/bash
# Stage 1 RGB + LayerNorm — test de l'impact de la LayerNorm sur mlp_norm et cos_sim
set -e

export HF_HOME=/scratch/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/hf_cache/hub
export PYTHONPATH=/home/garate/MultiSmolVLA:/home/garate/MultiSmolVLA/third_party/lerobot/src:$PYTHONPATH

cd /home/garate/MultiSmolVLA

DATA_DIR=/scratch/libero_thermal/data/train
if [ ! -f "/scratch/libero_thermal/.done" ]; then
    python -c '
from huggingface_hub import snapshot_download
snapshot_download(
    "TomasAnderegg/libero_10_thermal",
    repo_type="dataset",
    local_dir="/scratch/libero_thermal",
)
'
    touch /scratch/libero_thermal/.done
fi

python -u scripts/train_full_pipeline.py \
    --data_dir "$DATA_DIR" \
    --fourm_model XL \
    --no_block1 \
    --freeze_4m \
    --freeze_smolvlm \
    --freeze_action_expert \
    --distill \
    --bf16 \
    --lr_mlp 1e-3 \
    --lr 1e-4 \
    --batch_size 16 \
    --steps 10000 \
    --save_every 2000 \
    --log_every 50 \
    --num_workers 2 \
    --output_dir /scratch/checkpoints/stage1_rgb_ln \
    --wandb \
    --wandb_project multismolvla \
    --wandb_run_name stage1_rgb_ln
