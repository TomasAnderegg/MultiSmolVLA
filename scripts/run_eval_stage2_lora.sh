#!/bin/bash
set -e
cd /home/garate/MultiSmolVLA

# Find latest checkpoint in stage2_lora
CKPT=$(ls /scratch/checkpoints/stage2_lora/pipeline_step*.pt 2>/dev/null | sort -t p -k3 -n | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found in /scratch/checkpoints/stage2_lora/" >&2
    exit 1
fi
echo "[eval] Using checkpoint: $CKPT"

python -u scripts/eval_pipeline.py \
    --checkpoint "$CKPT" \
    --task libero_10 \
    --n_episodes 5 \
    --batch_size 1 \
    --output_dir /home/garate/MultiSmolVLA/eval_results/multismolvla_stage2_lora 2>&1
