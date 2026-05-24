#!/bin/bash
set -e
cd /home/garate/MultiSmolVLA
python -u scripts/eval_pipeline.py \
    --checkpoint /scratch/checkpoints/stage1b_dropout/pipeline_step50000.pt \
    --task libero_10 \
    --n_episodes 5 \
    --batch_size 1 \
    --output_dir /home/garate/MultiSmolVLA/eval_results/multismolvla_stage1b_dropout 2>&1
