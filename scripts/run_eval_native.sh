#!/bin/bash
set -e
cd /home/garate/MultiSmolVLA
python -u scripts/eval_native_smolvla.py \
    --task libero_10 \
    --n_episodes 5 \
    --batch_size 1 \
    --output_dir /home/garate/MultiSmolVLA/eval_results/smolvla_native_libero_10 \
    2>&1
