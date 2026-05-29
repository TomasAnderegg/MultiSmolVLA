#!/bin/bash
# Eval SmolVLA v2 with MaskGIT iterative decoding (Phase 1 — RGB only)
# Usage (RunAI): see runai submit command below
set -e

export HF_HOME=/scratch/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/hf_cache/hub
export PYTHONPATH=/home/garate/MultiSmolVLA:/home/garate/MultiSmolVLA/third_party/lerobot/src:$PYTHONPATH

cd /home/garate/MultiSmolVLA

python -u scripts/eval_native_smolvla_v2.py \
    --task         libero_10 \
    --task_ids     2 5 \
    --n_episodes   5 \
    --batch_size   1 \
    --output_dir   /scratch/eval_results/smolvla_maskgit_v2 \
    --use_fourm \
    --maskgit_steps  4 \
    --divae_steps    4 \
    --tok_temperature 0.8 \
    2>&1
