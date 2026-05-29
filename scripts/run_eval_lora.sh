#!/bin/bash
set -e

if [ ! -d /scratch/LIBERO ]; then
    echo "[setup] Cloning LIBERO..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /scratch/LIBERO
else
    echo "[setup] LIBERO already present"
fi

export PYTHONPATH="/home/garate/MultiSmolVLA/third_party/lerobot/src:/home/garate/MultiSmolVLA:/scratch/LIBERO:$PYTHONPATH"

cd /home/garate/MultiSmolVLA

python -u scripts/eval_native_smolvla_v2.py \
    --use_fourm \
    --use_tok_encoder \
    --lora_checkpoint /scratch/finetune_4m_lora/lora_step001500.pt \
    --maskgit_steps 4 \
    --tok_temperature 0.0 \
    --n_episodes 5 \
    2>&1 | tee /home/garate/MultiSmolVLA/logs/eval_lora_1500.log
