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
    --rgb_passthrough \
    --n_episodes 5 \
    2>&1 | tee /home/garate/MultiSmolVLA/logs/eval_passthrough.log
