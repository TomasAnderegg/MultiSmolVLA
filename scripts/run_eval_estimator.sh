#!/bin/bash
# Eval avec depth+seg estimés en temps réel depuis RGB
# + LoRA fine-tuné sur LIBERO (depth+seg→rgb)
# Arg optionnel: p_drop_rgb (ex: 0.0, 0.5, 1.0)
set -e

P_DROP_RGB=${1:-0.0}   # 0.0 = baseline, 1.0 = RGB toujours obstrué
LORA_CKPT=${2:-/scratch/finetune_4m_lora_rgb/lora_step011500.pt}

if [ ! -d /scratch/LIBERO ]; then
    echo "[setup] Cloning LIBERO..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /scratch/LIBERO
else
    echo "[setup] LIBERO already present"
fi

export PYTHONPATH="/home/garate/MultiSmolVLA/third_party/lerobot/src:/home/garate/MultiSmolVLA:/scratch/LIBERO:$PYTHONPATH"

cd /home/garate/MultiSmolVLA

echo "[eval] p_drop_rgb=${P_DROP_RGB}  lora=${LORA_CKPT}"

python -u scripts/eval_native_smolvla_v2.py \
    --use_fourm \
    --use_tok_encoder \
    --use_depth \
    --use_seg \
    --use_estimator \
    --lora_checkpoint "${LORA_CKPT}" \
    --maskgit_steps 4 \
    --tok_temperature 0.0 \
    --p_drop_rgb  "${P_DROP_RGB}" \
    --dropout_alpha 0.0 \
    --n_episodes 5 \
    --output_dir "eval_results/lora_estimator_drop${P_DROP_RGB}" \
    2>&1 | tee "/home/garate/MultiSmolVLA/logs/eval_estimator_drop${P_DROP_RGB}.log"
