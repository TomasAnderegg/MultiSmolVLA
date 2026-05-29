#!/bin/bash
# Test 3 configurations de modalités:
#   rgb          : RGB seul, passthrough (baseline propre)
#   rgb_depth    : RGB + depth estimé → 4M LoRA
#   rgb_depth_seg: RGB + depth + seg estimés → 4M LoRA
#
# Usage: bash run_eval_modalities.sh <mode>
# mode: rgb | rgb_depth | rgb_depth_seg
set -e

MODE=${1:-rgb}
LORA_CKPT=${LORA_CKPT:-/scratch/finetune_4m_lora_rgb/lora_step043000.pt}

if [ ! -d /scratch/LIBERO ]; then
    echo "[setup] Cloning LIBERO..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /scratch/LIBERO
fi

export PYTHONPATH="/home/garate/MultiSmolVLA/third_party/lerobot/src:/home/garate/MultiSmolVLA:/scratch/LIBERO:$PYTHONPATH"
cd /home/garate/MultiSmolVLA
echo "[eval] Mode: ${MODE}"

case "$MODE" in

  rgb)
    # RGB seul — passthrough DiVAE (reconstruction parfaite, baseline)
    python -u scripts/eval_native_smolvla_v2.py \
      --use_fourm \
      --use_tok_encoder \
      --rgb_passthrough \
      --n_episodes 5 \
      --output_dir "eval_results/test_rgb" \
      2>&1 | tee logs/eval_test_rgb.log
    ;;

  rgb_depth)
    # RGB + depth estimé en temps réel → 4M LoRA génère RGB
    python -u scripts/eval_native_smolvla_v2.py \
      --use_fourm \
      --use_tok_encoder \
      --use_depth \
      --use_estimator \
      --lora_checkpoint "${LORA_CKPT}" \
      --maskgit_steps 8 \
      --tok_temperature 3.0 \
      --cfg_scale 2.0 \
      --n_episodes 5 \
      --output_dir "eval_results/test_rgb_depth" \
      2>&1 | tee logs/eval_test_rgb_depth.log
    ;;

  rgb_depth_seg)
    # RGB + depth + seg estimés → 4M LoRA génère RGB
    python -u scripts/eval_native_smolvla_v2.py \
      --use_fourm \
      --use_tok_encoder \
      --use_depth \
      --use_seg \
      --use_estimator \
      --lora_checkpoint "${LORA_CKPT}" \
      --maskgit_steps 8 \
      --tok_temperature 3.0 \
      --cfg_scale 2.0 \
      --n_episodes 5 \
      --output_dir "eval_results/test_rgb_depth_seg" \
      2>&1 | tee logs/eval_test_rgb_depth_seg.log
    ;;

  *)
    echo "Usage: $0 <rgb|rgb_depth|rgb_depth_seg>"
    exit 1
    ;;
esac
