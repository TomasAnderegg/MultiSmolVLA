#!/bin/bash
#SBATCH --job-name=eval_vanilla
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/garate/MultiSmolVLA/logs/eval_vanilla_%j.out

mkdir -p /home/garate/MultiSmolVLA/logs
mkdir -p /home/garate/MultiSmolVLA/eval_results/vanilla_smolvla

eval "$(conda shell.bash hook)"
conda activate /scratch/izar/$USER/envs/multismolvla

export PYTHONPATH="/scratch/izar/garate/LIBERO:/home/garate/MultiSmolVLA/third_party/lerobot/src"
export HF_HOME="/scratch/izar/$USER/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/scratch/izar/$USER/huggingface_cache/hub"

echo "Job started at $(date)"
nvidia-smi

python /home/garate/MultiSmolVLA/third_party/lerobot/src/lerobot/scripts/lerobot_eval.py \
    --policy.path=lerobot/smolvla_libero \
    --env.type=libero \
    --eval.batch_size=1 \
    --eval.n_episodes=5 \
    --policy.device=cuda \
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

echo "Job finished at $(date)"
