#!/bin/bash
#SBATCH --job-name=dpo-train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dpo-%j.out
#SBATCH --error=logs/dpo-%j.err

set -euo pipefail

module purge
module load cuda
# module load anaconda3
# source activate dpo

export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
mkdir -p logs dpo_checkpoints

python dpo_train.py \
  --preference_dataset YOUR_USER/alpaca-gpt2-dpo-prefs-v1 \
  --output_dir "$SCRATCH/dpo/dpo_checkpoints" \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 5e-7 \
  --beta 0.1
