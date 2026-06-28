#!/bin/bash
#SBATCH --job-name=dpo-generate
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/generate-%j.out
#SBATCH --error=logs/generate-%j.err

set -euo pipefail

# Adjust for your DeltaAI environment (see NCSA training guide).
module purge
module load cuda
# module load anaconda3
# source activate dpo

export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
mkdir -p logs data/generated

# Login once on a login node: huggingface-cli login
python generate_responses.py \
  --max_examples 1000 \
  --batch_size 8 \
  --num_samples 4 \
  --output_dir "$SCRATCH/dpo/data/generated" \
  --resume \
  --push_to_hub \
  --hub_repo_id YOUR_USER/alpaca-gpt2-sft-samples-v1 \
  --private
