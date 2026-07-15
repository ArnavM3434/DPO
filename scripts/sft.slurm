#!/bin/bash
#SBATCH --job-name=sft-train
#SBATCH --account=bfgp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sft-%j.out
#SBATCH --error=logs/sft-%j.err

set -euo pipefail

PROJECT_DIR="/work/hdd/bfgp/arnavm7/DPO"
CONDA_ENV="/work/hdd/bfgp/arnavm7/envs/eval"

module use /sw/user/modules/python
module load python/miniforge3_pytorch/2.10.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

pip install datasets peft wandb --quiet

export HF_HOME="${HF_HOME:-/work/hdd/bfgp/arnavm7/huggingface}"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-gpt2-sft}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sft-try-again-$(date +%Y%m%d-%H%M%S)}"
export WANDB_DIR="${PROJECT_DIR}/wandb"
# If compute nodes can't reach wandb.ai, uncomment:
# export WANDB_MODE=offline

mkdir -p logs checkpoints wandb

python sft.py
