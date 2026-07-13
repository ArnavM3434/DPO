#!/bin/bash
#SBATCH --job-name=sft-train
#SBATCH --account=bfgp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
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

pip install datasets peft --quiet

export HF_HOME="${HF_HOME:-/work/hdd/bfgp/arnavm7/huggingface}"
export PYTHONUNBUFFERED=1

mkdir -p logs checkpoints

python sft.py
