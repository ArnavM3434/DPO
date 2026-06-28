#!/bin/bash
#SBATCH --job-name=dpo-generate
#SBATCH --account=bfgp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/generate-%j.out
#SBATCH --error=logs/generate-%j.err

set -euo pipefail

PROJECT_DIR="/work/hdd/bfgp/arnavm7/DPO"
CONDA_ENV="/work/hdd/bfgp/arnavm7/envs/eval"

module use /sw/user/modules/python
module load python/miniforge3_pytorch/2.10.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

# Install only what this env is missing; transformers/torch already come from the module.
pip install datasets peft --quiet

export HF_HOME="${HF_HOME:-/work/hdd/bfgp/arnavm7/huggingface}"
export PYTHONUNBUFFERED=1
mkdir -p logs data/generated

python generate_responses.py \
  --max_examples 5000 \
  --batch_size 8 \
  --num_samples 4 \
  --output_dir "${PROJECT_DIR}/data/generated" \
  --resume
