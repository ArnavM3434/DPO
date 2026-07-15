#!/bin/bash
#SBATCH --job-name=dpo-judge
#SBATCH --account=bfgp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/judge-%j.out
#SBATCH --error=logs/judge-%j.err

set -euo pipefail

PROJECT_DIR="/work/hdd/bfgp/arnavm7/DPO"
CONDA_ENV="/work/hdd/bfgp/arnavm7/envs/eval"
TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"

module use /sw/user/modules/python
module load python/miniforge3_pytorch/2.10.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

pip install datasets peft accelerate --quiet

export HF_HOME="${HF_HOME:-/work/hdd/bfgp/arnavm7/huggingface}"
export PYTHONUNBUFFERED=1
mkdir -p logs data/preferences

# Smoke test: add --max_examples 50
python judge_responses.py \
  --model_id "${TEACHER_MODEL}" \
  --input_path "${PROJECT_DIR}/data/generated/responses_train.jsonl" \
  --output_path "${PROJECT_DIR}/data/preferences/train.jsonl" \
  --resume
