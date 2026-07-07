# GPT-2 DPO on Alpaca

## Results

Blind ranking (Qwen 2.5 7B judge, 100 held-out Alpaca val prompts):
- DPO preferred to SFT **71%** of the time
- SFT preferred to pretrained **93%** of the time

## Links

- SFT: [gpt2-alpaca-second-try](https://huggingface.co/ArnavM3434/gpt2-alpaca-second-try)
- DPO: [gpt2-alpaca-dpo-v2](https://huggingface.co/ArnavM3434/gpt2-alpaca-dpo-v2)
- Generated responses: [alpaca-gpt2-sft-samples-v1](https://huggingface.co/datasets/ArnavM3434/alpaca-gpt2-sft-samples-v1)

## Pipeline

SFT on Alpaca → generate 4 SFT samples per prompt → Qwen teacher builds preference pairs → DPO → eval.

1. `generate_responses.py`
2. `judge_responses.py`
3. `dpo_train.py`
4. `eval_models.py`

SLURM scripts in `scripts/`.
