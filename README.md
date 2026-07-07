# GPT-2 DPO on Alpaca

Fine-tune GPT-2 on Alpaca with SFT, build preference pairs with a Qwen teacher, then run DPO.

**Models:** [ArnavM3434 on Hugging Face](https://huggingface.co/ArnavM3434)
- SFT: [gpt2-alpaca-second-try](https://huggingface.co/ArnavM3434/gpt2-alpaca-second-try)
- DPO: [gpt2-alpaca-dpo-v2](https://huggingface.co/ArnavM3434/gpt2-alpaca-dpo-v2)

## Results

Blind ranking (Qwen 2.5 7B judge, 100 held-out Alpaca val prompts):
- DPO preferred to SFT **71%** of the time
- SFT preferred to pretrained **93%** of the time

## Pipeline

1. `generate_responses.py` — 4 SFT samples per prompt
2. `judge_responses.py` — teacher chosen vs best SFT rejected
3. `dpo_train.py` — DPO on preference pairs
4. `eval_models.py` — pretrained vs SFT vs DPO

SLURM scripts in `scripts/`.
