#!/usr/bin/env python3
"""
Run DPO on preference pairs (prompt, chosen, rejected).

Run on the cluster after judge_responses.py produces preferences locally.

Example:
  python dpo_train.py \
      --preference_dataset ./data/preferences/train.jsonl \
      --output_dir ./dpo_checkpoints

  python dpo_train.py \
      --preference_dataset YOUR_USER/alpaca-gpt2-dpo-prefs-v1 \
      --output_dir ./dpo_checkpoints
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from trl import DPOConfig, DPOTrainer

from utils import SFT_MODEL_ID, load_sft_model_for_training, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training on preference pairs")
    parser.add_argument("--model_id", type=str, default=SFT_MODEL_ID, help="SFT checkpoint (policy + reference)")
    parser.add_argument("--preference_path", type=str, default=None, help="Local JSONL preferences")
    parser.add_argument("--preference_dataset", type=str, default=None, help="HF dataset repo id")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./dpo_checkpoints")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_split", type=float, default=0.05, help="Hold out fraction for eval")
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_preferences(args) -> Dataset:
    if args.preference_path and args.preference_dataset:
        raise ValueError("Provide only one of --preference_path or --preference_dataset")
    if not args.preference_path and not args.preference_dataset:
        raise ValueError("Provide --preference_path or --preference_dataset")

    if args.preference_dataset:
        ds = load_dataset(args.preference_dataset, split="train")
    else:
        rows = [
            json.loads(line)
            for line in Path(args.preference_path).read_text().splitlines()
            if line.strip()
        ]
        ds = Dataset.from_list(rows)

    if args.max_examples is not None:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    # TRL expects prompt / chosen / rejected
    cols = ds.column_names
    keep = ["prompt", "chosen", "rejected"]
    drop = [c for c in cols if c not in keep]
    ds = ds.remove_columns(drop)
    return ds


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading preference dataset...")
    prefs = load_preferences(args)
    if args.eval_split > 0:
        split = prefs.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds = split["train"]
        eval_ds = split["test"]
        eval_strategy = "steps"
        eval_steps = args.save_steps
    else:
        train_ds = prefs
        eval_ds = None
        eval_strategy = "no"
        eval_steps = None
    print(f"Train: {len(train_ds)}" + (f" | Eval: {len(eval_ds)}" if eval_ds else ""))

    print(f"Loading policy model: {args.model_id}")
    tokenizer = load_tokenizer(padding_side="right")

    policy, ref_model = load_sft_model_for_training(args.model_id, device)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DPO model to {args.output_dir}")


if __name__ == "__main__":
    main()
