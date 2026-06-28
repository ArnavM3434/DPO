#!/usr/bin/env python3
"""
Generate multiple completions per Alpaca prompt from the SFT model.

Run on the cluster, then optionally push results to Hugging Face Hub.

Example:
  python generate_responses.py --max_examples 1000 --push_to_hub \
      --hub_repo_id YOUR_USER/alpaca-gpt2-sft-samples-v1
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from utils import SFT_MODEL_ID, format_prompt, load_sft_model, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SFT model responses for DPO")
    parser.add_argument("--model_id", type=str, default=SFT_MODEL_ID)
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--max_examples", type=int, default=None, help="Limit prompts (for testing)")
    parser.add_argument("--num_samples", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--batch_size", type=int, default=8, help="Prompts per forward pass")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./data/generated")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo_id", type=str, default=None)
    parser.add_argument("--private", action="store_true", help="Private HF dataset repo")
    parser.add_argument("--resume", action="store_true", help="Skip examples already in output JSONL")
    return parser.parse_args()


def load_model_and_tokenizer(model_id: str, device: torch.device):
    # Left padding for batched generation: the model must continue from the
    # last real prompt token, not from pad tokens appended on the right.
    # SFT used right padding for training; that only affects padded training batches.
    tokenizer = load_tokenizer(padding_side="left")
    model = load_sft_model(model_id, device)
    return model, tokenizer


def decode_new_tokens(tokenizer, output_ids: torch.Tensor, prompt_len: int) -> str:
    new_tokens = output_ids[prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not text.startswith("Assistant:"):
        text = f"Assistant: {text}"
    return text


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[list[str]]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    input_lengths = inputs["attention_mask"].sum(dim=1)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    batch_size = len(prompts)
    all_responses: list[list[str]] = []
    for i in range(batch_size):
        prompt_len = input_lengths[i].item()
        sample_responses = []
        for j in range(num_samples):
            seq = outputs[i * num_samples + j]
            sample_responses.append(decode_new_tokens(tokenizer, seq, prompt_len))
        all_responses.append(sample_responses)
    return all_responses


def load_completed_ids(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open() as f:
        for line in f:
            row = json.loads(line)
            done.add(row["example_id"])
    return done


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"responses_{args.split}.jsonl"

    completed_ids = load_completed_ids(output_path) if args.resume else set()
    if completed_ids:
        print(f"Resuming: skipping {len(completed_ids)} completed examples")

    print(f"Loading Alpaca dataset (split={args.split})...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    if args.split == "validation":
        alpaca = alpaca.train_test_split(test_size=0.05, seed=42)["test"]

    if args.max_examples is not None:
        alpaca = alpaca.select(range(min(args.max_examples, len(alpaca))))

    print(f"Loading model: {args.model_id}")
    model, tokenizer = load_model_and_tokenizer(args.model_id, device)

    gen_config = {
        "model_id": args.model_id,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }

    records = []
    batch_prompts: list[str] = []
    batch_meta: list[dict] = []

    def flush_batch():
        nonlocal batch_prompts, batch_meta, records
        if not batch_prompts:
            return

        responses = generate_batch(
            model,
            tokenizer,
            batch_prompts,
            args.num_samples,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            device,
        )

        with output_path.open("a") as f:
            for meta, resp_list in zip(batch_meta, responses):
                row = {
                    **meta,
                    "responses": resp_list,
                    "gen_config": gen_config,
                }
                f.write(json.dumps(row) + "\n")
                records.append(row)

        batch_prompts = []
        batch_meta = []

    for example_id, example in enumerate(tqdm(alpaca, desc="Generating")):
        if example_id in completed_ids:
            continue

        instruction = example["instruction"]
        input_text = example.get("input") or ""
        prompt = format_prompt(instruction, input_text)

        batch_prompts.append(prompt)
        batch_meta.append(
            {
                "example_id": example_id,
                "instruction": instruction,
                "input": input_text,
                "prompt": prompt,
            }
        )

        if len(batch_prompts) >= args.batch_size:
            flush_batch()

    flush_batch()

    print(f"Wrote {len(records)} new rows to {output_path}")

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub_repo_id is required when using --push_to_hub")

        all_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
        ds = Dataset.from_list(all_rows)
        ds.push_to_hub(args.hub_repo_id, private=args.private)
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.hub_repo_id}")


if __name__ == "__main__":
    main()
