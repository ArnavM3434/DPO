#!/usr/bin/env python3
"""
Score generated responses locally with Ollama (Llama 3.2 3B).

Picks best and worst of N responses per prompt and writes a DPO-ready preference dataset.

Example:
  python judge_responses.py \
      --input_path ./data/generated/responses_train.jsonl \
      --output_path ./data/preferences/train.jsonl

  python judge_responses.py \
      --input_dataset YOUR_USER/alpaca-gpt2-sft-samples-v1 \
      --output_path ./data/preferences/train.jsonl \
      --push_to_hub --hub_repo_id YOUR_USER/alpaca-gpt2-dpo-prefs-v1
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from datasets import Dataset, load_dataset
from tqdm import tqdm

from utils import JUDGE_PROMPT_VERSION


JUDGE_SYSTEM = """You are an expert evaluator for instruction-following assistants.
Given a user instruction and several candidate assistant responses, pick the BEST and WORST response.

Judge on:
1. Correctness and factual accuracy
2. Following the instruction
3. Helpfulness and clarity

Respond with ONLY valid JSON in this exact format:
{"best": <1-based index>, "worst": <1-based index>, "reason": "<one sentence>"}

best and worst must be different indices."""


def parse_args():
    parser = argparse.ArgumentParser(description="Judge responses with Ollama")
    parser.add_argument("--input_path", type=str, default=None, help="Local JSONL from generate_responses.py")
    parser.add_argument("--input_dataset", type=str, default=None, help="HF dataset repo id")
    parser.add_argument("--output_path", type=str, default="./data/preferences/train.jsonl")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    parser.add_argument("--ollama_model", type=str, default="llama3.2:3b")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo_id", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--sleep_seconds", type=float, default=0.0, help="Pause between Ollama calls")
    return parser.parse_args()


def load_input_rows(args) -> list[dict]:
    if args.input_path and args.input_dataset:
        raise ValueError("Provide only one of --input_path or --input_dataset")
    if not args.input_path and not args.input_dataset:
        raise ValueError("Provide --input_path or --input_dataset")

    if args.input_dataset:
        ds = load_dataset(args.input_dataset, split="train")
        rows = [dict(row) for row in ds]
    else:
        rows = [json.loads(line) for line in Path(args.input_path).read_text().splitlines() if line.strip()]

    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    return rows


def load_completed_ids(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open() as f:
        for line in f:
            row = json.loads(line)
            done.add(row["example_id"])
    return done


def build_judge_prompt(row: dict) -> str:
    instruction = row["instruction"]
    input_text = row.get("input") or ""
    user_text = instruction if not input_text else f"{instruction}\n\nInput: {input_text}"

    lines = [
        JUDGE_SYSTEM,
        "",
        f"Instruction:\n{user_text}",
        "",
        "Candidate responses:",
    ]
    for i, response in enumerate(row["responses"], start=1):
        lines.append(f"\n[{i}]\n{response}")
    lines.append("\nReturn JSON only.")
    return "\n".join(lines)


def parse_judge_response(text: str, num_responses: int) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        best = int(parsed["best"])
        worst = int(parsed["worst"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if not (1 <= best <= num_responses and 1 <= worst <= num_responses):
        return None
    if best == worst:
        return None
    return {"best": best, "worst": worst, "reason": parsed.get("reason", "")}


def call_ollama(base_url: str, model: str, prompt: str) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_input_rows(args)
    completed_ids = load_completed_ids(output_path) if args.resume else set()
    if completed_ids:
        print(f"Resuming: skipping {len(completed_ids)} completed examples")

    kept = 0
    skipped = 0

    with output_path.open("a") as out_f:
        for row in tqdm(rows, desc="Judging"):
            example_id = row["example_id"]
            if example_id in completed_ids:
                continue

            judge_prompt = build_judge_prompt(row)
            try:
                raw = call_ollama(args.ollama_url, args.ollama_model, judge_prompt)
            except requests.RequestException as exc:
                print(f"Ollama error on example {example_id}: {exc}")
                skipped += 1
                continue

            parsed = parse_judge_response(raw, len(row["responses"]))
            if parsed is None:
                print(f"Could not parse judge output for example {example_id}: {raw[:200]}")
                skipped += 1
                continue

            chosen = row["responses"][parsed["best"] - 1]
            rejected = row["responses"][parsed["worst"] - 1]

            pref_row = {
                "example_id": example_id,
                "instruction": row["instruction"],
                "input": row.get("input") or "",
                "prompt": row["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "judge_model": args.ollama_model,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "judge_reason": parsed["reason"],
                "best_index": parsed["best"],
                "worst_index": parsed["worst"],
            }
            out_f.write(json.dumps(pref_row) + "\n")
            kept += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"Wrote {kept} preference pairs to {output_path} (skipped {skipped})")

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub_repo_id is required when using --push_to_hub")
        all_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
        ds = Dataset.from_list(all_rows)
        ds.push_to_hub(args.hub_repo_id, private=args.private)
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.hub_repo_id}")


if __name__ == "__main__":
    main()
