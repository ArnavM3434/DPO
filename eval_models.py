#!/usr/bin/env python3
"""
Evaluate pretrained GPT-2 vs SFT vs DPO on held-out Alpaca validation prompts.

Uses the 5%% validation split (seed=42) — disjoint from the train split used in
generate_responses.py. Qwen ranks all three responses per prompt.

Example:
  python eval_models.py \
      --dpo_model_path ./dpo_checkpoints \
      --output_path ./data/eval/results.jsonl
"""

import argparse
import gc
import json
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    SFT_MODEL_ID,
    format_prompt,
    load_base_model,
    load_dpo_model,
    load_sft_model,
    load_tokenizer,
)

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ALPACA_SPLIT_SEED = 42

RANK_SYSTEM = """You are an expert evaluator for instruction-following assistants.
Given a user instruction and three anonymous candidate assistant responses, rank them from best to worst.

Judge on:
1. Correctness and factual accuracy
2. Following the instruction
3. Helpfulness and clarity

Do not assume anything about how the responses were produced. Evaluate only the text.

Respond with ONLY valid JSON in this exact format:
{"ranking": [best_index, second_index, worst_index], "reason": "<one sentence>"}

Use each of 1, 2, 3 exactly once in the ranking list."""


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 / SFT / DPO with Qwen ranking")
    parser.add_argument("--dpo_model_path", type=str, required=True)
    parser.add_argument("--sft_model_id", type=str, default=SFT_MODEL_ID)
    parser.add_argument("--judge_model_id", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_path", type=str, default="./data/eval/results.jsonl")
    parser.add_argument("--summary_path", type=str, default="./data/eval/summary.json")
    parser.add_argument(
        "--generated_path",
        type=str,
        default=None,
        help="Optional: extra safety check to exclude prompts seen in generate_responses.py",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_validation_examples(num_examples: int, seed: int, generated_path: str | None) -> list[dict]:
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    val = alpaca.train_test_split(test_size=0.05, seed=ALPACA_SPLIT_SEED)["test"]
    val = val.shuffle(seed=seed).select(range(num_examples))

    used_prompts: set[str] = set()
    if generated_path and Path(generated_path).exists():
        for line in Path(generated_path).read_text().splitlines():
            if line.strip():
                used_prompts.add(json.loads(line)["prompt"])

    examples = []
    for i, ex in enumerate(val):
        instruction = ex["instruction"]
        input_text = ex.get("input") or ""
        prompt = format_prompt(instruction, input_text)
        if prompt in used_prompts:
            continue
        examples.append(
            {
                "eval_id": i,
                "instruction": instruction,
                "input": input_text,
                "prompt": prompt,
            }
        )
    if len(examples) < num_examples:
        print(
            f"Note: using {len(examples)} validation examples "
            f"(requested {num_examples}; split is disjoint from train by default)"
        )
    return examples[:num_examples]


def load_completed_ids(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    return {json.loads(line)["eval_id"] for line in output_path.read_text().splitlines() if line.strip()}


def decode_new_tokens(tokenizer, output_ids: torch.Tensor, input_width: int) -> str:
    new_tokens = output_ids[input_width:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not text.startswith("Assistant:"):
        text = f"Assistant: {text}"
    return text


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    input_width = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return [decode_new_tokens(tokenizer, seq, input_width) for seq in outputs]


def generate_all_for_model(
    model,
    tokenizer,
    examples: list[dict],
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    desc: str,
) -> list[str]:
    responses: list[str] = []
    for start in tqdm(range(0, len(examples), batch_size), desc=desc, leave=False):
        batch = examples[start : start + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        responses.extend(generate_batch(model, tokenizer, prompts, max_new_tokens, device))
    return responses


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class QwenJudge:
    def __init__(self, model_id: str, device: torch.device):
        print(f"Loading judge model: {model_id}")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def rank_three(
        self,
        instruction: str,
        input_text: str,
        responses: dict[str, str],
        rng: random.Random,
    ) -> dict | None:
        user_text = instruction if not input_text else f"{instruction}\n\nInput: {input_text}"

        models = list(responses.keys())
        rng.shuffle(models)
        label_to_model = {i + 1: model for i, model in enumerate(models)}

        lines = [
            RANK_SYSTEM,
            "",
            f"Instruction:\n{user_text}",
            "",
            "Candidate responses:",
        ]
        for label in [1, 2, 3]:
            lines.append(f"\n[{label}]\n{responses[label_to_model[label]]}")
        lines.append("\nReturn JSON only.")
        prompt = "\n".join(lines)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        parsed = parse_ranking(text)
        if parsed is None:
            return None
        parsed["label_to_model"] = label_to_model
        return parsed


def parse_ranking(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        ranking = [int(x) for x in parsed["ranking"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if sorted(ranking) != [1, 2, 3]:
        return None
    return {"ranking": ranking, "reason": parsed.get("reason", "")}


def rank_to_metrics(ranking: list[int], label_to_model: dict[int, str]) -> dict:
    # ranking = [best_label, second_label, worst_label]
    model_rank = {label_to_model[label]: ranking.index(label) for label in [1, 2, 3]}
    return {
        "dpo_beats_sft": model_rank["dpo"] < model_rank["sft"],
        "sft_beats_pretrained": model_rank["sft"] < model_rank["pretrained"],
        "best_model": label_to_model[ranking[0]],
        "ranking_models": [label_to_model[label] for label in ranking],
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path)

    examples = load_validation_examples(args.num_examples, args.seed, args.generated_path)
    completed = load_completed_ids(output_path) if args.resume else set()
    pending = [ex for ex in examples if ex["eval_id"] not in completed]

    if not pending and args.resume:
        print("All examples already evaluated; recomputing summary from output file.")
    else:
        tokenizer = load_tokenizer(padding_side="left")
        prompts = [ex["prompt"] for ex in pending]

        print("Generating with pretrained GPT-2...")
        base_model = load_base_model(device)
        base_responses = generate_all_for_model(
            base_model, tokenizer, pending, args.batch_size, args.max_new_tokens, device, "pretrained"
        )
        free_model(base_model)

        print("Generating with SFT model...")
        sft_model = load_sft_model(args.sft_model_id, device)
        sft_responses = generate_all_for_model(
            sft_model, tokenizer, pending, args.batch_size, args.max_new_tokens, device, "sft"
        )
        free_model(sft_model)

        print("Generating with DPO model...")
        dpo_model = load_dpo_model(args.dpo_model_path, device)
        dpo_responses = generate_all_for_model(
            dpo_model, tokenizer, pending, args.batch_size, args.max_new_tokens, device, "dpo"
        )
        free_model(dpo_model)

        print("Ranking with Qwen...")
        judge = QwenJudge(args.judge_model_id, device)

        with output_path.open("a") as f:
            for ex, pretrained, sft, dpo in tqdm(
                zip(pending, base_responses, sft_responses, dpo_responses),
                total=len(pending),
                desc="Judging",
            ):
                responses = {"pretrained": pretrained, "sft": sft, "dpo": dpo}
                rng = random.Random(args.seed + ex["eval_id"])
                ranked = judge.rank_three(ex["instruction"], ex["input"], responses, rng)
                if ranked is None:
                    print(f"Could not parse ranking for eval_id={ex['eval_id']}")
                    continue

                metrics = rank_to_metrics(ranked["ranking"], ranked["label_to_model"])
                row = {
                    "eval_id": ex["eval_id"],
                    "instruction": ex["instruction"],
                    "input": ex["input"],
                    "prompt": ex["prompt"],
                    "responses": responses,
                    "ranking": ranked["ranking"],
                    "label_to_model": {str(k): v for k, v in ranked["label_to_model"].items()},
                    "judge_reason": ranked["reason"],
                    "judge_model": args.judge_model_id,
                    "blind_ranking": True,
                    **metrics,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()

        free_model(judge.model)

    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    n = len(rows)
    dpo_beats_sft = sum(1 for r in rows if r.get("dpo_beats_sft"))
    sft_beats_pretrained = sum(1 for r in rows if r.get("sft_beats_pretrained"))
    best_counts = {"pretrained": 0, "sft": 0, "dpo": 0}
    for r in rows:
        best_counts[r.get("best_model", "")] = best_counts.get(r.get("best_model", ""), 0) + 1

    summary = {
        "num_examples": n,
        "dpo_beats_sft": dpo_beats_sft,
        "dpo_beats_sft_rate": dpo_beats_sft / n if n else 0.0,
        "sft_beats_pretrained": sft_beats_pretrained,
        "sft_beats_pretrained_rate": sft_beats_pretrained / n if n else 0.0,
        "best_model_counts": best_counts,
        "dpo_model_path": args.dpo_model_path,
        "sft_model_id": args.sft_model_id,
        "judge_model_id": args.judge_model_id,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote per-example results to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
