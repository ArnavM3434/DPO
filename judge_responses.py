#!/usr/bin/env python3
"""
Build DPO preference pairs with a HuggingFace teacher model on GPU.

Default strategy (teacher):
  - chosen:   teacher generates a fresh response for the prompt
  - rejected: best of the 4 SFT model responses (ranked by the teacher)

Example:
  python judge_responses.py \
      --model_id Qwen/Qwen2.5-7B-Instruct \
      --input_path ./data/generated/responses_train.jsonl \
      --output_path ./data/preferences/train.jsonl
"""

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import JUDGE_PROMPT_VERSION, format_completion, strip_assistant_prefix

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

RANK_SYSTEM = """You are an expert evaluator for instruction-following assistants.
Given a user instruction and several candidate assistant responses, pick the single BEST response.

Judge on:
1. Correctness and factual accuracy
2. Following the instruction
3. Helpfulness and clarity

Respond with ONLY valid JSON in this exact format:
{"best": <1-based index>, "reason": "<one sentence>"}"""

TEACHER_SYSTEM = (
    "You are a helpful assistant. Answer the user's instruction clearly and accurately."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build DPO preferences with a teacher model")
    parser.add_argument("--input_path", type=str, default=None, help="Local JSONL from generate_responses.py")
    parser.add_argument("--input_dataset", type=str, default=None, help="HF dataset repo id")
    parser.add_argument("--output_path", type=str, default="./data/preferences/train.jsonl")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--mode",
        type=str,
        default="teacher",
        choices=["teacher", "rank_best_worst"],
        help="teacher: chosen=teacher gen, rejected=best SFT. rank_best_worst: best/worst of 4 SFT.",
    )
    parser.add_argument("--gen_temperature", type=float, default=0.7)
    parser.add_argument("--gen_max_new_tokens", type=int, default=512)
    parser.add_argument("--rank_max_new_tokens", type=int, default=128)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N input rows")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo_id", type=str, default=None)
    parser.add_argument("--private", action="store_true")
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

    if args.offset:
        rows = rows[args.offset :]
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


def user_text_from_row(row: dict) -> str:
    instruction = row["instruction"]
    input_text = row.get("input") or ""
    if input_text:
        return f"{instruction}\n\nInput: {input_text}"
    return instruction


def build_rank_prompt(row: dict) -> str:
    lines = [
        RANK_SYSTEM,
        "",
        f"Instruction:\n{user_text_from_row(row)}",
        "",
        "Candidate responses:",
    ]
    for i, response in enumerate(row["responses"], start=1):
        lines.append(f"\n[{i}]\n{response}")
    lines.append("\nReturn JSON only.")
    return "\n".join(lines)


def build_rank_best_worst_prompt(row: dict) -> str:
    return build_rank_prompt(row).replace(
        "pick the single BEST response",
        "pick the BEST and WORST response",
    ).replace(
        '{"best": <1-based index>, "reason": "<one sentence>"}',
        '{"best": <1-based index>, "worst": <1-based index>, "reason": "<one sentence>"}\n\nbest and worst must be different indices.',
    )


def parse_rank_response(text: str, num_responses: int) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        best = int(parsed["best"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if not (1 <= best <= num_responses):
        return None
    return {"best": best, "reason": parsed.get("reason", "")}


def parse_rank_best_worst_response(text: str, num_responses: int) -> dict | None:
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


class TeacherModel:
    def __init__(self, model_id: str, device: torch.device):
        print(f"Loading teacher model: {model_id}")
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
    def generate_prompt(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_chat(
        self, messages: list[dict], temperature: float, max_new_tokens: int
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate_prompt(prompt, temperature, max_new_tokens)


def load_teacher(model_id: str) -> TeacherModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("Warning: much slower without a GPU")
    return TeacherModel(model_id, device)


def generate_teacher_chosen(
    teacher: TeacherModel, row: dict, temperature: float, max_new_tokens: int
) -> str:
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": user_text_from_row(row)},
    ]
    text = teacher.generate_chat(messages, temperature, max_new_tokens)
    body = strip_assistant_prefix(text.strip())
    return format_completion(body)


def process_teacher(row: dict, teacher: TeacherModel, args) -> dict | None:
    rank_prompt = build_rank_prompt(row)
    rank_raw = teacher.generate_prompt(rank_prompt, temperature=0.1, max_new_tokens=args.rank_max_new_tokens)
    ranked = parse_rank_response(rank_raw, len(row["responses"]))
    if ranked is None:
        return None

    rejected = row["responses"][ranked["best"] - 1]
    chosen = generate_teacher_chosen(teacher, row, args.gen_temperature, args.gen_max_new_tokens)

    return {
        "chosen": chosen,
        "rejected": rejected,
        "judge_reason": ranked["reason"],
        "rejected_sft_index": ranked["best"],
        "chosen_source": "teacher_generation",
        "rejected_source": "best_sft_of_4",
    }


def process_rank_best_worst(row: dict, teacher: TeacherModel, args) -> dict | None:
    rank_prompt = build_rank_best_worst_prompt(row)
    rank_raw = teacher.generate_prompt(rank_prompt, temperature=0.1, max_new_tokens=args.rank_max_new_tokens)
    ranked = parse_rank_best_worst_response(rank_raw, len(row["responses"]))
    if ranked is None:
        return None

    return {
        "chosen": row["responses"][ranked["best"] - 1],
        "rejected": row["responses"][ranked["worst"] - 1],
        "judge_reason": ranked["reason"],
        "best_index": ranked["best"],
        "worst_index": ranked["worst"],
        "chosen_source": "best_sft_of_4",
        "rejected_source": "worst_sft_of_4",
    }


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    teacher = load_teacher(args.model_id)
    rows = load_input_rows(args)
    completed_ids = load_completed_ids(output_path) if args.resume else set()
    if completed_ids:
        print(f"Resuming: skipping {len(completed_ids)} completed examples")

    process_fn = process_teacher if args.mode == "teacher" else process_rank_best_worst

    kept = 0
    skipped = 0

    with output_path.open("a") as out_f:
        for row in tqdm(rows, desc="Building preferences"):
            example_id = row["example_id"]
            if example_id in completed_ids:
                continue

            try:
                result = process_fn(row, teacher, args)
            except torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as exc:
                print(f"Error on example {example_id}: {exc}")
                skipped += 1
                continue

            if result is None:
                print(f"Could not build preference pair for example {example_id}")
                skipped += 1
                continue

            pref_row = {
                "example_id": example_id,
                "instruction": row["instruction"],
                "input": row.get("input") or "",
                "prompt": row["prompt"],
                "judge_model": teacher.model_id,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "preference_mode": args.mode,
                **result,
            }
            out_f.write(json.dumps(pref_row) + "\n")
            out_f.flush()
            kept += 1

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
