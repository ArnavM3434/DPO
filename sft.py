
import os
import math
import json
from pathlib import Path
from typing import Dict, Any
import glob
import shutil

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from peft import LoraConfig, get_peft_model, PeftModel

from torch.utils.tensorboard import SummaryWriter

from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import os


model_name = "gpt2"

from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("EOS token:", tokenizer.eos_token)
print("EOS token ID:", tokenizer.eos_token_id)
print("PAD token:", tokenizer.pad_token)
print("PAD token ID:", tokenizer.pad_token_id)

tokenizer.pad_token = tokenizer.eos_token

print("PAD token ID:", tokenizer.pad_token_id)

print("Tokenizer max length:", tokenizer.model_max_length)

tokenizer.padding_side = 'right'


#Alpaca Dataset

from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca")

train_ds = dataset["train"]

#Change to Standard Prompt Completion

def find_prompt_token_length(tokenizer, prompt: str, input_ids: list[int]) -> int:
    """Find where the prompt ends inside a jointly tokenized sequence."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) >= len(prompt_ids) and input_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)

    # Rare BPE boundary mismatch: standalone prompt tokens != joint prefix.
    for i in range(len(input_ids) + 1):
        if tokenizer.decode(input_ids[:i]) == prompt:
            return i
    return min(len(prompt_ids), len(input_ids))


def preprocess_function(example):
    prompt = f"Human: {example['instruction']} {example['input']} Assistant: "
    completion = example["output"]
    if tokenizer.eos_token and not completion.endswith(tokenizer.eos_token):
        completion = completion + tokenizer.eos_token

    full_text = prompt + completion
    max_length = 1024

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Mask prompt tokens using the joint tokenization, not tokenize(prompt) alone.
    prompt_len = find_prompt_token_length(tokenizer, prompt, input_ids)
    labels = input_ids.copy()
    for i in range(prompt_len):
        labels[i] = -100

    # Mask padding only (attention_mask == 0). Do not mask by pad_token_id — for GPT-2
    # eos and pad share the same id, so the real trailing EOS must stay in labels.
    for i in range(len(labels)):
        if attention_mask[i] == 0:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_ds = train_ds.map(preprocess_function, remove_columns=["instruction", "input", "output", "text"])

#SFT Trainer

from transformers import Trainer, TrainingArguments

#Peft Config

LORA_CONFIG = dict(
    r=32,
    lora_alpha=64,
    target_modules=["c_attn", "c_proj", "q_attn", "wte", "wpe"],
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM",
)
lora_config = LoraConfig(**LORA_CONFIG)

model = get_peft_model(model, lora_config)


def inspect_trainable_params(model):
    total = 0
    trainable = 0
    details = []
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            details.append(n)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    print("Example trainable params:", details[:20])
    return details

inspect_trainable_params(model)

for name, parameter in model.named_parameters():
    if "lora_" in name:
        parameter.requires_grad = True

inspect_trainable_params(model)

training_args = TrainingArguments(
    output_dir = "./checkpoints",
    eval_strategy = "no",
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 2,
    learning_rate = 2e-5,
    num_train_epochs = 5,
    lr_scheduler_type = 'cosine',
    warmup_steps = 100,
    save_steps = 200,
    logging_strategy = "steps",
    logging_steps = 50,
    save_strategy = "steps",
    save_total_limit = 2,
  )

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    tokenizer=tokenizer
)

trainer.train()