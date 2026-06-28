
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

#Pretrained GPT2 Behavior

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

test_prompts = [
    "What is machine learning?",
    "Explain reinforcement learning",
    "How does PPO work?",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        do_sample=False,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
    print("-" * 80)

#Alpaca Dataset

from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca")

from datasets import DatasetDict
split = dataset['train'].train_test_split(test_size=0.05, seed=42)
dataset = DatasetDict({
    'train': split['train'],
    'validation': split['test']
})


train_ds = dataset['train']
val_ds = dataset['validation']
train_ds[0]

train_ds[500]

#Change to Standard Prompt Completion

def preprocess_function(example):
    prompt = f"Human: {example['instruction']} {example['input']} "
    completion = f"Assistant: {example['output']}"
    full_text = prompt + completion
    tokenized = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt",
    )
    #only want to compute loss on completion
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels = tokenized["input_ids"].clone()
    labels[:, :prompt_len] = -100

    #mask out padding (except the first pad token which is the eos token)
    pad_token_id = tokenizer.eos_token_id
    for i in range(labels.shape[0]):
        eos_positions = (labels[i] == pad_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 1:
            # Keep the first EOS (end of completion) and mask the rest (padding)
            labels[i, eos_positions[1:]] = -100

    # fraction = (labels.squeeze() != -100).float().mean()
    # print(fraction)
    return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
    }

train_ds = train_ds.map(preprocess_function, remove_columns=["instruction", "input", "output", "text"])
val_ds = val_ds.map(preprocess_function, remove_columns=["instruction", "input", "output", "text"])

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

#model = PeftModel.from_pretrained(model, "checkpoints/checkpoint-700")

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
    eval_strategy = 'steps',
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate = 2e-5,
    num_train_epochs = 3,
    lr_scheduler_type = 'cosine',
    warmup_steps = 100,
    save_steps = 50,
    logging_strategy = "steps",
    logging_steps = 50,
    save_strategy = "steps",
    save_total_limit = 2,
    eval_steps = 200,
  )

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    tokenizer=tokenizer
)

trainer.train()