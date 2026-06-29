"""Shared helpers for the Alpaca → generate → judge → DPO pipeline."""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT_MODEL_ID = "ArnavM3434/gpt2-alpaca-second-try"
BASE_MODEL_ID = "gpt2"
JUDGE_PROMPT_VERSION = "v3-transformers-teacher"


def format_prompt(instruction: str, input_text: str = "") -> str:
    """Match the prompt format used in sft.py."""
    return f"Human: {instruction} {input_text} "


def format_completion(output: str) -> str:
    return f"Assistant: {output}"


def strip_assistant_prefix(text: str) -> str:
    prefix = "Assistant:"
    if text.startswith(prefix):
        return text[len(prefix) :].lstrip()
    return text


def load_tokenizer(padding_side: str = "right") -> AutoTokenizer:
    """Tokenizer lives on the base model; the adapter repo has no tokenizer files."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_sft_model(adapter_id: str, device: torch.device) -> AutoPeftModelForCausalLM:
    """Load base gpt2 + LoRA adapter from Hub (adapter-only checkpoint)."""
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_id, torch_dtype="auto")
    model.to(device)
    model.eval()
    return model


def load_sft_model_for_training(adapter_id: str, device: torch.device):
    """Policy + frozen reference for DPO; both start from the same SFT adapter."""
    policy = AutoPeftModelForCausalLM.from_pretrained(adapter_id, torch_dtype="auto")
    policy.to(device)

    ref_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype="auto")
    ref_model = AutoPeftModelForCausalLM.from_pretrained(ref_base, adapter_id)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return policy, ref_model
