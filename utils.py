"""Shared helpers for the Alpaca → generate → judge → DPO pipeline."""

from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM, PeftModel
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


def align_dpo_completion(tokenizer, prompt: str, completion: str) -> str:
    """Align completion so tokenize(prompt) is a prefix of tokenize(prompt + completion)."""
    if completion.startswith(prompt):
        completion = completion[len(prompt) :]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    joint_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]

    if joint_ids[: len(prompt_ids)] == prompt_ids:
        return completion

    # BPE splits differently at the prompt/completion boundary — split via joint tokenization.
    for prompt_len in range(len(joint_ids) + 1):
        if tokenizer.decode(joint_ids[:prompt_len]) == prompt:
            return tokenizer.decode(joint_ids[prompt_len:], skip_special_tokens=True)

    return completion


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
    """Policy (trainable SFT LoRA) + frozen reference copy for DPO."""
    policy = AutoPeftModelForCausalLM.from_pretrained(
        adapter_id,
        torch_dtype="auto",
        is_trainable=True,
    )
    policy.to(device)
    policy.train()
    for name, param in policy.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    policy.enable_input_require_grads()

    ref_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype="auto")
    ref_model = PeftModel.from_pretrained(ref_base, adapter_id, is_trainable=False)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"Policy trainable params: {trainable:,} / {total:,}")
    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters — DPO cannot learn.")

    return policy, ref_model


def load_base_model(device: torch.device) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype="auto")
    model.to(device)
    model.eval()
    return model


def load_dpo_model(checkpoint_path: str, device: torch.device):
    """Load a DPO checkpoint (adapter-only or full weights)."""
    path = Path(checkpoint_path)
    if (path / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype="auto")
        model = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype="auto")
    model.to(device)
    model.eval()
    return model
