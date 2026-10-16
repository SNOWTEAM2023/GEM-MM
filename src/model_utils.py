"""Load Qwen3-VL for GEM-MM training."""
from __future__ import annotations

import os

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_model_and_processor(config, device):
    """Load Qwen3-VL weights and processor onto ``device``.

    ``config.model_name`` may be a Hugging Face id or a local directory
    (the paper runs used a local Qwen3-VL-4B-Instruct checkout).
    """
    model_name = config.model_name
    local = bool(model_name) and os.path.isdir(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading Qwen3-VL from {model_name} ({dtype})")

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        trust_remote_code=True,
        local_files_only=local,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=local,
    )
    model.to(device)
    if getattr(config, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    return model, processor
