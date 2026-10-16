"""Qwen-VL chat messages. Same layout as the paper training code."""
from __future__ import annotations

import os

from qwen_vl_utils import process_vision_info


def build_messages(system_prompt, prompt_text, image_path, include_assistant=None):
    image_ref = image_path
    if image_path and os.path.isfile(image_path):
        image_ref = f"file://{os.path.abspath(image_path)}"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    if include_assistant is not None:
        messages.append({"role": "assistant", "content": include_assistant})
    return messages


def prepare_multimodal_inputs(processor, messages, device=None):
    has_assistant = any(m["role"] == "assistant" for m in messages)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not has_assistant,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if device is not None:
        inputs = inputs.to(device)
    return inputs, text
