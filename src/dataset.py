"""Multimodal preference dataset. SFT targets ``chosen``; SEGA uses the prompt."""
from __future__ import annotations

import json
import os
from typing import Any

from torch.utils.data import Dataset

from src.message_utils import build_messages, prepare_multimodal_inputs


class MultimodalPreferenceDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_root: str,
        processor: Any,
        config: Any,
        mode: str = "sft",
        max_samples: int = 0,
    ) -> None:
        self.image_root = image_root
        self.processor = processor
        self.config = config
        self.mode = mode
        self.rows: list[dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if max_samples and max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def _image_path(self, row: dict) -> str:
        rel = row.get("image") or row.get("image_path")
        if not rel:
            raise FileNotFoundError("row is missing an image path")
        path = rel if os.path.isabs(rel) else os.path.join(self.image_root, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    def _encode(self, system_prompt: str, prompt: str, image_path: str, assistant: str | None):
        messages = build_messages(system_prompt, prompt, image_path, include_assistant=assistant)
        inputs, _ = prepare_multimodal_inputs(self.processor, messages)
        return inputs

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        prompt = row.get("prompt") or row.get("question") or ""
        image_path = self._image_path(row)
        if self.mode == "sft":
            target = str(row.get("chosen") or row.get("answer") or "").strip()
            full = self._encode(self.config.sft_system_prompt, prompt, image_path, target)
            prompt_only = self._encode(self.config.sft_system_prompt, prompt, image_path, None)
            input_ids = full["input_ids"]
            labels = input_ids.clone()
            prompt_len = int(prompt_only["input_ids"].shape[1])
            labels[:, :prompt_len] = -100
            return {
                "input_ids": input_ids,
                "attention_mask": full["attention_mask"],
                "labels": labels,
                "pixel_values": full.get("pixel_values"),
                "image_grid_thw": full.get("image_grid_thw"),
            }

        system = self.config.system_prompt
        inputs = self._encode(system, prompt, image_path, None)
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "prompt_len": int(inputs["input_ids"].shape[1]),
            "prompt": prompt,
            "image_path": image_path,
        }
