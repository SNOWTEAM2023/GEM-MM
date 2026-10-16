"""GEM-MM: Entropy-Guided Preference Alignment for Vision-Language Models.

  python GEM_MM.py \
    --model_name /path/to/Qwen3-VL-4B-Instruct \
    --data_path data/preference_data.jsonl \
    --image_root data/images
"""
from __future__ import annotations

import argparse
import os

import torch

from src.config import Config
from src.dataset import MultimodalPreferenceDataset
from src.gem_trainer import GEMTrainer
from src.model_utils import load_model_and_processor
from src.sft_trainer import run_sft

_LOCAL_CANDIDATES = (
    os.environ.get("MODEL_NAME", ""),
    "/home/ychua060/hdd/data/autodl-tmp/Qwen3-VL-4B-Instruct",
    "/root/data/cys/models/Qwen3-VL-4B-Instruct",
)


def main():
    parser = argparse.ArgumentParser(description="Run GEM-MM (SFT then SEGA) on Qwen3-VL")
    parser.add_argument("--data_path", type=str, default="data/preference_data.jsonl")
    parser.add_argument("--image_root", type=str, default="data/images")
    parser.add_argument("--output_dir", type=str, default="output/gem_mm_final")
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument("--skip_sega", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--k_candidates", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Qwen3-VL id or local directory. Defaults to a local 4B checkout if present.",
    )
    args = parser.parse_args()

    config = Config()
    model_name = args.model_name
    if not model_name:
        for cand in _LOCAL_CANDIDATES:
            if cand and os.path.isdir(cand):
                model_name = cand
                break
    if model_name:
        config.model_name = model_name
    if args.k_candidates is not None:
        config.k_candidates = args.k_candidates
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
        config.sega_max_new_tokens = args.max_new_tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, processor = load_model_and_processor(config, device)

    if not args.skip_sft:
        sft_dataset = MultimodalPreferenceDataset(
            args.data_path,
            args.image_root,
            processor,
            config,
            mode="sft",
            max_samples=args.max_samples,
        )
        model = run_sft(model, sft_dataset, config, device, processor)

    if not args.skip_sega:
        gem_dataset = MultimodalPreferenceDataset(
            args.data_path,
            args.image_root,
            processor,
            config,
            mode="gen",
            max_samples=args.max_samples,
        )
        trainer = GEMTrainer(model, processor, config, device)
        trainer.train_sega(gem_dataset)

    print(f"Saving GEM-MM model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
