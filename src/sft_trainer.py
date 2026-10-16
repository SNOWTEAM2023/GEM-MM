"""Supervised warm-start on chosen multimodal answers (next-token CE)."""
from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


def _collate(batch):
    return batch[0]


def run_sft(model, dataset, config, device, processor=None):
    """One or more epochs of full-parameter CE. Prompt tokens are labeled -100."""
    del processor  # labels are built in the dataset
    model.train()
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.sft_lr,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=_collate,
    )
    print(f"[SFT] epochs={config.sft_epochs} lr={config.sft_lr} n={len(dataset)}")
    for epoch in range(config.sft_epochs):
        for batch in tqdm(loader, desc=f"sft-epoch-{epoch + 1}"):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            outputs = model(**{k: v for k, v in batch.items() if v is not None})
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    print("[SFT] completed")
    return model
