"""On-policy SEGA update for Qwen3-VL.

Same step as the paper trainer's entropy path:
  sample k answers, score r = -H_final + λ H_fork,
  group-standardize advantages, then
  L = mean_i( -A_i * sum_t log π(y_t) ).
"""
from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.entropy_scorer import EntropyScorer


class GEMTrainer:
    def __init__(self, model, processor, config, device):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device
        self.scorer = EntropyScorer(config, processor)
        self.optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config.sega_lr,
        )

    def _vision(self, batch):
        keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        return {k: batch[k] for k in keys if batch.get(k) is not None}

    def train_sega(self, dataset):
        k = self.config.k_candidates
        max_new = getattr(self.config, "sega_max_new_tokens", None) or self.config.max_new_tokens
        print(
            f"[SEGA] k={k} λ={self.config.lambda_weight} ρ={self.config.top_m_ratio} "
            f"lr={self.config.sega_lr} max_new={max_new}"
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        for epoch in range(self.config.sega_epochs):
            for batch in tqdm(loader, desc=f"sega-epoch-{epoch + 1}"):
                batch = {
                    key: value.to(self.device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                prompt_len = int(batch["prompt_len"])
                gen_inputs = self._vision(batch)

                self.model.eval()
                was_cache = self.model.config.use_cache
                self.model.config.use_cache = True
                with torch.no_grad():
                    outputs = self.model.generate(
                        **gen_inputs,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=self.config.temperature,
                        num_return_sequences=k,
                        pad_token_id=pad_id,
                    )
                self.model.config.use_cache = was_cache

                scores = []
                log_probs = []
                valid = 0
                for i in range(k):
                    seq = outputs[i : i + 1]
                    forward = {
                        "input_ids": seq,
                        "attention_mask": torch.ones_like(seq),
                        "pixel_values": gen_inputs.get("pixel_values"),
                        "image_grid_thw": gen_inputs.get("image_grid_thw"),
                    }
                    forward = {kk: vv for kk, vv in forward.items() if vv is not None}
                    with torch.no_grad():
                        self.model.eval()
                        scored = self.model(**forward)
                        scores.append(
                            self.scorer.get_gem_score(seq[0], scored.logits[0], prompt_len)
                        )
                    self.model.train()
                    trained = self.model(**forward)
                    logits_seg = trained.logits[0, prompt_len - 1 : -1, :]
                    labels_seg = seq[0, prompt_len:]
                    if len(labels_seg) == 0:
                        log_probs.append(torch.zeros((), device=self.device))
                        continue
                    nll = torch.nn.functional.cross_entropy(
                        logits_seg, labels_seg, reduction="none"
                    )
                    log_probs.append((-nll).sum())
                    valid += 1

                if valid < 2:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                scores_t = torch.stack(scores)
                advantages = scores_t - scores_t.mean()
                if advantages.std() > 1e-6:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                loss = scores_t.new_zeros(())
                for i in range(k):
                    lp = log_probs[i]
                    if isinstance(lp, torch.Tensor) and lp.requires_grad:
                        loss = loss + (-advantages[i].detach() * lp)
                loss = loss / valid
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        print("[SEGA] completed")
