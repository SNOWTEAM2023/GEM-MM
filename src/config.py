from dataclasses import dataclass


@dataclass
class Config:
    """Knobs from the Qwen3-VL GEM-MM training run.

    Full fine-tune, k=5, λ=2.0, ρ=0.05, temperature=0.9, SEGA lr=1e-5,
    192 new tokens, SFT lr=2e-5, repo_limit128 system prompt.
    """

    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_length: int = 512
    max_new_tokens: int = 192
    sega_max_new_tokens: int = 192
    min_pixels: int = 200_704
    max_pixels: int = 401_408
    gradient_checkpointing: bool = True

    sft_lr: float = 2e-5
    sft_epochs: int = 1
    sft_batch_size: int = 1
    sft_system_prompt: str = "You are a helpful vision assistant."

    k_candidates: int = 5
    temperature: float = 0.9
    sega_lr: float = 1e-5
    sega_epochs: int = 1

    lambda_weight: float = 2.0
    top_m_ratio: float = 0.05

    system_prompt: str = (
        "You are a helpful vision assistant. "
        "Think step-by-step and then provide your Final Answer. "
        "Keep your step-by-step reasoning within 128 tokens before stating "
        "the final answer."
    )
