"""Autoregressive sampling from a trained Transformer."""

from __future__ import annotations

import numpy as np

from transformer.data import CharVocab
from transformer.linear import softmax
from transformer.model import Transformer


def generate(
    model: Transformer,
    vocab: CharVocab,
    start: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    rng: np.random.Generator | None = None,
    use_cache: bool = False,
) -> str:
    """Autoregressive char sampling. `use_cache=True` requires prompt+new <= max_seq_len."""
    if rng is None:
        rng = np.random.default_rng()
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    model.set_training(False)
    indices = list(vocab.encode(start))

    if use_cache:
        if len(indices) + max_new_tokens > model.max_seq_len:
            raise ValueError(
                f"use_cache requires len(prompt)+max_new_tokens <= max_seq_len; "
                f"got {len(indices)}+{max_new_tokens} > {model.max_seq_len}"
            )
        caches = model.init_caches()
        logits, caches = model.forward_step(np.array([indices]), caches, position=0)
        last_logits = logits[0, -1, :]
        position = len(indices)
        for _ in range(max_new_tokens):
            probs = softmax(last_logits / temperature)
            next_idx = int(rng.choice(len(probs), p=probs))
            indices.append(next_idx)
            logits, caches = model.forward_step(
                np.array([[next_idx]]), caches, position=position
            )
            last_logits = logits[0, 0, :]
            position += 1
    else:
        for _ in range(max_new_tokens):
            context = indices[-model.max_seq_len :]
            logits = model.forward(np.array([context]))
            probs = softmax(logits[0, -1, :] / temperature)
            next_idx = int(rng.choice(len(probs), p=probs))
            indices.append(next_idx)

    return vocab.decode(indices)
