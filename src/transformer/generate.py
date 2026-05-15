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
) -> str:
    """Generate text one character at a time, feeding samples back as context.

    Context is truncated to `model.max_seq_len` so positional encoding stays
    valid. Use `temperature` < 1 to sharpen, > 1 to flatten the distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    indices = list(vocab.encode(start))

    for _ in range(max_new_tokens):
        context = indices[-model.max_seq_len :]
        logits = model.forward(np.array([context]))
        next_logits = logits[0, -1, :] / temperature
        probs = softmax(next_logits)
        next_idx = int(rng.choice(len(probs), p=probs))
        indices.append(next_idx)

    return vocab.decode(indices)
