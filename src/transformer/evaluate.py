"""Deterministic perplexity over the full dataset."""

from __future__ import annotations

import numpy as np

from transformer.linear import cross_entropy_loss
from transformer.model import Transformer


def perplexity(
    model: Transformer,
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
) -> float:
    """Token-weighted perplexity over contiguous non-overlapping windows of `data`."""
    was_training = [d.training for d in model.dropouts()]
    model.set_training(False)
    try:
        n_windows = (len(data) - 1) // seq_len
        if n_windows == 0:
            raise ValueError(f"data length {len(data)} too short for seq_len={seq_len}")

        starts = [w * seq_len for w in range(n_windows)]
        total_nll = 0.0
        total_tokens = 0
        for i in range(0, n_windows, batch_size):
            batch = starts[i : i + batch_size]
            x = np.stack([data[s : s + seq_len] for s in batch])
            y = np.stack([data[s + 1 : s + seq_len + 1] for s in batch])
            logits = model.forward(x)
            B, T, V = logits.shape
            loss, _ = cross_entropy_loss(logits.reshape(-1, V), y.reshape(-1))
            total_nll += loss * (B * T)
            total_tokens += B * T

        return float(np.exp(total_nll / total_tokens))
    finally:
        for d, prev in zip(model.dropouts(), was_training):
            d.training = prev
