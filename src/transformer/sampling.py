"""Logit filtering for sampling: top-k and top-p (nucleus) truncation."""

from __future__ import annotations

import numpy as np

from transformer.linear import softmax


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits; mask the rest to -inf. Operates on the last axis."""
    if k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -k, axis=-1)[..., -k, np.newaxis]
    return np.where(logits < kth, -np.inf, logits)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of tokens with cumulative prob >= p; mask rest. 1D only."""
    order = np.argsort(logits)[::-1]
    cumulative = np.cumsum(softmax(logits[order]))
    cutoff = int(np.searchsorted(cumulative, p)) + 1
    filtered = np.full_like(logits, -np.inf)
    keep = order[:cutoff]
    filtered[keep] = logits[keep]
    return filtered
