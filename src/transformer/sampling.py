"""Logit filtering for sampling: top-k and top-p (nucleus) truncation."""

from __future__ import annotations

import numpy as np

from transformer.linear import softmax


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits; mask the rest to -inf. Last-axis operation.

    Ties at the threshold may keep more than k entries (intentional: we never
    pick a tie-break order that depends on numpy version).
    """
    if k <= 0:
        raise ValueError(f"top_k must be positive, got {k}")
    if k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -k, axis=-1)[..., -k, np.newaxis]
    return np.where(logits < kth, -np.inf, logits)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of tokens with cumulative prob >= p; mask rest. 1D only."""
    if not 0.0 < p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    if logits.ndim != 1:
        raise ValueError(f"top_p_filter expects 1D logits, got {logits.ndim}D")
    order = np.argsort(logits, kind="stable")[::-1]
    cumulative = np.cumsum(softmax(logits[order]))
    cutoff = int(np.searchsorted(cumulative, p)) + 1
    filtered = np.full_like(logits, -np.inf)
    keep = order[:cutoff]
    filtered[keep] = logits[keep]
    return filtered
