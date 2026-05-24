"""Logit filtering for sampling: top-k truncation."""

from __future__ import annotations

import numpy as np


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits; mask the rest to -inf. Operates on the last axis."""
    if k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -k, axis=-1)[..., -k, np.newaxis]
    return np.where(logits < kth, -np.inf, logits)
