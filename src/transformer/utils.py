"""Determinism helpers."""

from __future__ import annotations

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed numpy global RNG and return an explicit Generator with the same seed."""
    np.random.seed(seed)
    return np.random.default_rng(seed)
