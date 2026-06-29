"""Determinism helpers."""

from __future__ import annotations

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed numpy's global RNG and return an explicit same-seed Generator."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def randn(rng: np.random.Generator | None, *shape: int) -> np.ndarray:
    """Standard-normal draw of the given shape from `rng`, or the global RNG if None."""
    if rng is None:
        return np.random.randn(*shape)
    return rng.standard_normal(shape)
