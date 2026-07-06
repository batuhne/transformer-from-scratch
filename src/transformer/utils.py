"""Determinism helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike


def set_seed(seed: int) -> np.random.Generator:
    """Seed numpy's global RNG and return an explicit same-seed Generator."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def randn(
    rng: np.random.Generator | None, *shape: int, dtype: DTypeLike = np.float64
) -> np.ndarray:
    """Standard-normal draw of the given shape from `rng`, or the global RNG if None."""
    arr = np.random.randn(*shape) if rng is None else rng.standard_normal(shape)
    return arr.astype(dtype, copy=False)
