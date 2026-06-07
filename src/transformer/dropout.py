"""Inverted dropout with train/eval mode."""

from __future__ import annotations

import numpy as np


class Dropout:
    """Inverted dropout: zero with prob p, survivors scaled by 1/(1-p)."""

    def __init__(self, p: float = 0.0, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p
        self.training = True
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        keep = 1.0 - self.p
        self.mask = (self.rng.random(x.shape) < keep).astype(x.dtype) / keep
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return dout
        return dout * self.mask
