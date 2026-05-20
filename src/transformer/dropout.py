"""Inverted dropout with train/eval mode."""

from __future__ import annotations

import numpy as np


class Dropout:
    """Inverted dropout: in train mode, zero each element with probability `p`
    and scale survivors by `1/(1-p)` so expected activation is unchanged.
    In eval mode (or with `p == 0`), forward and backward are identity.

    The scaling lets inference run with the unmodified weights: no rescaling
    at test time is needed, which is the standard GPT-style convention.
    """

    def __init__(self, p: float = 0.0) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p
        self.training = True
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        keep = 1.0 - self.p
        self.mask = (np.random.rand(*x.shape) < keep).astype(x.dtype) / keep
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return dout
        return dout * self.mask
