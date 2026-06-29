"""Position-wise feed-forward network: Linear -> ReLU -> Linear."""

from __future__ import annotations

import numpy as np

from transformer.linear import Linear, ReLU


class FeedForward:
    """Two-layer MLP applied independently at each sequence position."""

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator | None = None) -> None:
        # He init: both layers feed a ReLU network, so scale by sqrt(2/fan_in).
        self.linear1 = Linear(d_model, d_ff, init="he", rng=rng)
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model, init="he", rng=rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.linear2.forward(self.relu.forward(self.linear1.forward(x)))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.linear1.backward(self.relu.backward(self.linear2.backward(dout)))

    def params(self) -> list[tuple[object, str]]:
        return [
            (self.linear1, "W"),
            (self.linear1, "b"),
            (self.linear2, "W"),
            (self.linear2, "b"),
        ]
