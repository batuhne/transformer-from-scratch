"""Position-wise feed-forward network: Linear -> ReLU -> Linear."""

from __future__ import annotations

import numpy as np

from transformer.linear import Linear, ReLU


class FeedForward:
    """Two-layer MLP applied independently at each sequence position."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        self.linear1 = Linear(d_model, d_ff)
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.linear2.forward(self.relu.forward(self.linear1.forward(x)))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.linear1.backward(self.relu.backward(self.linear2.backward(dout)))

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "W1": self.linear1.W,
            "b1": self.linear1.b,
            "W2": self.linear2.W,
            "b2": self.linear2.b,
        }
