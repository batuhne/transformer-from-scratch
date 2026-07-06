"""Layer normalization over the last axis."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike


class LayerNorm:
    """Per-token feature normalization with learnable gamma, beta."""

    def __init__(self, d_model: int, eps: float = 1e-5, dtype: DTypeLike = np.float64) -> None:
        self.gamma = np.ones(d_model, dtype=dtype)
        self.beta = np.zeros(d_model, dtype=dtype)
        self.eps = eps

        self.dgamma: np.ndarray | None = None
        self.dbeta: np.ndarray | None = None

        self.x_hat: np.ndarray
        self.std_inv: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = (x - mean) * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        D = dout.shape[-1]
        dout_flat = dout.reshape(-1, D)
        x_hat_flat = self.x_hat.reshape(-1, D)

        self.dgamma = np.sum(dout_flat * x_hat_flat, axis=0)
        self.dbeta = np.sum(dout_flat, axis=0)

        dx_hat = dout * self.gamma
        # dx = (1/sigma) * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
        return self.std_inv * (
            dx_hat
            - np.mean(dx_hat, axis=-1, keepdims=True)
            - self.x_hat * np.mean(dx_hat * self.x_hat, axis=-1, keepdims=True)
        )
