"""Single-head scaled dot-product attention with optional causal mask."""

from __future__ import annotations

import numpy as np

from transformer.linear import softmax, softmax_backward
from transformer.utils import randn


def causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular bool mask of shape (1, T, T); True blocks the position."""
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return mask[np.newaxis, :, :]


class SingleHeadAttention:
    """Scaled dot-product attention (teaching reference; the model uses MultiHeadAttention)."""

    def __init__(self, d_model: int, d_k: int, rng: np.random.Generator | None = None) -> None:
        scale = np.sqrt(2.0 / (d_model + d_k))
        self.W_Q = randn(rng, d_model, d_k) * scale
        self.W_K = randn(rng, d_model, d_k) * scale
        self.W_V = randn(rng, d_model, d_k) * scale
        self.d_k = d_k

        self.dW_Q: np.ndarray | None = None
        self.dW_K: np.ndarray | None = None
        self.dW_V: np.ndarray | None = None

        self.x: np.ndarray
        self.Q: np.ndarray
        self.K: np.ndarray
        self.V: np.ndarray
        self.attn_weights: np.ndarray

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        self.x = x
        self.Q = x @ self.W_Q
        self.K = x @ self.W_K
        self.V = x @ self.W_V

        scores = (self.Q @ self.K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, -np.inf, scores)

        self.attn_weights = softmax(scores)
        return self.attn_weights @ self.V

    def backward(self, dout: np.ndarray) -> np.ndarray:
        d_k = dout.shape[-1]

        d_attn = dout @ self.V.transpose(0, 2, 1)
        dV = self.attn_weights.transpose(0, 2, 1) @ dout

        d_scores = softmax_backward(d_attn, self.attn_weights)
        d_scores /= np.sqrt(self.d_k)

        dQ = d_scores @ self.K
        dK = d_scores.transpose(0, 2, 1) @ self.Q

        x_flat = self.x.reshape(-1, self.x.shape[-1])
        self.dW_Q = x_flat.T @ dQ.reshape(-1, d_k)
        self.dW_K = x_flat.T @ dK.reshape(-1, d_k)
        self.dW_V = x_flat.T @ dV.reshape(-1, d_k)

        return dQ @ self.W_Q.T + dK @ self.W_K.T + dV @ self.W_V.T
