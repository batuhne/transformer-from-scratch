"""Single-head scaled dot-product attention with optional causal mask."""

from __future__ import annotations

import numpy as np

from transformer.linear import softmax


def causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular bool mask of shape (1, T, T); True blocks the position."""
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return mask[np.newaxis, :, :]


class SingleHeadAttention:
    """Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V."""

    def __init__(self, d_model: int, d_k: int) -> None:
        scale = np.sqrt(2.0 / (d_model + d_k))
        self.W_Q = np.random.randn(d_model, d_k) * scale
        self.W_K = np.random.randn(d_model, d_k) * scale
        self.W_V = np.random.randn(d_model, d_k) * scale
        self.d_k = d_k

        self.dW_Q: np.ndarray | None = None
        self.dW_K: np.ndarray | None = None
        self.dW_V: np.ndarray | None = None

        self.x: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.attn_weights: np.ndarray | None = None

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        self.x = x
        self.Q = x @ self.W_Q
        self.K = x @ self.W_K
        self.V = x @ self.W_V

        scores = (self.Q @ self.K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        self.attn_weights = softmax(scores)
        return self.attn_weights @ self.V

    def backward(self, dout: np.ndarray) -> np.ndarray:
        d_k = dout.shape[-1]

        d_attn = dout @ self.V.transpose(0, 2, 1)
        dV = self.attn_weights.transpose(0, 2, 1) @ dout

        # Softmax jacobian, row-wise: dS = A * (dA - sum_j(dA_j * A_j))
        sum_term = np.sum(d_attn * self.attn_weights, axis=-1, keepdims=True)
        d_scores = self.attn_weights * (d_attn - sum_term)
        d_scores /= np.sqrt(self.d_k)

        dQ = d_scores @ self.K
        dK = d_scores.transpose(0, 2, 1) @ self.Q

        x_flat = self.x.reshape(-1, self.x.shape[-1])
        self.dW_Q = x_flat.T @ dQ.reshape(-1, d_k)
        self.dW_K = x_flat.T @ dK.reshape(-1, d_k)
        self.dW_V = x_flat.T @ dV.reshape(-1, d_k)

        return dQ @ self.W_Q.T + dK @ self.W_K.T + dV @ self.W_V.T
