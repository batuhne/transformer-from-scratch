"""Multi-head attention."""

from __future__ import annotations

import numpy as np

from transformer.linear import softmax


class MultiHeadAttention:
    """Multi-head scaled dot-product attention with output projection.

    Splits d_model into n_heads parallel attention heads of size d_k = d_model
    // n_heads, then concatenates and projects with W_O.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        self.dW_Q: np.ndarray | None = None
        self.dW_K: np.ndarray | None = None
        self.dW_V: np.ndarray | None = None
        self.dW_O: np.ndarray | None = None

        self.x: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.attn_weights: np.ndarray | None = None
        self.attn_output: np.ndarray | None = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, d_model) -> (B, n_heads, T, d_k)."""
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, n_heads, T, d_k) -> (B, T, d_model)."""
        B, _, T, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Forward pass.

        Args:
            x: shape (B, T, d_model).
            mask: bool array broadcastable to (B, n_heads, T, T); True blocks.

        Returns:
            Output of shape (B, T, d_model).
        """
        self.x = x

        self.Q = self._split_heads(x @ self.W_Q)
        self.K = self._split_heads(x @ self.W_K)
        self.V = self._split_heads(x @ self.W_V)

        scores = (self.Q @ self.K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        self.attn_weights = softmax(scores)
        attn_out = self.attn_weights @ self.V
        self.attn_output = self._merge_heads(attn_out)
        return self.attn_output @ self.W_O

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass.

        Args:
            dout: gradient w.r.t. output, shape (B, T, d_model).

        Returns:
            Gradient w.r.t. input x, shape (B, T, d_model).
        """
        D = dout.shape[-1]

        self.dW_O = self.attn_output.reshape(-1, D).T @ dout.reshape(-1, D)
        d_attn_output = dout @ self.W_O.T
        d_attn_out = self._split_heads(d_attn_output)

        d_attn = d_attn_out @ self.V.transpose(0, 1, 3, 2)
        dV = self.attn_weights.transpose(0, 1, 3, 2) @ d_attn_out

        sum_term = np.sum(d_attn * self.attn_weights, axis=-1, keepdims=True)
        d_scores = self.attn_weights * (d_attn - sum_term)
        d_scores /= np.sqrt(self.d_k)

        dQ = d_scores @ self.K
        dK = d_scores.transpose(0, 1, 3, 2) @ self.Q

        dQ = self._merge_heads(dQ)
        dK = self._merge_heads(dK)
        dV = self._merge_heads(dV)

        x_flat = self.x.reshape(-1, D)
        self.dW_Q = x_flat.T @ dQ.reshape(-1, D)
        self.dW_K = x_flat.T @ dK.reshape(-1, D)
        self.dW_V = x_flat.T @ dV.reshape(-1, D)

        return dQ @ self.W_Q.T + dK @ self.W_K.T + dV @ self.W_V.T
