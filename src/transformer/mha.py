"""Multi-head attention."""

from __future__ import annotations

import numpy as np

from transformer.dropout import Dropout
from transformer.linear import softmax


class MultiHeadAttention:
    """n_heads parallel scaled-dot-product attentions, concatenated + W_O."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        self.attn_dropout = Dropout(dropout, rng=rng)

        self.dW_Q: np.ndarray | None = None
        self.dW_K: np.ndarray | None = None
        self.dW_V: np.ndarray | None = None
        self.dW_O: np.ndarray | None = None

        self.x: np.ndarray
        self.Q: np.ndarray
        self.K: np.ndarray
        self.V: np.ndarray
        self.attn_weights: np.ndarray
        self.attn_post: np.ndarray
        self.attn_output: np.ndarray

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, d_model) -> (B, n_heads, T, d_k)."""
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, n_heads, T, d_k) -> (B, T, d_model)."""
        B, _, T, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        self.x = x

        self.Q = self._split_heads(x @ self.W_Q)
        self.K = self._split_heads(x @ self.W_K)
        self.V = self._split_heads(x @ self.W_V)

        scores = (self.Q @ self.K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        self.attn_weights = softmax(scores)
        self.attn_post = self.attn_dropout.forward(self.attn_weights)
        attn_out = self.attn_post @ self.V
        self.attn_output = self._merge_heads(attn_out)
        return self.attn_output @ self.W_O

    def forward_step(
        self,
        x: np.ndarray,
        K_cache: np.ndarray | None,
        V_cache: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Incremental attention; appends `x`'s K,V to caches. Returns (out, K, V)."""
        Q = self._split_heads(x @ self.W_Q)
        K_new = self._split_heads(x @ self.W_K)
        V_new = self._split_heads(x @ self.W_V)

        if K_cache is not None:
            K = np.concatenate([K_cache, K_new], axis=2)
            V = np.concatenate([V_cache, V_new], axis=2)
        else:
            K, V = K_new, V_new

        T_cache = K_cache.shape[2] if K_cache is not None else 0
        T_new = x.shape[1]
        T_total = T_cache + T_new

        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if T_new > 1:
            i = np.arange(T_new)[:, None]
            j = np.arange(T_total)[None, :]
            mask = j > (T_cache + i)
            scores = np.where(mask, -1e9, scores)

        attn = softmax(scores)
        self.attn_weights = attn
        out = self._merge_heads(attn @ V)
        return out @ self.W_O, K, V

    def backward(self, dout: np.ndarray) -> np.ndarray:
        D = dout.shape[-1]

        self.dW_O = self.attn_output.reshape(-1, D).T @ dout.reshape(-1, D)
        d_attn_output = dout @ self.W_O.T
        d_attn_out = self._split_heads(d_attn_output)

        d_attn_post = d_attn_out @ self.V.transpose(0, 1, 3, 2)
        dV = self.attn_post.transpose(0, 1, 3, 2) @ d_attn_out

        d_attn = self.attn_dropout.backward(d_attn_post)

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
