"""Token embedding and sinusoidal positional encoding."""

from __future__ import annotations

import numpy as np


class Embedding:
    """Learnable token embedding lookup."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        self.W = np.random.randn(vocab_size, d_model) * 0.02
        self.vocab_size = vocab_size
        self.dW: np.ndarray | None = None
        self.indices: np.ndarray | None = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """Look up embeddings for token indices of shape (B, T)."""
        self.indices = indices
        return self.W[indices]

    def backward(self, dout: np.ndarray) -> None:
        """Accumulate gradient into self.dW.

        np.add.at handles repeated indices correctly, unlike buffered +=.
        """
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.indices, dout)


def get_positional_encoding(max_seq_len: int, d_model: int) -> np.ndarray:
    """Fixed sinusoidal positional encoding from Vaswani et al.

    Returns array of shape (max_seq_len, d_model).
    """
    pe = np.zeros((max_seq_len, d_model))
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
