"""Gradient check for Embedding and sanity checks for positional encoding."""

from __future__ import annotations

import numpy as np
from conftest import numerical_gradient, relative_error

from transformer.embedding import Embedding, get_positional_encoding


def test_embedding_dW_matches_numerical_gradient() -> None:
    emb = Embedding(vocab_size=5, d_model=4)
    indices = np.array([[0, 1, 2], [1, 3, 0]])  # repeats verify gradient accumulation

    out = emb.forward(indices)
    dout = np.random.randn(*out.shape)
    emb.backward(dout)

    def loss_at_W(W: np.ndarray) -> float:
        emb.W = W
        return float(np.sum(emb.forward(indices) * dout))

    num_dW = numerical_gradient(loss_at_W, emb.W.copy())
    assert relative_error(emb.dW, num_dW) < 1e-5


def test_positional_encoding_shape_and_bounds() -> None:
    pe = get_positional_encoding(max_seq_len=32, d_model=64)
    assert pe.shape == (32, 64)
    assert pe.min() >= -1.0 - 1e-12
    assert pe.max() <= 1.0 + 1e-12
    # Row 0: sin(0)=0 in even cols, cos(0)=1 in odd cols
    assert np.allclose(pe[0, 0::2], 0.0)
    assert np.allclose(pe[0, 1::2], 1.0)
