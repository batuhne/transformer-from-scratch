"""Gradient checks for MultiHeadAttention."""

from __future__ import annotations

import numpy as np
from conftest import numerical_gradient, relative_error

from transformer.attention import causal_mask
from transformer.mha import MultiHeadAttention


def _setup() -> tuple[MultiHeadAttention, np.ndarray, np.ndarray, np.ndarray]:
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = np.random.randn(2, 4, 8)
    mask = causal_mask(4)
    out = mha.forward(x, mask=mask)
    dout = np.random.randn(*out.shape)
    mha.backward(dout)
    return mha, x, mask, dout


def _loss_factory(mha, name, x, mask, dout):
    def loss(W):
        setattr(mha, name, W)
        return float(np.sum(mha.forward(x, mask=mask) * dout))

    return loss


def test_mha_projection_gradients_match_numerical() -> None:
    mha, x, mask, dout = _setup()
    for name in ("W_Q", "W_K", "W_V", "W_O"):
        num = numerical_gradient(_loss_factory(mha, name, x, mask, dout), getattr(mha, name).copy())
        analytical = getattr(mha, f"d{name}")
        assert relative_error(analytical, num) < 1e-5, f"{name} gradient mismatch"


def test_mha_dx_matches_numerical() -> None:
    mha, x, mask, dout = _setup()
    dx = mha.backward(dout)

    def loss(x_in: np.ndarray) -> float:
        return float(np.sum(mha.forward(x_in, mask=mask) * dout))

    num_dx = numerical_gradient(loss, x.copy())
    assert relative_error(dx, num_dx) < 1e-5


def test_mha_attention_weights_respect_causal_mask() -> None:
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = np.random.randn(1, 4, 8)
    mha.forward(x, mask=causal_mask(4))
    # Upper triangle of attention weights must be effectively zero per head.
    upper = np.triu(mha.attn_weights[0], k=1)
    assert np.allclose(upper, 0.0, atol=1e-8)
    # Each row of attention weights still sums to 1.
    row_sums = mha.attn_weights[0].sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
