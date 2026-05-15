"""Gradient checks for SingleHeadAttention."""

from __future__ import annotations

import numpy as np
from conftest import numerical_gradient, relative_error

from transformer.attention import SingleHeadAttention, causal_mask


def _setup() -> tuple[SingleHeadAttention, np.ndarray, np.ndarray, np.ndarray]:
    attn = SingleHeadAttention(d_model=8, d_k=4)
    x = np.random.randn(2, 4, 8)
    mask = causal_mask(4)
    out = attn.forward(x, mask=mask)
    dout = np.random.randn(*out.shape)
    attn.backward(dout)
    return attn, x, mask, dout


def _loss_factory(attn, name, x, mask, dout):
    def loss(W):
        setattr(attn, name, W)
        return float(np.sum(attn.forward(x, mask=mask) * dout))

    return loss


def test_attention_projection_gradients_match_numerical() -> None:
    attn, x, mask, dout = _setup()
    for name in ("W_Q", "W_K", "W_V"):
        num = numerical_gradient(_loss_factory(attn, name, x, mask, dout), getattr(attn, name).copy())
        analytical = getattr(attn, f"d{name}")
        assert relative_error(analytical, num) < 1e-5, f"{name} gradient mismatch"


def test_attention_dx_matches_numerical() -> None:
    attn, x, mask, dout = _setup()
    dx = attn.backward(dout)  # recompute to capture dx return

    def loss(x_in: np.ndarray) -> float:
        return float(np.sum(attn.forward(x_in, mask=mask) * dout))

    num_dx = numerical_gradient(loss, x.copy())
    assert relative_error(dx, num_dx) < 1e-5


def test_causal_mask_blocks_future_positions() -> None:
    mask = causal_mask(5)
    assert mask.shape == (1, 5, 5)
    assert mask.dtype == bool
    # Diagonal and lower triangle should be unmasked (False).
    assert not mask[0].diagonal().any()
    assert not np.tril(mask[0], k=0).any()
    # Strict upper triangle should be masked (True).
    assert np.triu(mask[0], k=1).sum() == 10  # 5*(5-1)/2
