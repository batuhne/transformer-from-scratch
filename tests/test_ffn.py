"""End-to-end gradient check for FeedForward."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import numerical_gradient, relative_error

from transformer.ffn import FeedForward


def test_ffn_dx_and_param_gradients_match_numerical() -> None:
    ffn = FeedForward(d_model=4, d_ff=8)
    x = np.random.randn(2, 3, 4)
    out = ffn.forward(x)
    dout = np.random.randn(*out.shape)
    dx = ffn.backward(dout)

    def loss_x(x_in: np.ndarray) -> float:
        return float(np.sum(ffn.forward(x_in) * dout))

    num_dx = numerical_gradient(loss_x, x.copy())
    assert relative_error(dx, num_dx) < 1e-5

    for name in ("W1", "W2"):
        layer = ffn.linear1 if name == "W1" else ffn.linear2

        def loss_W(W: np.ndarray, layer=layer) -> float:
            layer.W = W
            return float(np.sum(ffn.forward(x) * dout))

        num_dW = numerical_gradient(loss_W, layer.W.copy())
        assert relative_error(layer.dW, num_dW) < 1e-5, f"{name} mismatch"


def test_ffn_uses_he_init_scale() -> None:
    d_model, d_ff = 64, 256
    ffn = FeedForward(d_model=d_model, d_ff=d_ff)
    assert np.std(ffn.linear1.W) == pytest.approx(np.sqrt(2.0 / d_model), rel=0.1)
    assert np.std(ffn.linear2.W) == pytest.approx(np.sqrt(2.0 / d_ff), rel=0.1)
