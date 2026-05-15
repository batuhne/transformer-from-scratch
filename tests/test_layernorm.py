"""Gradient checks for LayerNorm."""

from __future__ import annotations

import numpy as np
from conftest import numerical_gradient, relative_error

from transformer.layernorm import LayerNorm


def _setup() -> tuple[LayerNorm, np.ndarray, np.ndarray]:
    ln = LayerNorm(d_model=8)
    x = np.random.randn(2, 4, 8) * 3 + 1
    dout = np.random.randn(2, 4, 8)
    ln.forward(x)
    ln.backward(dout)
    return ln, x, dout


def test_layernorm_normalizes_to_zero_mean_unit_variance() -> None:
    ln = LayerNorm(d_model=16)
    x = np.random.randn(3, 5, 16) * 4 + 2
    y = ln.forward(x)
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(y.std(axis=-1), 1.0, atol=1e-4)


def test_layernorm_dx_matches_numerical() -> None:
    ln, x, dout = _setup()
    dx = ln.backward(dout)

    def loss(x_in: np.ndarray) -> float:
        return float(np.sum(ln.forward(x_in) * dout))

    num_dx = numerical_gradient(loss, x.copy())
    assert relative_error(dx, num_dx) < 1e-5


def test_layernorm_dgamma_dbeta_match_numerical() -> None:
    ln, x, dout = _setup()

    def loss_gamma(g: np.ndarray) -> float:
        ln.gamma = g
        return float(np.sum(ln.forward(x) * dout))

    num_dgamma = numerical_gradient(loss_gamma, ln.gamma.copy())
    assert relative_error(ln.dgamma, num_dgamma) < 1e-5

    def loss_beta(b: np.ndarray) -> float:
        ln.beta = b
        return float(np.sum(ln.forward(x) * dout))

    num_dbeta = numerical_gradient(loss_beta, ln.beta.copy())
    assert relative_error(ln.dbeta, num_dbeta) < 1e-5
