"""Gradient checks for Linear, ReLU, and softmax+cross-entropy."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import numerical_gradient, relative_error

from transformer.linear import Linear, ReLU, cross_entropy_loss, softmax


def test_linear_dW_db_dx_match_numerical_gradient() -> None:
    layer = Linear(3, 4)
    x = np.random.randn(2, 3)
    targets = np.array([1, 3])

    logits = layer.forward(x)
    _, dlogits = cross_entropy_loss(logits, targets)
    dx = layer.backward(dlogits)

    def loss_with_W(W: np.ndarray) -> float:
        layer.W = W
        loss, _ = cross_entropy_loss(layer.forward(x), targets)
        return loss

    num_dW = numerical_gradient(loss_with_W, layer.W.copy())
    assert relative_error(layer.dW, num_dW) < 1e-5

    def loss_with_b(b: np.ndarray) -> float:
        layer.b = b
        loss, _ = cross_entropy_loss(layer.forward(x), targets)
        return loss

    num_db = numerical_gradient(loss_with_b, layer.b.copy())
    assert relative_error(layer.db, num_db) < 1e-5

    def loss_with_x(x_in: np.ndarray) -> float:
        loss, _ = cross_entropy_loss(layer.forward(x_in), targets)
        return loss

    num_dx = numerical_gradient(loss_with_x, x.copy())
    assert relative_error(dx, num_dx) < 1e-5


def test_relu_backward_matches_mask() -> None:
    relu = ReLU()
    x = np.random.randn(4, 5)
    relu.forward(x)
    dout = np.random.randn(4, 5)
    dx = relu.backward(dout)

    expected = dout * (x > 0)
    assert np.allclose(dx, expected)


def test_softmax_cross_entropy_gradient_matches_numerical() -> None:
    logits = np.random.randn(5, 7)
    targets = np.array([0, 3, 6, 2, 1])

    _, dlogits = cross_entropy_loss(logits, targets)

    def loss_at(z: np.ndarray) -> float:
        loss, _ = cross_entropy_loss(z, targets)
        return loss

    num_dlogits = numerical_gradient(loss_at, logits.copy())
    assert relative_error(dlogits, num_dlogits) < 1e-5


def test_softmax_all_neg_inf_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        softmax(np.array([-np.inf, -np.inf, -np.inf]))
