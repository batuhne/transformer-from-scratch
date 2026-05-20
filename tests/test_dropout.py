"""Tests for inverted dropout: identity in eval, masking with scaling in train."""

from __future__ import annotations

import numpy as np
import pytest

from transformer.dropout import Dropout


def test_dropout_eval_mode_is_identity() -> None:
    drop = Dropout(p=0.5)
    drop.training = False
    x = np.random.randn(4, 8)
    out = drop.forward(x)
    assert np.array_equal(out, x)
    grad = drop.backward(np.ones_like(x))
    assert np.array_equal(grad, np.ones_like(x))


def test_dropout_p_zero_is_identity_in_train_mode() -> None:
    drop = Dropout(p=0.0)
    x = np.random.randn(4, 8)
    out = drop.forward(x)
    assert np.array_equal(out, x)


def test_dropout_train_mode_masks_and_rescales() -> None:
    np.random.seed(0)
    drop = Dropout(p=0.4)
    x = np.ones((1000, 1000))
    out = drop.forward(x)
    # Survivors are scaled by 1/(1-p); dropped entries are 0.
    survivor_rate = float(np.mean(out > 0))
    expected_survivor_rate = 1.0 - 0.4
    assert abs(survivor_rate - expected_survivor_rate) < 0.01
    assert np.allclose(out[out > 0], 1.0 / expected_survivor_rate)


def test_dropout_preserves_expectation() -> None:
    np.random.seed(0)
    drop = Dropout(p=0.3)
    x = np.full((500, 500), 7.0)
    out = drop.forward(x)
    # E[out] should equal x because inverted dropout rescales by 1/(1-p).
    assert abs(float(np.mean(out)) - 7.0) < 0.05


def test_dropout_backward_applies_same_mask() -> None:
    np.random.seed(0)
    drop = Dropout(p=0.5)
    x = np.random.randn(10, 10)
    out = drop.forward(x)
    grad = drop.backward(np.ones_like(x))
    # Backward zeros out the same entries forward did, scaled by 1/(1-p).
    zero_positions_forward = out == 0
    zero_positions_backward = grad == 0
    assert np.array_equal(zero_positions_forward, zero_positions_backward)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_dropout_rejects_invalid_p(bad: float) -> None:
    with pytest.raises(ValueError):
        Dropout(p=bad)
