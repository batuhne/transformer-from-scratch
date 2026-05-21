"""Smoke tests for Adam: it converges on a tiny convex problem."""

from __future__ import annotations

import numpy as np
import pytest

from transformer.optim import Adam


class _Quadratic:
    """f(w) = 0.5 * ||w - target||^2, minimum at w = target."""

    def __init__(self, target: np.ndarray) -> None:
        self.W = np.zeros_like(target)
        self.dW: np.ndarray | None = None
        self.target = target

    def step(self) -> float:
        self.dW = self.W - self.target
        return 0.5 * float(np.sum(self.dW**2))


def test_adam_converges_on_quadratic() -> None:
    target = np.array([3.0, -2.0, 1.5, 0.1])
    problem = _Quadratic(target)
    opt = Adam([(problem, "W")], lr=0.1, max_norm=None)

    initial_loss = problem.step()
    for _ in range(500):
        problem.step()
        opt.step()

    assert problem.step() < 1e-6
    assert np.allclose(problem.W, target, atol=1e-3)
    assert initial_loss > problem.step()


def test_adam_bias_correction_first_step_matches_grad() -> None:
    """On step 1, m_hat / sqrt(v_hat) reduces to grad / |grad| (sign(grad))."""
    p = _Quadratic(target=np.array([1.0]))
    p.W = np.array([0.5])
    opt = Adam([(p, "W")], lr=1.0, max_norm=None, eps=0.0)

    p.step()
    opt.step()

    # grad = W - target = 0.5 - 1.0 = -0.5; first-step update = lr * sign(grad)
    assert np.allclose(p.W, 0.5 - 1.0 * (-1.0), atol=1e-12)


def test_adam_skips_params_with_no_gradient() -> None:
    """If dW is None, the parameter must remain unchanged."""

    class _Param:
        W = np.array([1.0, 2.0])
        dW = None

    p = _Param()
    opt = Adam([(p, "W")], lr=0.1)
    opt.step()
    assert np.array_equal(p.W, np.array([1.0, 2.0]))


def test_adam_global_scale_rescales_when_total_norm_exceeds_max() -> None:
    """If the concatenated gradient has norm > max_norm, scale = max_norm/norm."""

    class _P:
        W = np.zeros(2)
        dW = np.array([3.0, 4.0])  # ||dW||_2 = 5

    p = _P()
    opt = Adam([(p, "W")], max_norm=1.0)
    assert opt._global_scale() == pytest.approx(1.0 / 5.0, abs=1e-5)


def test_adam_global_scale_is_unity_when_total_norm_under_max() -> None:
    class _P:
        W = np.zeros(2)
        dW = np.array([0.3, 0.4])  # norm = 0.5 < 1.0

    p = _P()
    opt = Adam([(p, "W")], max_norm=1.0)
    assert opt._global_scale() == 1.0


def test_adam_global_scale_aggregates_across_multiple_params() -> None:
    """Norm is computed over the concatenation of all grads, not per-param."""

    class _P:
        W = np.zeros(1)
        dW = np.array([3.0])

    class _Q:
        W = np.zeros(1)
        dW = np.array([4.0])

    p, q = _P(), _Q()
    opt = Adam([(p, "W"), (q, "W")], max_norm=1.0)
    # Concat norm = sqrt(9 + 16) = 5; scale = 1/5.
    assert opt._global_scale() == pytest.approx(0.2, abs=1e-5)


def test_adam_global_scale_disabled_returns_unity() -> None:
    class _P:
        W = np.zeros(2)
        dW = np.array([100.0, 100.0])

    p = _P()
    opt = Adam([(p, "W")], max_norm=None)
    assert opt._global_scale() == 1.0


def test_adamw_decays_2d_param_with_zero_gradient() -> None:
    """A 2D param with zero grad still shrinks by (1 - lr*wd) under AdamW."""

    class _P:
        W = np.full((2, 2), 5.0)
        dW = np.zeros((2, 2))

    p = _P()
    opt = Adam([(p, "W")], lr=0.1, weight_decay=0.5, max_norm=None)
    opt.step()
    # Adam update with zero grad = zero; only weight decay fires.
    # delta = lr * wd * W_old = 0.1 * 0.5 * 5 = 0.25
    expected = 5.0 - 0.25
    assert np.allclose(p.W, expected)


def test_adamw_skips_decay_for_1d_param() -> None:
    """1D params (biases, gamma, beta) must NOT be decayed."""

    class _P:
        W = np.full(4, 5.0)  # 1D, so no decay
        dW = np.zeros(4)

    p = _P()
    opt = Adam([(p, "W")], lr=0.1, weight_decay=0.5, max_norm=None)
    opt.step()
    assert np.array_equal(p.W, np.full(4, 5.0))


def test_adamw_decay_zero_matches_plain_adam() -> None:
    """weight_decay=0 must produce the same update as the no-decay path."""
    np.random.seed(0)
    W0 = np.random.randn(3, 3)
    dW = np.random.randn(3, 3)

    class _P:
        pass

    p1 = _P()
    p1.W = W0.copy()
    p1.dW = dW
    opt1 = Adam([(p1, "W")], lr=0.1, weight_decay=0.0, max_norm=None)
    opt1.step()

    p2 = _P()
    p2.W = W0.copy()
    p2.dW = dW
    opt2 = Adam([(p2, "W")], lr=0.1, max_norm=None)
    opt2.step()

    assert np.array_equal(p1.W, p2.W)
