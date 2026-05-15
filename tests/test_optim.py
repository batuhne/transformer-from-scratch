"""Smoke tests for Adam: it converges on a tiny convex problem."""

from __future__ import annotations

import numpy as np

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
    opt = Adam([(problem, "W")], lr=0.1, clip=None)

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
    opt = Adam([(p, "W")], lr=1.0, clip=None, eps=0.0)

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
