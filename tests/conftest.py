"""Shared fixtures and gradient-check helpers for the test suite."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_numpy() -> None:
    """Seed numpy before every test so gradient checks stay reproducible."""
    np.random.seed(0)


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference numerical gradient of scalar f at x.

    x is modified in place during evaluation and restored before returning.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        grad[idx] = (fp - fm) / (2.0 * eps)
        x[idx] = old
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Max element-wise relative error between two arrays."""
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-8)))
