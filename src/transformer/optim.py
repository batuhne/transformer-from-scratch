"""Adam optimizer with bias correction and global-norm gradient clipping."""

from __future__ import annotations

import numpy as np


class Adam:
    """Adam (Kingma & Ba, 2014).

    Operates on a flat list of (object, param_name) pairs; reads the gradient
    from `obj.d<name>` and writes the updated value back to `obj.<name>`.

    `max_norm` enables global-norm gradient clipping: if the total L2 norm of
    the concatenated gradient vector exceeds `max_norm`, every gradient is
    scaled by `max_norm / total_norm` before forming the EMA estimates. This
    preserves the gradient direction; per-element clipping does not.
    """

    def __init__(
        self,
        params: list[tuple[object, str]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_norm: float | None = 1.0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_norm = max_norm
        self.t = 0

        self.m: list[np.ndarray] = []
        self.v: list[np.ndarray] = []
        for obj, name in params:
            p = getattr(obj, name)
            self.m.append(np.zeros_like(p))
            self.v.append(np.zeros_like(p))

    def _global_scale(self) -> float:
        """Return the scalar to multiply every grad by for global-norm clipping."""
        if self.max_norm is None:
            return 1.0
        total_sq = 0.0
        for obj, name in self.params:
            grad = getattr(obj, "d" + name)
            if grad is not None:
                total_sq += float(np.sum(grad * grad))
        total_norm = float(np.sqrt(total_sq))
        if total_norm <= self.max_norm:
            return 1.0
        return self.max_norm / (total_norm + 1e-6)

    def step(self) -> None:
        """Apply one update across all registered parameters."""
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        scale = self._global_scale()

        for i, (obj, name) in enumerate(self.params):
            grad = getattr(obj, "d" + name)
            if grad is None:
                continue
            if scale != 1.0:
                grad = grad * scale

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)

            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            param = getattr(obj, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(obj, name, param)
