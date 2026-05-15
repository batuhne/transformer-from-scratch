"""Adam optimizer with bias correction and per-element gradient clipping."""

from __future__ import annotations

import numpy as np


class Adam:
    """Adam (Kingma & Ba, 2014).

    Operates on a flat list of (object, param_name) pairs; reads the gradient
    from `obj.d<name>` and writes the updated value back to `obj.<name>`.
    """

    def __init__(
        self,
        params: list[tuple[object, str]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        clip: float | None = 1.0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip = clip
        self.t = 0

        self.m: list[np.ndarray] = []
        self.v: list[np.ndarray] = []
        for obj, name in params:
            p = getattr(obj, name)
            self.m.append(np.zeros_like(p))
            self.v.append(np.zeros_like(p))

    def step(self) -> None:
        """Apply one update across all registered parameters."""
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t

        for i, (obj, name) in enumerate(self.params):
            grad = getattr(obj, "d" + name)
            if grad is None:
                continue
            if self.clip is not None:
                grad = np.clip(grad, -self.clip, self.clip)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)

            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            param = getattr(obj, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(obj, name, param)
