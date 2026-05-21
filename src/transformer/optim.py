"""Adam(W) optimizer: bias correction, global-norm clip, decoupled weight decay."""

from __future__ import annotations

import numpy as np


class Adam:
    """Adam with optional global-norm clip and AdamW decoupled weight decay."""

    def __init__(
        self,
        params: list[tuple[object, str]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_norm: float | None = 1.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_norm = max_norm
        self.weight_decay = weight_decay
        self.t = 0

        self.m: list[np.ndarray] = []
        self.v: list[np.ndarray] = []
        for obj, name in params:
            p = getattr(obj, name)
            self.m.append(np.zeros_like(p))
            self.v.append(np.zeros_like(p))

    def _global_scale(self) -> float:
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
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0.0 and param.ndim >= 2:
                update = update + self.lr * self.weight_decay * param
            param -= update
            setattr(obj, name, param)
