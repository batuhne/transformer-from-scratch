"""Linear layer, ReLU, softmax, and cross-entropy loss."""

from __future__ import annotations

from typing import Literal

import numpy as np

from transformer.utils import randn


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    x_max = np.max(x, axis=-1, keepdims=True)
    if not np.all(np.isfinite(x_max)):
        raise ValueError("softmax row has no finite element")
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def softmax_backward(d_out: np.ndarray, softmax_out: np.ndarray) -> np.ndarray:
    """Gradient of a row-wise softmax: A * (dA - sum(dA * A))."""
    sum_term = np.sum(d_out * softmax_out, axis=-1, keepdims=True)
    return softmax_out * (d_out - sum_term)


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean cross-entropy; returns (loss, gradient w.r.t. logits)."""
    N = logits.shape[0]
    z_max = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - z_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    nll = -log_probs[np.arange(N), targets]
    loss = float(np.mean(nll))
    dlogits = np.exp(log_probs)
    dlogits[np.arange(N), targets] -= 1
    dlogits /= N
    return loss, dlogits


class Linear:
    """Affine map y = x @ W + b. Xavier init by default, He for ReLU inputs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: Literal["xavier", "he"] = "xavier",
        rng: np.random.Generator | None = None,
    ) -> None:
        if init == "he":
            scale = np.sqrt(2.0 / in_features)
        else:
            scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = randn(rng, in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self.x: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        dout_flat = dout.reshape(-1, dout.shape[-1])
        self.dW = x_flat.T @ dout_flat
        self.db = dout_flat.sum(axis=0)
        return dout @ self.W.T


class ReLU:
    """Rectified linear activation."""

    def __init__(self) -> None:
        self.mask: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0).astype(x.dtype)
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask
