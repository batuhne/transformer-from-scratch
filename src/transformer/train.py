"""Training loop for the char-level transformer."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from transformer.data import get_batch
from transformer.linear import cross_entropy_loss
from transformer.model import Transformer
from transformer.optim import Adam


@dataclass
class TrainConfig:
    n_steps: int = 3000
    batch_size: int = 32
    seq_len: int = 32
    lr: float = 1e-3
    log_every: int = 200


def train(
    model: Transformer,
    data: np.ndarray,
    config: TrainConfig,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Run SGD with Adam. Returns the per-step training loss history."""
    if rng is None:
        rng = np.random.default_rng()
    if config.seq_len > model.max_seq_len:
        raise ValueError(
            f"seq_len={config.seq_len} exceeds model.max_seq_len={model.max_seq_len}"
        )

    optimizer = Adam(model.params(), lr=config.lr)
    losses: list[float] = []
    t0 = time.time()

    for step in range(config.n_steps):
        x, y = get_batch(data, config.batch_size, config.seq_len, rng=rng)

        logits = model.forward(x)
        B, T, V = logits.shape
        loss, dlogits_flat = cross_entropy_loss(logits.reshape(-1, V), y.reshape(-1))
        losses.append(loss)

        model.backward(dlogits_flat.reshape(B, T, V))
        optimizer.step()

        if config.log_every and (step + 1) % config.log_every == 0:
            avg = float(np.mean(losses[-config.log_every :]))
            sps = (step + 1) / (time.time() - t0)
            print(f"step {step + 1:5d}/{config.n_steps} | loss {avg:.4f} | {sps:.1f} steps/s")

    return losses
