"""Training loop for the char-level transformer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from transformer.data import get_batch
from transformer.linear import cross_entropy_loss
from transformer.model import Transformer
from transformer.optim import Adam
from transformer.schedule import cosine_warmup_lr


@dataclass
class TrainConfig:
    n_steps: int = 3000
    batch_size: int = 32
    seq_len: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    min_lr_ratio: float = 1.0
    max_norm: float | None = 1.0
    log_every: int = 200
    eval_every: int = 500
    eval_batches: int = 20


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[tuple[int, float]] = field(default_factory=list)


def eval_loss(
    model: Transformer,
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    rng: np.random.Generator,
) -> float:
    """Mean cross-entropy over `n_batches` random windows of `data`, no backward."""
    losses: list[float] = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, seq_len, rng=rng)
        logits = model.forward(x)
        B, T, V = logits.shape
        loss, _ = cross_entropy_loss(logits.reshape(-1, V), y.reshape(-1))
        losses.append(loss)
    return float(np.mean(losses))


def train(
    model: Transformer,
    train_data: np.ndarray,
    config: TrainConfig,
    val_data: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> TrainHistory:
    """Run Adam SGD. Returns per-step train losses and periodic val losses."""
    if rng is None:
        rng = np.random.default_rng()
    if config.seq_len > model.max_seq_len:
        raise ValueError(
            f"seq_len={config.seq_len} exceeds model.max_seq_len={model.max_seq_len}"
        )
    # Independent child stream for eval batch sampling, so val_loss is
    # decoupled from the exact point the training stream has reached.
    eval_rng = rng.spawn(1)[0]

    optimizer = Adam(
        model.params(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        max_norm=config.max_norm,
    )
    history = TrainHistory()
    t0 = time.time()
    model.set_training(True)
    use_schedule = config.warmup_steps > 0 or config.min_lr_ratio < 1.0
    min_lr = config.lr * config.min_lr_ratio

    for step in range(config.n_steps):
        if use_schedule:
            optimizer.lr = cosine_warmup_lr(
                step + 1, config.lr, config.warmup_steps, config.n_steps, min_lr
            )
        x, y = get_batch(train_data, config.batch_size, config.seq_len, rng=rng)

        logits = model.forward(x)
        B, T, V = logits.shape
        loss, dlogits_flat = cross_entropy_loss(logits.reshape(-1, V), y.reshape(-1))
        history.train_loss.append(loss)

        model.backward(dlogits_flat.reshape(B, T, V))
        optimizer.step()

        step1 = step + 1
        if config.log_every and step1 % config.log_every == 0:
            avg = float(np.mean(history.train_loss[-config.log_every :]))
            sps = step1 / (time.time() - t0)
            print(f"step {step1:5d}/{config.n_steps} | loss {avg:.4f} | {sps:.1f} steps/s")

        if val_data is not None and config.eval_every and step1 % config.eval_every == 0:
            model.set_training(False)
            vl = eval_loss(
                model,
                val_data,
                config.batch_size,
                config.seq_len,
                config.eval_batches,
                eval_rng,
            )
            model.set_training(True)
            history.val_loss.append((step1, vl))
            if config.log_every:
                print(f"step {step1:5d}/{config.n_steps} | val_loss {vl:.4f}")

    return history
