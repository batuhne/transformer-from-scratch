"""Learning rate schedules: linear warmup followed by cosine decay."""

from __future__ import annotations

import math


def cosine_warmup_lr(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup then cosine decay. `step` is 1-indexed."""
    if step <= 0:
        return 0.0
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
