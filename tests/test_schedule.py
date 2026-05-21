"""Tests for the linear warmup + cosine decay learning rate schedule."""

from __future__ import annotations

import math

import pytest

from transformer.schedule import cosine_warmup_lr


def test_schedule_warmup_first_step_is_one_over_warmup_fraction() -> None:
    lr = cosine_warmup_lr(step=1, base_lr=1.0, warmup_steps=100, total_steps=1000)
    assert lr == pytest.approx(0.01, abs=1e-9)


def test_schedule_warmup_grows_linearly() -> None:
    for s in [10, 25, 50, 75]:
        lr = cosine_warmup_lr(s, base_lr=1.0, warmup_steps=100, total_steps=1000)
        assert lr == pytest.approx(s / 100, abs=1e-9)


def test_schedule_at_warmup_boundary_returns_base_lr() -> None:
    lr = cosine_warmup_lr(step=100, base_lr=1.0, warmup_steps=100, total_steps=1000)
    assert lr == pytest.approx(1.0, abs=1e-9)


def test_schedule_at_total_steps_returns_min_lr() -> None:
    lr = cosine_warmup_lr(
        step=1000, base_lr=1.0, warmup_steps=100, total_steps=1000, min_lr=0.05
    )
    assert lr == pytest.approx(0.05, abs=1e-9)


def test_schedule_midway_cosine_value_is_correct() -> None:
    """Midway through the post-warmup phase, progress = 0.5 so cos(pi/2) = 0,
    and lr = min_lr + 0.5*(base_lr - min_lr)."""
    lr = cosine_warmup_lr(
        step=550, base_lr=1.0, warmup_steps=100, total_steps=1000, min_lr=0.0
    )
    # progress = (550 - 100) / (1000 - 100) = 450/900 = 0.5
    expected = 0.0 + 0.5 * (1.0 - 0.0) * (1.0 + math.cos(math.pi * 0.5))
    assert lr == pytest.approx(expected, abs=1e-9)


def test_schedule_with_no_warmup_starts_near_base_lr() -> None:
    lr = cosine_warmup_lr(step=1, base_lr=1.0, warmup_steps=0, total_steps=1000)
    # progress = 1/1000 ~= 0; cos(0.001*pi) ~= 1; lr ~= base_lr
    assert lr == pytest.approx(1.0, abs=1e-3)


def test_schedule_clamps_step_beyond_total() -> None:
    lr = cosine_warmup_lr(
        step=2000, base_lr=1.0, warmup_steps=100, total_steps=1000, min_lr=0.1
    )
    assert lr == pytest.approx(0.1, abs=1e-9)
