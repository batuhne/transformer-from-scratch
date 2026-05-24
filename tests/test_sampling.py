"""Tests for logit filtering used in sampling."""

from __future__ import annotations

import numpy as np

from transformer.sampling import top_k_filter


def test_top_k_keeps_exactly_k_finite_logits() -> None:
    logits = np.array([1.0, 5.0, 2.0, 4.0, 3.0])
    filtered = top_k_filter(logits, k=2)
    assert np.sum(np.isfinite(filtered)) == 2
    # The two survivors must be the two largest values.
    assert filtered[1] == 5.0 and filtered[3] == 4.0
    assert np.isneginf(filtered[[0, 2, 4]]).all()


def test_top_k_larger_than_vocab_is_noop() -> None:
    logits = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(top_k_filter(logits, k=10), logits)


def test_top_k_does_not_mutate_input() -> None:
    logits = np.array([1.0, 5.0, 2.0])
    top_k_filter(logits, k=1)
    assert np.array_equal(logits, np.array([1.0, 5.0, 2.0]))
