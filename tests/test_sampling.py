"""Tests for logit filtering used in sampling."""

from __future__ import annotations

import numpy as np
import pytest

from transformer.linear import softmax
from transformer.sampling import top_k_filter, top_p_filter


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


def test_top_p_kept_mass_reaches_p() -> None:
    # probs after softmax: roughly [0.644, 0.236, 0.087, 0.032] for these logits
    logits = np.log(np.array([0.60, 0.22, 0.10, 0.08]))
    filtered = top_p_filter(logits, p=0.9)
    kept = np.isfinite(filtered)
    # Renormalized kept mass must cover at least p of the original distribution.
    assert softmax(logits)[kept].sum() >= 0.9
    # And the set must be minimal: dropping the last survivor would fall below p.
    n_kept = int(kept.sum())
    top_probs = np.sort(softmax(logits))[::-1]
    assert top_probs[: n_kept - 1].sum() < 0.9


def test_top_p_peaked_distribution_keeps_single_token() -> None:
    logits = np.log(np.array([0.97, 0.02, 0.01]))
    filtered = top_p_filter(logits, p=0.9)
    assert int(np.isfinite(filtered).sum()) == 1
    assert filtered[0] == logits[0]


def test_top_p_does_not_mutate_input() -> None:
    logits = np.array([1.0, 2.0, 3.0])
    top_p_filter(logits, p=0.5)
    assert np.array_equal(logits, np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("k", [0, -1, -100])
def test_top_k_rejects_nonpositive_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        top_k_filter(np.array([1.0, 2.0, 3.0]), k=k)


@pytest.mark.parametrize("p", [0.0, -0.1, 1.5, 2.0])
def test_top_p_rejects_out_of_range(p: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        top_p_filter(np.array([1.0, 2.0, 3.0]), p=p)


def test_top_p_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="1D"):
        top_p_filter(np.array([[1.0, 2.0], [3.0, 4.0]]), p=0.9)


def test_top_p_stable_under_ties() -> None:
    # Two equal-prob entries at the cutoff; stable sort guarantees deterministic
    # selection regardless of numpy partition internals.
    logits = np.array([2.0, 1.0, 1.0, 0.0])
    a = top_p_filter(logits, p=0.5)
    b = top_p_filter(logits, p=0.5)
    assert np.array_equal(np.isneginf(a), np.isneginf(b))
