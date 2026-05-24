"""Tests for deterministic perplexity."""

from __future__ import annotations

import numpy as np
import pytest

from transformer.evaluate import perplexity
from transformer.model import Transformer


def _setup() -> tuple[Transformer, np.ndarray]:
    model = Transformer(
        vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_layers=1, max_seq_len=8
    )
    data = np.arange(7).repeat(20) % 7
    return model, data


def test_perplexity_is_deterministic() -> None:
    model, data = _setup()
    assert perplexity(model, data, batch_size=4, seq_len=8) == perplexity(
        model, data, batch_size=4, seq_len=8
    )


def test_perplexity_independent_of_batch_size() -> None:
    """Token weighting makes the result invariant to how windows are batched."""
    model, data = _setup()
    a = perplexity(model, data, batch_size=2, seq_len=8)
    b = perplexity(model, data, batch_size=5, seq_len=8)
    assert np.isclose(a, b)


def test_untrained_perplexity_near_vocab_size() -> None:
    model, data = _setup()
    ppl = perplexity(model, data, batch_size=4, seq_len=8)
    # Xavier-init model is roughly uniform: perplexity within 2x of vocab size.
    assert 1.0 < ppl < 2 * 7


def test_perplexity_rejects_too_short_data() -> None:
    model, _ = _setup()
    with pytest.raises(ValueError):
        perplexity(model, np.arange(5), batch_size=1, seq_len=8)
