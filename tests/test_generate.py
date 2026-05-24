"""Sanity tests for autoregressive generation."""

from __future__ import annotations

import numpy as np
import pytest

from transformer.data import CharVocab
from transformer.generate import generate
from transformer.model import Transformer


def _tiny_setup() -> tuple[Transformer, CharVocab]:
    text = "abcdefghij"
    vocab = CharVocab.from_text(text)
    model = Transformer(
        vocab_size=vocab.size,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        max_seq_len=8,
    )
    return model, vocab


def test_generate_returns_expected_length() -> None:
    model, vocab = _tiny_setup()
    out = generate(model, vocab, start="ab", max_new_tokens=20, temperature=1.0)
    assert len(out) == 2 + 20
    assert set(out).issubset(set(vocab.chars))


def test_generate_is_deterministic_with_seeded_rng() -> None:
    model, vocab = _tiny_setup()
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    a = generate(model, vocab, start="a", max_new_tokens=15, rng=rng1)
    b = generate(model, vocab, start="a", max_new_tokens=15, rng=rng2)
    assert a == b


def test_generate_rejects_nonpositive_temperature() -> None:
    model, vocab = _tiny_setup()
    with pytest.raises(ValueError):
        generate(model, vocab, start="a", max_new_tokens=1, temperature=0.0)


def test_generate_kv_cache_matches_no_cache() -> None:
    model, vocab = _tiny_setup()
    a = generate(model, vocab, start="ab", max_new_tokens=4,
                 temperature=1.0, rng=np.random.default_rng(0), use_cache=False)
    b = generate(model, vocab, start="ab", max_new_tokens=4,
                 temperature=1.0, rng=np.random.default_rng(0), use_cache=True)
    assert a == b


def test_generate_kv_cache_rejects_overlong_request() -> None:
    model, vocab = _tiny_setup()
    with pytest.raises(ValueError):
        generate(model, vocab, start="abc", max_new_tokens=10, use_cache=True)
