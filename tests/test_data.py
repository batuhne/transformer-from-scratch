"""Tests for char vocab and batch sampling."""

from __future__ import annotations

import numpy as np

from transformer.data import CharVocab, get_batch


def test_charvocab_roundtrip() -> None:
    vocab = CharVocab.from_text("hello world")
    assert vocab.size == len(set("hello world"))
    s = "hello"
    assert vocab.decode(vocab.encode(s)) == s


def test_charvocab_chars_sorted_for_determinism() -> None:
    vocab1 = CharVocab.from_text("bca")
    vocab2 = CharVocab.from_text("acb")
    assert vocab1.chars == vocab2.chars


def test_get_batch_shapes_and_target_shift() -> None:
    data = np.arange(100, dtype=np.int64)
    rng = np.random.default_rng(seed=0)
    x, y = get_batch(data, batch_size=4, seq_len=8, rng=rng)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # Each y row should be x row shifted by one position.
    assert np.array_equal(y[:, :-1], x[:, 1:])
