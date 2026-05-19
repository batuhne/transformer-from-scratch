"""Tests for char vocab, batch sampling, and train/val split."""

from __future__ import annotations

import numpy as np
import pytest

from transformer.data import CharVocab, get_batch, split_data


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


def test_split_data_contiguous_with_no_overlap() -> None:
    data = np.arange(100, dtype=np.int64)
    train_data, val_data = split_data(data, val_fraction=0.1)
    assert len(train_data) == 90
    assert len(val_data) == 10
    assert train_data[-1] + 1 == val_data[0]
    assert np.array_equal(np.concatenate([train_data, val_data]), data)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_split_data_rejects_invalid_fraction(bad: float) -> None:
    data = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError):
        split_data(data, val_fraction=bad)
