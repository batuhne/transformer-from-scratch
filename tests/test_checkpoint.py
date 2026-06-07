"""Round-trip tests for weight save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transformer.checkpoint import load_pretrained, load_weights, save_weights
from transformer.data import CharVocab
from transformer.model import Transformer


def _model(tie: bool = False) -> Transformer:
    return Transformer(
        vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_layers=2,
        max_seq_len=8, tie_weights=tie,
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    a = _model()
    x = np.random.randint(0, 7, size=(2, 5))
    out_a = a.forward(x)

    save_weights(a, tmp_path / "w.npz")
    b = _model()  # different random init
    assert not np.allclose(b.forward(x), out_a), "fresh model should differ first"

    load_weights(b, tmp_path / "w.npz")
    assert np.allclose(b.forward(x), out_a)


def test_load_preserves_weight_tying(tmp_path: Path) -> None:
    a = _model(tie=True)
    save_weights(a, tmp_path / "w.npz")
    b = _model(tie=True)
    load_weights(b, tmp_path / "w.npz")
    assert np.shares_memory(b.embedding.W, b.output_proj.W)
    assert np.array_equal(b.output_proj.W, b.embedding.W.T)


def test_load_rejects_shape_mismatch(tmp_path: Path) -> None:
    a = _model()
    save_weights(a, tmp_path / "w.npz")
    # Smaller model: same architecture except d_model differs, so shapes mismatch.
    b = Transformer(
        vocab_size=7, d_model=4, n_heads=2, d_ff=8, n_layers=2,
        max_seq_len=8, tie_weights=False,
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        load_weights(b, tmp_path / "w.npz")


def test_load_rejects_missing_key(tmp_path: Path) -> None:
    a = _model()
    save_weights(a, tmp_path / "w.npz")
    # Different layer count produces a different param list, so some indices miss.
    b = Transformer(
        vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_layers=3,
        max_seq_len=8, tie_weights=False,
    )
    with pytest.raises(KeyError):
        load_weights(b, tmp_path / "w.npz")


def test_load_pretrained_rebuilds_model_and_vocab(tmp_path: Path) -> None:
    vocab = CharVocab.from_text("abcdefg")
    a = _model(tie=True)
    x = np.random.randint(0, 7, size=(2, 5))
    out_a = a.forward(x)

    save_weights(a, tmp_path / "w.npz", vocab=vocab)
    b, vocab_b = load_pretrained(tmp_path / "w.npz")
    assert vocab_b is not None
    assert vocab_b.chars == vocab.chars
    assert b.vocab_size == a.vocab_size
    assert b.d_model == a.d_model
    assert np.allclose(b.forward(x), out_a)


def test_load_pretrained_without_vocab(tmp_path: Path) -> None:
    a = _model()
    save_weights(a, tmp_path / "w.npz")  # no vocab passed
    _, vocab_b = load_pretrained(tmp_path / "w.npz")
    assert vocab_b is None
