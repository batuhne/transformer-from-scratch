"""Round-trip tests for weight save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from transformer.checkpoint import load_weights, save_weights
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
