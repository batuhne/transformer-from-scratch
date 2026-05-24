"""Smoke tests for attention visualization (Agg backend, no display)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from transformer.data import CharVocab
from transformer.model import Transformer
from transformer.visualize import plot_all_heads, plot_attention


def _setup() -> tuple[Transformer, CharVocab]:
    vocab = CharVocab.from_text("abcdefgh")
    model = Transformer(
        vocab_size=vocab.size, d_model=8, n_heads=2, d_ff=16, n_layers=2, max_seq_len=8
    )
    return model, vocab


def test_plot_attention_returns_axis_with_square_extent() -> None:
    model, vocab = _setup()
    ax = plot_attention(model, vocab, prompt="abcd", layer=1, head=1)
    assert len(ax.get_xticks()) == 4
    assert len(ax.get_yticks()) == 4


def test_plot_all_heads_creates_one_axis_per_head() -> None:
    model, vocab = _setup()
    fig = plot_all_heads(model, vocab, prompt="abc", layer=0)
    assert len(fig.axes) == 2
