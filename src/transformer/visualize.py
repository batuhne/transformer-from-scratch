"""Attention heatmap visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from transformer.data import CharVocab
from transformer.model import Transformer


def _weights_and_tokens(
    model: Transformer,
    vocab: CharVocab,
    prompt: str,
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    model.set_training(False)
    model.forward(np.array([vocab.encode(prompt)]))
    attn = model.blocks[layer].mha.attn_weights[0]
    tokens = [c if c != "\n" else "\\n" for c in prompt]
    return attn, tokens


def _draw(ax: Axes, weights: np.ndarray, tokens: list[str], fontsize: int) -> None:
    T = len(tokens)
    ax.imshow(weights[:T, :T], cmap="hot", vmin=0)
    ax.set_xticks(range(T))
    ax.set_xticklabels(tokens, fontsize=fontsize, rotation=90)
    ax.set_yticks(range(T))
    ax.set_yticklabels(tokens, fontsize=fontsize)


def plot_attention(
    model: Transformer,
    vocab: CharVocab,
    prompt: str,
    layer: int = 0,
    head: int = 0,
    ax: Axes | None = None,
) -> Axes:
    """Heatmap of one head's attention weights over the prompt."""
    attn, tokens = _weights_and_tokens(model, vocab, prompt, layer)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    _draw(ax, attn[head], tokens, fontsize=8)
    ax.set_xlabel("key (attended to)")
    ax.set_ylabel("query (attending from)")
    ax.set_title(f"Block {layer}, head {head}")
    return ax


def plot_all_heads(
    model: Transformer,
    vocab: CharVocab,
    prompt: str,
    layer: int = 0,
) -> Figure:
    """Grid of every head's attention map for one layer."""
    attn, tokens = _weights_and_tokens(model, vocab, prompt, layer)
    n_heads = attn.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(3.2 * n_heads, 3.4), squeeze=False)
    for h in range(n_heads):
        _draw(axes[0, h], attn[h], tokens, fontsize=6)
        axes[0, h].set_title(f"head {h}")
    fig.suptitle(f"Block {layer}: attention per head")
    fig.tight_layout()
    return fig
