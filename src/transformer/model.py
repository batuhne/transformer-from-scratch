"""Decoder-only transformer for character-level language modeling."""

from __future__ import annotations

import numpy as np

from transformer.attention import causal_mask
from transformer.block import TransformerBlock
from transformer.embedding import Embedding, get_positional_encoding
from transformer.layernorm import LayerNorm
from transformer.linear import Linear


class Transformer:
    """Embedding + N transformer blocks + final LN + output projection."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = Embedding(vocab_size, d_model)
        self.pe = get_positional_encoding(max_seq_len, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_final = LayerNorm(d_model)
        self.output_proj = Linear(d_model, vocab_size)

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """Map token indices (B, T) to logits (B, T, vocab_size)."""
        T = indices.shape[1]
        x = self.embedding.forward(indices) + self.pe[:T]
        mask = causal_mask(T)
        for block in self.blocks:
            x = block.forward(x, mask=mask)
        x = self.ln_final.forward(x)
        return self.output_proj.forward(x)

    def backward(self, dlogits: np.ndarray) -> None:
        dx = self.output_proj.backward(dlogits)
        dx = self.ln_final.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.embedding.backward(dx)

    def params(self) -> list[tuple[object, str]]:
        out: list[tuple[object, str]] = [(self.embedding, "W")]
        for block in self.blocks:
            out.extend(block.params())
        out.extend(
            [
                (self.ln_final, "gamma"),
                (self.ln_final, "beta"),
                (self.output_proj, "W"),
                (self.output_proj, "b"),
            ]
        )
        return out

    def count_params(self) -> int:
        return sum(getattr(obj, name).size for obj, name in self.params())
