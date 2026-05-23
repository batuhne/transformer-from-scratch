"""Decoder-only transformer for character-level language modeling."""

from __future__ import annotations

import numpy as np

from transformer.attention import causal_mask
from transformer.block import TransformerBlock
from transformer.dropout import Dropout
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
        dropout: float = 0.0,
        tie_weights: bool = False,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights

        self.embedding = Embedding(vocab_size, d_model)
        self.pe = get_positional_encoding(max_seq_len, d_model)
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)
        self.output_proj = Linear(d_model, vocab_size)
        if tie_weights:
            # output_proj.W becomes a transposed view of embedding.W: the two
            # share storage, so in-place Adam updates to embedding.W propagate.
            self.output_proj.W = self.embedding.W.T

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """Map token indices (B, T) to logits (B, T, vocab_size)."""
        T = indices.shape[1]
        x = self.embedding.forward(indices) + self.pe[:T]
        mask = causal_mask(T)
        for block in self.blocks:
            x = block.forward(x, mask=mask)
        x = self.ln_final.forward(x)
        return self.output_proj.forward(x)

    def init_caches(self) -> list[dict[str, np.ndarray] | None]:
        return [None] * len(self.blocks)

    def forward_step(
        self,
        indices: np.ndarray,
        caches: list[dict[str, np.ndarray] | None],
        position: int,
    ) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        """Incremental forward; `position` is the absolute pos of the first new token."""
        T_new = indices.shape[1]
        if position + T_new > self.max_seq_len:
            raise ValueError(
                f"position+T_new={position + T_new} exceeds max_seq_len={self.max_seq_len}"
            )
        x = self.embedding.forward(indices) + self.pe[position : position + T_new]
        new_caches: list[dict[str, np.ndarray]] = []
        for block, cache in zip(self.blocks, caches):
            x, new_cache = block.forward_step(x, cache)
            new_caches.append(new_cache)
        x = self.ln_final.forward(x)
        return self.output_proj.forward(x), new_caches

    def backward(self, dlogits: np.ndarray) -> None:
        dx = self.output_proj.backward(dlogits)
        dx = self.ln_final.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        self.embedding.backward(dx)
        if self.tie_weights:
            # Shared W collects gradient from both the lookup and the output
            # projection; fold the projection contribution into embedding.dW.
            self.embedding.dW += self.output_proj.dW.T

    def params(self) -> list[tuple[object, str]]:
        out: list[tuple[object, str]] = [(self.embedding, "W")]
        for block in self.blocks:
            out.extend(block.params())
        out.extend([(self.ln_final, "gamma"), (self.ln_final, "beta")])
        if not self.tie_weights:
            out.append((self.output_proj, "W"))
        out.append((self.output_proj, "b"))
        return out

    def count_params(self) -> int:
        return sum(getattr(obj, name).size for obj, name in self.params())

    def dropouts(self) -> list[Dropout]:
        return [d for block in self.blocks for d in block.dropouts()]

    def set_training(self, training: bool) -> None:
        """Toggle every Dropout in the model. Disable for val/inference."""
        for d in self.dropouts():
            d.training = training
