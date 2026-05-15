"""Pre-LayerNorm decoder transformer block: MHA sub-layer + FFN sub-layer."""

from __future__ import annotations

import numpy as np

from transformer.ffn import FeedForward
from transformer.layernorm import LayerNorm
from transformer.mha import MultiHeadAttention


class TransformerBlock:
    """Pre-LN block: y = x + MHA(LN(x)); z = y + FFN(LN(y))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        x = x + self.mha.forward(self.ln1.forward(x), mask=mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # FFN sub-layer: dout splits between residual path and sublayer path.
        d_after_attn = dout + self.ln2.backward(self.ffn.backward(dout))
        # MHA sub-layer: same split.
        return d_after_attn + self.ln1.backward(self.mha.backward(d_after_attn))

    def params(self) -> list[tuple[object, str]]:
        return [
            (self.ln1, "gamma"),
            (self.ln1, "beta"),
            (self.mha, "W_Q"),
            (self.mha, "W_K"),
            (self.mha, "W_V"),
            (self.mha, "W_O"),
            (self.ln2, "gamma"),
            (self.ln2, "beta"),
            *self.ffn.params(),
        ]
