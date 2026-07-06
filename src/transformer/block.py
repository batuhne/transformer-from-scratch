"""Pre-LayerNorm decoder transformer block: MHA sub-layer + FFN sub-layer."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from transformer.dropout import Dropout
from transformer.ffn import FeedForward
from transformer.layernorm import LayerNorm
from transformer.mha import MultiHeadAttention


class TransformerBlock:
    """Pre-LN block: y = x + drop(MHA(LN(x))); z = y + drop(FFN(LN(y)))."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
        dtype: DTypeLike = np.float64,
    ) -> None:
        self.ln1 = LayerNorm(d_model, dtype=dtype)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout=dropout, rng=rng, dtype=dtype)
        self.drop1 = Dropout(dropout, rng=rng)
        self.ln2 = LayerNorm(d_model, dtype=dtype)
        self.ffn = FeedForward(d_model, d_ff, rng=rng, dtype=dtype)
        self.drop2 = Dropout(dropout, rng=rng)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        x = x + self.drop1.forward(self.mha.forward(self.ln1.forward(x), mask=mask))
        x = x + self.drop2.forward(self.ffn.forward(self.ln2.forward(x)))
        return x

    def forward_step(
        self,
        x: np.ndarray,
        kv_cache: dict[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Incremental block forward; dropout skipped (eval-only path)."""
        kv = (kv_cache["K"], kv_cache["V"]) if kv_cache is not None else None
        attn_out, K_new, V_new = self.mha.forward_step(self.ln1.forward(x), kv)
        x = x + attn_out
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x, {"K": K_new, "V": V_new}

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # FFN sub-layer: dout splits between residual path and sublayer path.
        d_after_attn = dout + self.ln2.backward(self.ffn.backward(self.drop2.backward(dout)))
        # MHA sub-layer: same split.
        d_mha = self.mha.backward(self.drop1.backward(d_after_attn))
        return d_after_attn + self.ln1.backward(d_mha)

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

    def dropouts(self) -> list[Dropout]:
        return [self.mha.attn_dropout, self.drop1, self.drop2]
