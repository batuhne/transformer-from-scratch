"""Gradient checks for MultiHeadAttention."""

from __future__ import annotations

import numpy as np
from conftest import numerical_gradient, relative_error

from transformer.attention import causal_mask
from transformer.mha import MultiHeadAttention


def _setup() -> tuple[MultiHeadAttention, np.ndarray, np.ndarray, np.ndarray]:
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = np.random.randn(2, 4, 8)
    mask = causal_mask(4)
    out = mha.forward(x, mask=mask)
    dout = np.random.randn(*out.shape)
    mha.backward(dout)
    return mha, x, mask, dout


def _loss_factory(mha, name, x, mask, dout):
    def loss(W):
        setattr(mha, name, W)
        return float(np.sum(mha.forward(x, mask=mask) * dout))

    return loss


def test_mha_projection_gradients_match_numerical() -> None:
    mha, x, mask, dout = _setup()
    for name in ("W_Q", "W_K", "W_V", "W_O"):
        num = numerical_gradient(_loss_factory(mha, name, x, mask, dout), getattr(mha, name).copy())
        analytical = getattr(mha, f"d{name}")
        assert relative_error(analytical, num) < 1e-5, f"{name} gradient mismatch"


def test_mha_dx_matches_numerical() -> None:
    mha, x, mask, dout = _setup()
    dx = mha.backward(dout)

    def loss(x_in: np.ndarray) -> float:
        return float(np.sum(mha.forward(x_in, mask=mask) * dout))

    num_dx = numerical_gradient(loss, x.copy())
    assert relative_error(dx, num_dx) < 1e-5


def test_forward_step_bulk_matches_forward() -> None:
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = np.random.randn(2, 5, 8)
    ref = mha.forward(x, mask=causal_mask(5))
    out, K, V = mha.forward_step(x, None)
    assert np.allclose(out, ref, atol=1e-10)
    assert K.shape == (2, 2, 5, 4) and V.shape == (2, 2, 5, 4)


def test_forward_step_incremental_matches_forward() -> None:
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = np.random.randn(1, 5, 8)
    ref = mha.forward(x, mask=causal_mask(5))
    kv = None
    outs = []
    for t in range(5):
        out_t, k, v = mha.forward_step(x[:, t : t + 1, :], kv)
        kv = (k, v)
        outs.append(out_t)
    incremental = np.concatenate(outs, axis=1)
    assert np.allclose(incremental, ref, atol=1e-10)


def test_forward_step_records_attn_weights() -> None:
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    # Prefill: 3 tokens, no cache. attn_weights: (B, n_heads, T_new, T_total).
    _, K, V = mha.forward_step(np.random.randn(1, 3, 8), None)
    assert mha.attn_weights is not None
    assert mha.attn_weights.shape == (1, 2, 3, 3)
    # Incremental: one new token attends to all 4 keys in the running cache.
    mha.forward_step(np.random.randn(1, 1, 8), (K, V))
    assert mha.attn_weights.shape == (1, 2, 1, 4)


def test_chunked_prefill_then_incremental_matches_full_forward() -> None:
    """Prompt prefill (T_new=3) + 2 incremental steps must equal forward(T=5)."""
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = np.random.randn(1, 5, 8)
    ref = mha.forward(x, mask=causal_mask(5))

    prefill_out, K, V = mha.forward_step(x[:, :3, :], None)
    step3_out, K, V = mha.forward_step(x[:, 3:4, :], (K, V))
    step4_out, K, V = mha.forward_step(x[:, 4:5, :], (K, V))
    combined = np.concatenate([prefill_out, step3_out, step4_out], axis=1)
    assert np.allclose(combined, ref, atol=1e-10)


def test_chunked_prefill_split_equals_single_prefill() -> None:
    """Prefilling 5 at once must equal prefilling (3, 2) sequentially."""
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = np.random.randn(1, 5, 8)

    full_out, K_full, V_full = mha.forward_step(x, None)

    chunk1_out, K, V = mha.forward_step(x[:, :3, :], None)
    chunk2_out, K, V = mha.forward_step(x[:, 3:, :], (K, V))
    split_out = np.concatenate([chunk1_out, chunk2_out], axis=1)

    assert np.allclose(split_out, full_out, atol=1e-10)
    assert np.allclose(K, K_full, atol=1e-10)
    assert np.allclose(V, V_full, atol=1e-10)


def test_chunked_step_attn_weights_shape() -> None:
    """A T_new=2 step over a T_cache=3 cache yields attn_weights of shape (B,H,2,5)."""
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    _, K, V = mha.forward_step(np.random.randn(1, 3, 8), None)
    mha.forward_step(np.random.randn(1, 2, 8), (K, V))
    assert mha.attn_weights.shape == (1, 2, 2, 5)


def test_mha_attention_weights_respect_causal_mask() -> None:
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = np.random.randn(1, 4, 8)
    mha.forward(x, mask=causal_mask(4))
    # Upper triangle of attention weights must be effectively zero per head.
    upper = np.triu(mha.attn_weights[0], k=1)
    assert np.allclose(upper, 0.0, atol=1e-8)
    # Each row of attention weights still sums to 1.
    row_sums = mha.attn_weights[0].sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
