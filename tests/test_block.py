"""End-to-end weight gradient check through a full TransformerBlock."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import numerical_gradient, relative_error

from transformer.attention import causal_mask
from transformer.block import TransformerBlock

_WEIGHT_SPECS = [
    ("ln1", "gamma"),
    ("ln1", "beta"),
    ("ln2", "gamma"),
    ("ln2", "beta"),
    ("mha", "W_Q"),
    ("mha", "W_K"),
    ("mha", "W_V"),
    ("mha", "W_O"),
]


def _setup() -> tuple[TransformerBlock, np.ndarray, np.ndarray, np.ndarray]:
    # dropout=0 so numerical_gradient's two forward passes see identical activations.
    block = TransformerBlock(d_model=8, n_heads=2, d_ff=16, dropout=0.0)
    x = np.random.randn(1, 4, 8)
    mask = causal_mask(4)
    out = block.forward(x, mask=mask)
    dout = np.random.randn(*out.shape)
    block.backward(dout)
    return block, x, mask, dout


@pytest.mark.parametrize("owner,attr", _WEIGHT_SPECS)
def test_block_weight_gradient_matches_numerical(owner: str, attr: str) -> None:
    block, x, mask, dout = _setup()
    target = getattr(block, owner)
    analytical = getattr(target, "d" + attr).copy()

    def loss(W: np.ndarray) -> float:
        setattr(target, attr, W)
        return float(np.sum(block.forward(x, mask=mask) * dout))

    num = numerical_gradient(loss, getattr(target, attr).copy())
    assert relative_error(analytical, num) < 1e-5, f"{owner}.{attr} gradient mismatch"


@pytest.mark.parametrize("layer_attr", ["linear1", "linear2"])
@pytest.mark.parametrize("param", ["W", "b"])
def test_block_ffn_gradient_matches_numerical(layer_attr: str, param: str) -> None:
    block, x, mask, dout = _setup()
    target = getattr(block.ffn, layer_attr)
    analytical = getattr(target, "d" + param).copy()

    def loss(W: np.ndarray) -> float:
        setattr(target, param, W)
        return float(np.sum(block.forward(x, mask=mask) * dout))

    num = numerical_gradient(loss, getattr(target, param).copy())
    assert relative_error(analytical, num) < 1e-5, (
        f"ffn.{layer_attr}.{param} gradient mismatch"
    )


def test_block_residual_propagates_input_gradient() -> None:
    """dL/dx must pick up both the residual identity and the sublayer paths."""
    block, x, mask, dout = _setup()
    dx_analytical = block.backward(dout)

    def loss(x_in: np.ndarray) -> float:
        return float(np.sum(block.forward(x_in, mask=mask) * dout))

    dx_numerical = numerical_gradient(loss, x.copy())
    assert relative_error(dx_analytical, dx_numerical) < 1e-5
