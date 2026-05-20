"""End-to-end smoke and gradient check for the full Transformer."""

from __future__ import annotations

import numpy as np
from conftest import numerical_gradient, relative_error

from transformer.linear import cross_entropy_loss
from transformer.model import Transformer


def _tiny_model() -> Transformer:
    return Transformer(
        vocab_size=7,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        max_seq_len=16,
    )


def test_model_forward_shape_and_initial_loss_near_random() -> None:
    vocab_size = 7
    model = _tiny_model()
    indices = np.random.randint(0, vocab_size, size=(3, 5))
    logits = model.forward(indices)
    assert logits.shape == (3, 5, vocab_size)

    targets = np.random.randint(0, vocab_size, size=(3, 5))
    loss, _ = cross_entropy_loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
    # Untrained Xavier init should sit within an order of magnitude of ln(V).
    assert 0.5 * np.log(vocab_size) < loss < 3.0 * np.log(vocab_size)


def test_model_backward_populates_all_parameter_gradients() -> None:
    vocab_size = 7
    model = _tiny_model()
    indices = np.random.randint(0, vocab_size, size=(2, 4))
    logits = model.forward(indices)
    targets = np.random.randint(0, vocab_size, size=(2, 4))

    B, T, V = logits.shape
    _, dlogits_flat = cross_entropy_loss(logits.reshape(-1, V), targets.reshape(-1))
    model.backward(dlogits_flat.reshape(B, T, V))

    for obj, name in model.params():
        grad = getattr(obj, "d" + name)
        param = getattr(obj, name)
        assert grad is not None, f"missing grad for {type(obj).__name__}.{name}"
        assert grad.shape == param.shape


def test_model_output_proj_bias_gradient_matches_numerical() -> None:
    """End-to-end check that gradient flows correctly through the full stack."""
    vocab_size = 5
    model = Transformer(
        vocab_size=vocab_size, d_model=4, n_heads=2, d_ff=8, n_layers=1, max_seq_len=8
    )
    indices = np.random.randint(0, vocab_size, size=(2, 3))
    targets = np.random.randint(0, vocab_size, size=(2, 3))

    logits = model.forward(indices)
    B, T, V = logits.shape
    _, dlogits_flat = cross_entropy_loss(logits.reshape(-1, V), targets.reshape(-1))
    model.backward(dlogits_flat.reshape(B, T, V))

    def loss_with_b(b: np.ndarray) -> float:
        model.output_proj.b = b
        logits = model.forward(indices)
        loss, _ = cross_entropy_loss(logits.reshape(-1, V), targets.reshape(-1))
        return loss

    num_db = numerical_gradient(loss_with_b, model.output_proj.b.copy())
    assert relative_error(model.output_proj.db, num_db) < 1e-5


def test_model_dropout_enabled_forward_backward_produces_finite_grads() -> None:
    np.random.seed(0)
    vocab_size = 7
    model = Transformer(
        vocab_size=vocab_size,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=2,
        max_seq_len=16,
        dropout=0.2,
    )
    model.set_training(True)
    indices = np.random.randint(0, vocab_size, size=(2, 4))
    targets = np.random.randint(0, vocab_size, size=(2, 4))
    logits = model.forward(indices)
    B, T, V = logits.shape
    _, dlogits_flat = cross_entropy_loss(logits.reshape(-1, V), targets.reshape(-1))
    model.backward(dlogits_flat.reshape(B, T, V))
    for obj, name in model.params():
        grad = getattr(obj, "d" + name)
        assert np.all(np.isfinite(grad)), f"non-finite grad in {type(obj).__name__}.{name}"


def test_model_set_training_toggles_every_dropout() -> None:
    model = Transformer(
        vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_layers=2, max_seq_len=16, dropout=0.1
    )
    drops = model.dropouts()
    assert len(drops) == 2 * 3, "expected 3 dropouts per block (attn + 2 resid)"
    model.set_training(False)
    assert all(not d.training for d in drops)
    model.set_training(True)
    assert all(d.training for d in drops)


def test_model_eval_mode_is_deterministic_under_dropout() -> None:
    """Same input twice in eval mode produces identical logits even with dropout>0."""
    np.random.seed(0)
    model = Transformer(
        vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_layers=2, max_seq_len=16, dropout=0.5
    )
    model.set_training(False)
    indices = np.random.randint(0, 7, size=(2, 4))
    logits1 = model.forward(indices)
    logits2 = model.forward(indices)
    assert np.array_equal(logits1, logits2)
