"""End-to-end smoke and gradient check for the full Transformer."""

from __future__ import annotations

import numpy as np
from conftest import numerical_gradient, relative_error

from transformer.linear import cross_entropy_loss
from transformer.model import Transformer
from transformer.optim import Adam


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


def test_tie_weights_reduces_param_count_by_vocab_times_d_model() -> None:
    kwargs = dict(vocab_size=11, d_model=8, n_heads=2, d_ff=16, n_layers=2, max_seq_len=16)
    untied = Transformer(**kwargs, tie_weights=False)
    tied = Transformer(**kwargs, tie_weights=True)
    assert untied.count_params() - tied.count_params() == 11 * 8


def test_tie_weights_shares_storage_between_embedding_and_output_proj() -> None:
    model = Transformer(
        vocab_size=11,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        max_seq_len=16,
        tie_weights=True,
    )
    assert np.shares_memory(model.embedding.W, model.output_proj.W)
    assert np.array_equal(model.output_proj.W, model.embedding.W.T)


def test_tie_weights_combined_gradient_matches_numerical() -> None:
    np.random.seed(0)
    vocab_size = 5
    model = Transformer(
        vocab_size=vocab_size,
        d_model=4,
        n_heads=2,
        d_ff=8,
        n_layers=1,
        max_seq_len=8,
        tie_weights=True,
    )
    indices = np.random.randint(0, vocab_size, size=(2, 3))
    targets = np.random.randint(0, vocab_size, size=(2, 3))

    logits = model.forward(indices)
    B, T, V = logits.shape
    _, dlogits_flat = cross_entropy_loss(logits.reshape(-1, V), targets.reshape(-1))
    model.backward(dlogits_flat.reshape(B, T, V))
    analytical = model.embedding.dW.copy()

    def loss_at(W: np.ndarray) -> float:
        # In-place write preserves the output_proj.W transpose view.
        model.embedding.W[:] = W
        logits = model.forward(indices)
        loss, _ = cross_entropy_loss(logits.reshape(-1, V), targets.reshape(-1))
        return loss

    num = numerical_gradient(loss_at, model.embedding.W.copy())
    assert relative_error(analytical, num) < 1e-5


def test_tie_weights_persists_through_adam_step() -> None:
    np.random.seed(0)
    vocab_size = 5
    model = Transformer(
        vocab_size=vocab_size,
        d_model=4,
        n_heads=2,
        d_ff=8,
        n_layers=1,
        max_seq_len=8,
        tie_weights=True,
    )
    indices = np.random.randint(0, vocab_size, size=(2, 3))
    targets = np.random.randint(0, vocab_size, size=(2, 3))
    logits = model.forward(indices)
    B, T, V = logits.shape
    _, dlogits_flat = cross_entropy_loss(logits.reshape(-1, V), targets.reshape(-1))
    model.backward(dlogits_flat.reshape(B, T, V))

    optimizer = Adam(model.params(), lr=1e-2)
    optimizer.step()

    assert np.shares_memory(model.embedding.W, model.output_proj.W)
    assert np.array_equal(model.output_proj.W, model.embedding.W.T)


def test_forward_step_bulk_matches_forward() -> None:
    np.random.seed(0)
    model = _tiny_model()
    model.set_training(False)
    indices = np.random.randint(0, 7, size=(2, 6))
    ref = model.forward(indices)
    logits, _ = model.forward_step(indices, model.init_caches(), position=0)
    assert np.allclose(logits, ref, atol=1e-10)


def test_forward_step_incremental_matches_forward() -> None:
    np.random.seed(0)
    model = _tiny_model()
    model.set_training(False)
    indices = np.random.randint(0, 7, size=(1, 6))
    ref = model.forward(indices)
    caches = model.init_caches()
    outs = []
    for t in range(indices.shape[1]):
        logits_t, caches = model.forward_step(indices[:, t : t + 1], caches, position=t)
        outs.append(logits_t)
    incremental = np.concatenate(outs, axis=1)
    assert np.allclose(incremental, ref, atol=1e-10)


def test_registries_cover_every_param_and_dropout() -> None:
    """params() and dropouts() must list every (W,dW) pair and Dropout reachable from model."""
    from transformer.dropout import Dropout

    model = Transformer(
        vocab_size=7, d_model=8, n_heads=2, d_ff=16, n_layers=2, max_seq_len=16, dropout=0.1
    )
    registered_params = {(id(obj), name) for obj, name in model.params()}
    registered_drops = {id(d) for d in model.dropouts()}

    discovered_params: set[tuple[int, str]] = set()
    discovered_drops: set[int] = set()

    def walk(o: object, seen: set[int]) -> None:
        if id(o) in seen or not hasattr(o, "__dict__"):
            return
        seen.add(id(o))
        if isinstance(o, Dropout):
            discovered_drops.add(id(o))
        for k, v in vars(o).items():
            # convention: a trainable array `name` always has a sibling `dname`.
            if isinstance(v, np.ndarray) and not k.startswith("d") and hasattr(o, "d" + k):
                discovered_params.add((id(o), k))
            if isinstance(v, list):
                for item in v:
                    walk(item, seen)
            elif not isinstance(v, (np.ndarray, dict, str, int, float, bool, type(None))):
                walk(v, seen)

    walk(model, set())
    # tie_weights shares storage between embedding.W and output_proj.W; params()
    # only lists embedding.W in that case, so allow the discovered output_proj.W.
    if model.tie_weights:
        discovered_params.discard((id(model.output_proj), "W"))

    assert discovered_drops == registered_drops, (
        f"dropouts() missing: {discovered_drops - registered_drops}"
    )
    assert discovered_params == registered_params, (
        f"params() missing: {discovered_params - registered_params}"
    )


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
