"""Smoke test: training reduces loss on a tiny corpus."""

from __future__ import annotations

import numpy as np

from transformer.data import CharVocab, split_data
from transformer.model import Transformer
from transformer.train import TrainConfig, train


def test_train_reduces_loss_on_repetitive_corpus() -> None:
    text = "abcabcabcabc" * 200
    vocab = CharVocab.from_text(text)
    data = np.array(vocab.encode(text), dtype=np.int64)

    model = Transformer(
        vocab_size=vocab.size,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1,
        max_seq_len=8,
    )
    config = TrainConfig(n_steps=200, batch_size=8, seq_len=8, lr=1e-2, log_every=0)
    history = train(model, data, config, rng=np.random.default_rng(0))

    assert len(history.train_loss) == config.n_steps
    initial = float(np.mean(history.train_loss[:20]))
    final = float(np.mean(history.train_loss[-20:]))
    assert final < 0.5 * initial, f"loss did not drop: {initial:.3f} -> {final:.3f}"


def test_train_records_val_loss_when_val_data_provided() -> None:
    text = "abcabcabcabc" * 200
    vocab = CharVocab.from_text(text)
    data = np.array(vocab.encode(text), dtype=np.int64)
    train_data, val_data = split_data(data, val_fraction=0.1)

    model = Transformer(
        vocab_size=vocab.size,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1,
        max_seq_len=8,
    )
    config = TrainConfig(
        n_steps=50,
        batch_size=8,
        seq_len=8,
        lr=1e-2,
        log_every=0,
        eval_every=25,
        eval_batches=3,
    )
    history = train(model, train_data, config, val_data=val_data, rng=np.random.default_rng(0))

    assert [step for step, _ in history.val_loss] == [25, 50]
    assert all(np.isfinite(loss) for _, loss in history.val_loss)
