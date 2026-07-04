"""Tests for set_seed: two runs with the same seed must match bit-for-bit."""

from __future__ import annotations

import numpy as np

from transformer.data import CharVocab
from transformer.model import Transformer
from transformer.train import TrainConfig, train
from transformer.utils import set_seed


def _one_run(seed: int) -> list[float]:
    rng = set_seed(seed)
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
        dropout=0.1,
        rng=rng,
    )
    config = TrainConfig(n_steps=30, batch_size=8, seq_len=8, lr=1e-2, log_every=0)
    return train(model, data, config, rng=rng).train_loss


def test_set_seed_produces_identical_loss_histories() -> None:
    a = _one_run(seed=123)
    b = _one_run(seed=123)
    assert a == b


def test_different_seeds_produce_different_loss_histories() -> None:
    a = _one_run(seed=1)
    b = _one_run(seed=2)
    assert a != b


def test_train_loss_is_invariant_to_eval_cadence() -> None:
    """Adding val passes must not perturb the training stream.

    Before fix C11, eval_loss drew batches from the same rng as training; toggling
    eval_every shifted every subsequent training batch. With a spawned eval rng,
    train_loss is identical regardless of eval cadence.
    """

    def _run(eval_every: int) -> list[float]:
        rng = set_seed(123)
        text = "abcabcabcabc" * 200
        vocab = CharVocab.from_text(text)
        data = np.array(vocab.encode(text), dtype=np.int64)
        val = data[-64:]
        model = Transformer(
            vocab_size=vocab.size,
            d_model=16,
            n_heads=2,
            d_ff=32,
            n_layers=1,
            max_seq_len=8,
            dropout=0.1,
            rng=rng,
        )
        config = TrainConfig(
            n_steps=30,
            batch_size=8,
            seq_len=8,
            lr=1e-2,
            log_every=0,
            eval_every=eval_every,
            eval_batches=3,
        )
        return train(model, data, config, val_data=val, rng=rng).train_loss

    assert _run(eval_every=5) == _run(eval_every=10)
