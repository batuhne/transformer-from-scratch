"""Save and load model weights as a flat .npz, with optional config and vocab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from transformer.data import CharVocab
from transformer.model import Transformer

_CONFIG_KEY = "__config_json__"
_VOCAB_KEY = "__vocab_chars__"


def _key(i: int, obj: object, name: str) -> str:
    return f"{i}.{type(obj).__name__}.{name}"


def _model_config(model: Transformer) -> dict:
    block = model.blocks[0]
    return {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "n_heads": block.mha.n_heads,
        "d_ff": block.ffn.linear1.W.shape[1],
        "n_layers": len(model.blocks),
        "max_seq_len": model.max_seq_len,
        "dropout": block.drop1.p,
        "tie_weights": model.tie_weights,
    }


def save_weights(
    model: Transformer,
    path: str | Path,
    vocab: CharVocab | None = None,
    config: dict | None = None,
) -> None:
    """Write trainable params to `path`. If `vocab` or `config` given, embed as meta."""
    arrays: dict[str, Any] = {
        _key(i, obj, name): getattr(obj, name) for i, (obj, name) in enumerate(model.params())
    }
    if config is None:
        config = _model_config(model)
    arrays[_CONFIG_KEY] = np.array(json.dumps(config))
    if vocab is not None:
        arrays[_VOCAB_KEY] = np.array("".join(vocab.chars))
    np.savez(path, **arrays)


def load_weights(model: Transformer, path: str | Path) -> None:
    """Load weights into `model` in place; preserves weight-tying views."""
    data = np.load(path)
    for i, (obj, name) in enumerate(model.params()):
        key = _key(i, obj, name)
        if key not in data:
            raise KeyError(f"checkpoint missing key {key!r}")
        target = getattr(obj, name)
        source = data[key]
        if target.shape != source.shape:
            raise ValueError(
                f"shape mismatch for {key}: model has {target.shape}, checkpoint has {source.shape}"
            )
        target[...] = source


def load_pretrained(path: str | Path) -> tuple[Transformer, CharVocab | None]:
    """Rebuild a Transformer (and CharVocab if embedded) directly from a checkpoint."""
    data = np.load(path)
    if _CONFIG_KEY not in data:
        raise KeyError(
            f"{path} has no embedded config; use load_weights with a manually "
            f"constructed model instead"
        )
    config = json.loads(str(data[_CONFIG_KEY]))
    model = Transformer(**config)
    load_weights(model, path)
    vocab: CharVocab | None = None
    if _VOCAB_KEY in data:
        chars = list(str(data[_VOCAB_KEY]))
        vocab = CharVocab(
            chars=chars,
            char_to_idx={c: i for i, c in enumerate(chars)},
            idx_to_char={i: c for i, c in enumerate(chars)},
        )
    return model, vocab
