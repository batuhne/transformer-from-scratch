"""Save and load model weights as a flat .npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from transformer.model import Transformer


def _key(i: int, obj: object, name: str) -> str:
    return f"{i}.{type(obj).__name__}.{name}"


def save_weights(model: Transformer, path: str | Path) -> None:
    """Write every trainable parameter to `path` as a .npz archive."""
    arrays = {
        _key(i, obj, name): getattr(obj, name)
        for i, (obj, name) in enumerate(model.params())
    }
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
                f"shape mismatch for {key}: model has {target.shape}, "
                f"checkpoint has {source.shape}"
            )
        target[...] = source
