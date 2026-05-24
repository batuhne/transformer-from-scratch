"""Save and load model weights as a flat .npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from transformer.model import Transformer


def save_weights(model: Transformer, path: str | Path) -> None:
    """Write every trainable parameter to `path` as a .npz archive."""
    arrays = {
        f"p{i}_{name}": getattr(obj, name)
        for i, (obj, name) in enumerate(model.params())
    }
    np.savez(path, **arrays)


def load_weights(model: Transformer, path: str | Path) -> None:
    """Load weights into `model` in place; preserves weight-tying views."""
    data = np.load(path)
    for i, (obj, name) in enumerate(model.params()):
        getattr(obj, name)[...] = data[f"p{i}_{name}"]
