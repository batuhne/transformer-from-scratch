"""Character-level vocabulary and batch sampling for the training corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CharVocab:
    """Bidirectional char <-> index map built from a text corpus."""

    chars: list[str]
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> CharVocab:
        chars = sorted(set(text))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        idx_to_char = {i: c for i, c in enumerate(chars)}
        return cls(chars=chars, char_to_idx=char_to_idx, idx_to_char=idx_to_char)

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.char_to_idx[c] for c in s]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx_to_char[i] for i in indices)


def load_corpus(path: str | Path) -> tuple[str, CharVocab, np.ndarray]:
    """Read a text file and return (raw text, vocab, encoded int64 array)."""
    text = Path(path).read_text()
    vocab = CharVocab.from_text(text)
    data = np.array(vocab.encode(text), dtype=np.int64)
    return text, vocab, data


def split_data(
    data: np.ndarray,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous train/val split. Val is the last `val_fraction` of the corpus.
    The split is contiguous (no shuffling) because adjacent characters are
    statistically dependent: shuffling would leak training context into the
    validation set and inflate val accuracy.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_train = int(len(data) * (1.0 - val_fraction))
    return data[:n_train], data[n_train:]


def get_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample `batch_size` contiguous windows of length `seq_len` from `data`.

    Returns:
        x: shape (batch_size, seq_len) input tokens.
        y: shape (batch_size, seq_len) targets shifted by one position.
    """
    if rng is None:
        rng = np.random.default_rng()
    max_start = len(data) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x, y
