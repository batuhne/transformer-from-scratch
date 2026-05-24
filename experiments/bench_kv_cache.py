"""Benchmark KV-cache speedup across context lengths."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from transformer.data import load_corpus
from transformer.generate import generate
from transformer.model import Transformer
from transformer.utils import set_seed

DATA = Path(__file__).resolve().parent.parent / "data" / "input.txt"


def time_generate(model, vocab, prompt, max_new, *, use_cache, trials=3) -> float:
    times = []
    for trial in range(trials):
        rng = np.random.default_rng(trial)
        t0 = time.perf_counter()
        generate(model, vocab, prompt, max_new_tokens=max_new,
                 use_cache=use_cache, rng=rng)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def main() -> None:
    _, vocab, _ = load_corpus(DATA)
    prompt = "First "
    print(f"{'T':>5s} {'no cache':>12s} {'cache':>10s} {'speedup':>9s} {'tok/s (cache)':>15s}")
    print("-" * 55)
    for max_seq_len in (32, 64, 128, 256):
        set_seed(42)
        model = Transformer(
            vocab_size=vocab.size,
            d_model=64, n_heads=4, d_ff=256, n_layers=3,
            max_seq_len=max_seq_len,
        )
        max_new = max_seq_len - len(prompt) - 1
        generate(model, vocab, prompt, max_new_tokens=5,
                 use_cache=False, rng=np.random.default_rng(0))
        generate(model, vocab, prompt, max_new_tokens=5,
                 use_cache=True, rng=np.random.default_rng(0))
        no = time_generate(model, vocab, prompt, max_new, use_cache=False)
        yes = time_generate(model, vocab, prompt, max_new, use_cache=True)
        print(f"{max_seq_len:>5d} {no * 1000:>10.1f} ms {yes * 1000:>8.1f} ms "
              f"{no / yes:>8.2f}x {max_new / yes:>14.1f}")


if __name__ == "__main__":
    main()
