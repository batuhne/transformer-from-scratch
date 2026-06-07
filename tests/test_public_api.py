"""Smoke test for the curated top-level package surface."""

from __future__ import annotations

import transformer


def test_public_api_lists_match_attrs() -> None:
    for name in transformer.__all__:
        assert hasattr(transformer, name), f"transformer.{name} declared but missing"


def test_public_api_imports() -> None:
    from transformer import (
        CharVocab,
        TrainConfig,
        Transformer,
        generate,
        load_pretrained,
        perplexity,
        set_seed,
        top_k_filter,
        top_p_filter,
        train,
    )

    # Sanity: each is a callable or class, none are None.
    for obj in (CharVocab, Transformer, TrainConfig, generate, load_pretrained,
                perplexity, set_seed, top_k_filter, top_p_filter, train):
        assert obj is not None
