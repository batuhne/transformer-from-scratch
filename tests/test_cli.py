"""Smoke test for the `python -m transformer train` entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from transformer.__main__ import main
from transformer.checkpoint import load_pretrained


def test_cli_train_writes_loadable_checkpoint(tmp_path: Path) -> None:
    out = tmp_path / "m.npz"
    code = main(
        [
            "train",
            "--steps",
            "2",
            "--batch",
            "4",
            "--seq-len",
            "16",
            "--data",
            "data/input.txt",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    model, vocab = load_pretrained(out)
    assert vocab is not None
    assert len(model.blocks) == 3


def test_cli_runs_as_module(tmp_path: Path) -> None:
    out = tmp_path / "m.npz"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "transformer",
            "train",
            "--steps",
            "2",
            "--batch",
            "4",
            "--seq-len",
            "16",
            "--data",
            "data/input.txt",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
