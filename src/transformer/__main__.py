"""CLI entrypoint: `python -m transformer train ...`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformer.checkpoint import save_weights
from transformer.data import load_corpus, split_data
from transformer.model import Transformer
from transformer.train import TrainConfig, train
from transformer.utils import set_seed


def cli_train(args: argparse.Namespace) -> int:
    rng = set_seed(args.seed)
    _, vocab, data = load_corpus(args.data)
    train_data, val_data = split_data(data, val_fraction=args.val_fraction)
    model = Transformer(
        vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        rng=rng,
    )
    config = TrainConfig(
        n_steps=args.steps,
        batch_size=args.batch,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    train(model, train_data, config, val_data=val_data, rng=rng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_weights(model, out, vocab=vocab)
    print(f"saved checkpoint to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transformer", description="NumPy char-level transformer")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a model and save a checkpoint")
    t.add_argument("--data", default="data/input.txt", help="path to the text corpus")
    t.add_argument("--out", default="checkpoints/model.npz", help="checkpoint output path")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--seq-len", type=int, default=32, dest="seq_len")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--tie-weights", action=argparse.BooleanOptionalAction,
                   default=True, dest="tie_weights")
    t.add_argument("--d-model", type=int, default=64, dest="d_model")
    t.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    t.add_argument("--d-ff", type=int, default=256, dest="d_ff")
    t.add_argument("--n-layers", type=int, default=3, dest="n_layers")
    t.add_argument("--val-fraction", type=float, default=0.1, dest="val_fraction")
    t.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    t.add_argument("--warmup-steps", type=int, default=100, dest="warmup_steps")
    t.add_argument("--min-lr-ratio", type=float, default=0.1, dest="min_lr_ratio")
    t.set_defaults(func=cli_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
