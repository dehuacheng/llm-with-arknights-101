#!/usr/bin/env python3
"""Stage 01 — fertility report for a trained tokenizer.

Fertility = tokens per CN character: lower means the vocabulary packs the
corpus into shorter sequences. It is reported on every flat split so the
train-vs-held-out gap — how much the vocabulary is fit to the train set — is
visible. Run after train_tokenizer.py:

    python3 01_tokenizer/fertility.py --name vocab_16k
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import bpe, corpus  # noqa: E402

HAN = re.compile(r"[一-鿿]")
SPLITS = ("train", "val", "test_t1", "test_t2")


def measure(tok, files):
    n_tok = n_han = n_chr = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        n_tok += len(tok.encode(text))
        n_chr += len(text)
        n_han += len(HAN.findall(text))
    return n_tok, n_han, n_chr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True,
                    help="tokenizer name under data/tokenizers/")
    args = ap.parse_args()

    path = corpus.DATA_DIR / "tokenizers" / args.name / "tokenizer.json"
    tok = bpe.ByteBPE.load(path)
    print(f"tokenizer '{args.name}'  ({tok.vocab_size:,} tokens, "
          f"{len(tok.merges):,} merges)\n")

    header = (f"{'split':<9} {'files':>6} {'tokens':>12} {'Han chars':>12} "
              f"{'tok/Han':>9} {'tok/char':>9}")
    print(header)
    print("-" * len(header))
    for split in SPLITS:
        files = corpus.split_files(split)
        n_tok, n_han, n_chr = measure(tok, files)
        tok_per_han = n_tok / n_han if n_han else 0.0
        tok_per_char = n_tok / n_chr if n_chr else 0.0
        print(f"{split:<9} {len(files):>6} {n_tok:>12,} "
              f"{n_han:>12,} {tok_per_han:>9.3f} {tok_per_char:>9.3f}")

    # example segmentations of public faction/operator names — IP-safe to show
    print("\nexample segmentations (a partial-byte token shows as the "
          "replacement char):")
    for name in ("罗德岛", "凯尔希", "整合运动", "博士"):
        pieces = [tok.decode([i]) for i in tok.encode(name)]
        print(f"  {name} -> {' | '.join(pieces)}  ({len(pieces)} tokens)")


if __name__ == "__main__":
    main()
