#!/usr/bin/env python3
"""Derive qa_val.jsonl from qa_train.jsonl by stratified-by-category sampling.

The Stage 04 agent shipped a single qa_train.jsonl (1919 rows). The SFT
train loop wants a held-out val set for early-stopping. Rather than splitting
inside train_sft.py (would change the train set every run), we derive it
once here and commit both files.

Stratify by `category` so the val set keeps every category visible (refusal
has only 16 rows total — without stratification a random 10% might miss it
entirely, hiding regressions on the named-failure-mode probe).

    python3 04_sft/derive_val_split.py            # writes qa_val.jsonl
    python3 04_sft/derive_val_split.py --ratio 0.1 --min-per-cat 2 --seed 1337
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratio", type=float, default=0.1,
                    help="Approximate fraction of rows in val (default 0.1)")
    ap.add_argument("--min-per-cat", type=int, default=2,
                    help="Lower bound on val rows per category (default 2)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--src", type=Path, default=ROOT / "data/sft/qa_train.jsonl")
    ap.add_argument("--train-out", type=Path,
                    default=ROOT / "data/sft/qa_train.jsonl")
    ap.add_argument("--val-out", type=Path,
                    default=ROOT / "data/sft/qa_val.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.src.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    print(f"loaded {len(rows):,} rows from {args.src}")

    # Group by category. Preserve original order within each group so the
    # remaining train set is deterministic given the seed.
    by_cat: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_cat[r.get("category", "?")].append(i)

    rng = np.random.default_rng(args.seed)
    val_idx: set[int] = set()
    for cat, idxs in by_cat.items():
        n_val = max(args.min_per_cat, round(len(idxs) * args.ratio))
        n_val = min(n_val, len(idxs) - 1)  # never leave a category with 0 train
        chosen = rng.choice(idxs, size=n_val, replace=False)
        val_idx.update(int(i) for i in chosen)
        print(f"  {cat:15} {len(idxs):5} -> {n_val:4} val / {len(idxs)-n_val:5} train")

    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    print(f"split: {len(train_rows):,} train / {len(val_rows):,} val "
          f"({len(val_rows)/len(rows):.1%})")

    if args.dry_run:
        return

    # Write val first, then train — if anything goes wrong on write of val
    # the larger train file is left untouched.
    with args.val_out.open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.val_out}")
    with args.train_out.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.train_out}")


if __name__ == "__main__":
    main()
