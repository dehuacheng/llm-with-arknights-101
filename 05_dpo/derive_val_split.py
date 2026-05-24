#!/usr/bin/env python3
"""Derive *_val.jsonl from *_train.jsonl by stratified-by-fault_type sampling.

The Stage 05 agent shipped bulk_train.jsonl (1790) and curated_train.jsonl (890).
The DPO train loop needs held-out val sets for early-stopping and pair-accuracy
reporting. Mirrors 04_sft/derive_val_split.py:

- Stratify on `fault_type` (not `category`) because the 2×2 ablation cares
  about whether the model learns to reject *plausible-but-wrong* answers; the
  fault axis is the natural stratification axis. The smallest fault_type
  (`wrong_date`: 11/13 rows total) needs preservation in val — random 10%
  could miss it entirely.
- Seed 1337 (same as Stage 04 val split).
- Writes val first then train so an interrupted run leaves the larger train
  file untouched.

    # default: derive both files
    python3 05_dpo/derive_val_split.py

    # custom ratio / min-per-cat / seed
    python3 05_dpo/derive_val_split.py --ratio 0.1 --min-per-cat 2 --seed 1337

    # one flavour at a time
    python3 05_dpo/derive_val_split.py --flavour curated
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLAVOURS = ("bulk", "curated")


def derive(flavour: str, ratio: float, min_per_cat: int, seed: int,
           dry_run: bool, force: bool, strat_field: str) -> None:
    src = ROOT / f"data/dpo/{flavour}_train.jsonl"
    train_out = ROOT / f"data/dpo/{flavour}_train.jsonl"
    val_out = ROOT / f"data/dpo/{flavour}_val.jsonl"

    # Idempotency guard: train_out == src, so a second run would re-split the
    # already-trimmed train file (silent dataset shrinkage + val/train mix).
    # Bail unless --force; the val file's presence is the natural sentinel.
    if val_out.exists() and not force and not dry_run:
        raise SystemExit(
            f"[{flavour}] {val_out.relative_to(ROOT)} already exists. "
            f"Re-running would shrink train_out (it shares the path of src). "
            f"Pass --force to overwrite, --dry-run to inspect.")

    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    print(f"\n[{flavour}] loaded {len(rows):,} rows from {src.name}")

    # Group by stratification field. Preserve original order within each group
    # so the remaining train set is deterministic given the seed.
    by_cat: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_cat[r.get(strat_field, "?")].append(i)

    rng = np.random.default_rng(seed)
    val_idx: set[int] = set()
    for cat, idxs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        n_val = max(min_per_cat, round(len(idxs) * ratio))
        n_val = min(n_val, len(idxs) - 1)  # never leave a category with 0 train
        chosen = rng.choice(idxs, size=n_val, replace=False)
        val_idx.update(int(i) for i in chosen)
        print(f"  {cat:15} {len(idxs):5} -> {n_val:4} val / {len(idxs)-n_val:5} train")

    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    print(f"  split: {len(train_rows):,} train / {len(val_rows):,} val "
          f"({len(val_rows)/len(rows):.1%})")

    if dry_run:
        return

    # Atomic-ish write: full-write both to .tmp siblings, THEN rename.
    # If anything fails mid-stream, the original src is untouched.
    val_tmp = val_out.with_name(val_out.name + ".tmp")
    train_tmp = train_out.with_name(train_out.name + ".tmp")
    with val_tmp.open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with train_tmp.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Both .tmp files exist; now atomically swap (rename is atomic on POSIX
    # within the same filesystem). Val first, then train — minimises the
    # window where src is gone.
    os.replace(val_tmp, val_out)
    os.replace(train_tmp, train_out)
    print(f"  wrote {val_out.relative_to(ROOT)}")
    print(f"  wrote {train_out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratio", type=float, default=0.1,
                    help="Approximate fraction of rows in val (default 0.1)")
    ap.add_argument("--min-per-cat", type=int, default=2,
                    help="Lower bound on val rows per fault_type (default 2)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--flavour", choices=FLAVOURS, default=None,
                    help="Only derive this flavour; default does both.")
    ap.add_argument("--strat-field", default="fault_type",
                    help="JSONL key to stratify on (default fault_type)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Allow re-split even if <flavour>_val.jsonl exists.")
    args = ap.parse_args()

    flavours = (args.flavour,) if args.flavour else FLAVOURS
    for fl in flavours:
        derive(fl, args.ratio, args.min_per_cat, args.seed,
               args.dry_run, args.force, args.strat_field)


if __name__ == "__main__":
    main()
