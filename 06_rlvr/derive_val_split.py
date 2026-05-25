#!/usr/bin/env python3
"""Derive data/rl/prompts_val.jsonl from data/rl/prompts_train.jsonl.

Stage 06 (RLVR / GRPO) wants a held-out RL val set for periodic val_reward
measurement. The Stage 05 agent ships prompts_train.jsonl but no val split.
This script derives the val side without modifying the train file (unlike
05_dpo/derive_val_split.py, which splits the source in place).

Design choices:

- **Stratify on `category`** (RL prompts are factoid-only by AGENT_BRIEF §6,
  so `fault_type` doesn't apply — `category` is the natural axis: character /
  faction / world / event / relationship).

- **Deterministic by sha256(seed|id)**: ranked within each category, take the
  top N hashes as val. Adding new rows to prompts_train.jsonl between runs
  *may* shift which rows are val (a new row with a low hash can bump an
  existing one) — but that's the right behaviour here: we want new prompts
  participating in val too. Stability across re-runs is preserved as long
  as the train file is unchanged.

- **Train file is never touched.** The training script (train_rlvr.py)
  loads prompts_train.jsonl AND prompts_val.jsonl, then removes val IDs
  from the train iterator at load time. This keeps the source-of-truth
  single (`data/rl/prompts_train.jsonl` is what the agent edits).

Usage:

    python3 06_rlvr/derive_val_split.py                  # default: 10%, min 5 per category
    python3 06_rlvr/derive_val_split.py --ratio 0.15     # custom ratio
    python3 06_rlvr/derive_val_split.py --dry-run        # inspect, don't write
    python3 06_rlvr/derive_val_split.py --force          # overwrite existing val file
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/rl/prompts_train.jsonl"
VAL_PATH = ROOT / "data/rl/prompts_val.jsonl"


def hash_rank(item_id: str, seed: int) -> float:
    """A [0, 1) value deterministic in (seed, id). SHA-256 over a salted key
    so two different seeds give independent rankings; stable across processes
    and platforms (unlike Python's randomised hash())."""
    digest = hashlib.sha256(f"{seed}|{item_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def derive(ratio: float, min_per_cat: int, seed: int,
           dry_run: bool, force: bool) -> None:
    if not TRAIN_PATH.exists():
        raise SystemExit(f"missing: {TRAIN_PATH.relative_to(ROOT)} "
                         f"(run the agent's data generation first)")
    if VAL_PATH.exists() and not force and not dry_run:
        raise SystemExit(
            f"{VAL_PATH.relative_to(ROOT)} already exists. "
            f"Pass --force to overwrite, --dry-run to inspect.")

    rows = [json.loads(l) for l in TRAIN_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    print(f"loaded {len(rows):,} rows from {TRAIN_PATH.relative_to(ROOT)}")

    # Group by category. Each group is ranked by hash(id) and the lowest-hash
    # round(N*ratio) rows go to val. Ties broken by id (lexicographic) for
    # determinism if two IDs hash to the same prefix (vanishingly unlikely).
    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "?")].append(r)

    val_ids: set[str] = set()
    print(f"{'category':15} {'total':>6} {'val':>6} {'train':>6}")
    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        n_val = max(min_per_cat, round(len(cat_rows) * ratio))
        n_val = min(n_val, len(cat_rows) - 1)  # never leave a category with 0 train
        ranked = sorted(cat_rows, key=lambda r: (hash_rank(r["id"], seed), r["id"]))
        for r in ranked[:n_val]:
            val_ids.add(r["id"])
        print(f"{cat:15} {len(cat_rows):>6} {n_val:>6} {len(cat_rows)-n_val:>6}")

    val_rows = [r for r in rows if r["id"] in val_ids]
    train_kept = len(rows) - len(val_rows)
    print(f"{'TOTAL':15} {len(rows):>6} {len(val_rows):>6} {train_kept:>6} "
          f"({len(val_rows)/len(rows):.1%} val)")

    if dry_run:
        return

    # Atomic write — full-write to .tmp, then rename.
    tmp = VAL_PATH.with_name(VAL_PATH.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, VAL_PATH)
    print(f"wrote {VAL_PATH.relative_to(ROOT)} ({len(val_rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratio", type=float, default=0.1,
                    help="Approximate fraction of rows in val (default 0.1)")
    ap.add_argument("--min-per-cat", type=int, default=5,
                    help="Lower bound on val rows per category (default 5)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwrite of an existing prompts_val.jsonl")
    args = ap.parse_args()
    derive(args.ratio, args.min_per_cat, args.seed, args.dry_run, args.force)


if __name__ == "__main__":
    main()
