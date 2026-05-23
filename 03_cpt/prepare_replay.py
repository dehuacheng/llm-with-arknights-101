#!/usr/bin/env python3
"""Stage 03 — prepare the general-Chinese replay stream.

Streams Chinese Wikipedia from Hugging Face, tokenizes with Qwen3's tokenizer,
and packs the same way as 03_cpt/prepare_data.py. Output:

    data/tokenized/qwen3-0.6b-base/replay/{train,val}.bin

A small but well-edited general-Chinese corpus is the standard CPT antidote
to catastrophic forgetting; Wikipedia is the obvious anchor distribution.
Article count is fixed (not "the whole dump") so all four ablation runs see
the *same* replay distribution under a fixed seed — comparability matters
more than size at our scale.

Dataset note: HF's legacy `wikipedia` builder only ships pre-built configs
for the `20220301.*` snapshots; arbitrary dates are not auto-built. The
maintained replacement is `wikimedia/wikipedia` (parquet mirror), which
exposes recent snapshots including `20231101.zh`. We pin the snapshot in
the config arg so all four ablation runs see the same articles.

Articles are shuffled before the train/val cut so neither slice is dominated
by alphabetically-clustered topics. The HF dataset already strips wiki
markup; we tokenize each article's `text` field directly. Output is written
atomically (.tmp → os.replace) with a .json sidecar recording the
fingerprint so a stale cache is detected after a snapshot/seed/count flip.

Usage:
    python3 03_cpt/prepare_replay.py
    python3 03_cpt/prepare_replay.py --n-train 10000 --n-val 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _fingerprint(dataset: str, config: str, n_train: int, n_val: int,
                 seed: int, tok_dir: Path) -> str:
    h = hashlib.sha1()
    h.update(f"{dataset}|{config}|{n_train}|{n_val}|{seed}".encode("utf-8"))
    tok_json = tok_dir / "tokenizer.json"
    if tok_json.exists():
        h.update(tok_json.read_bytes())
    return h.hexdigest()


def _write_atomic(out: Path, subset, tok, bos: int, eos: int) -> int:
    tmp = out.with_suffix(out.suffix + ".tmp")
    total = 0
    with tmp.open("wb") as f:
        for row in subset:
            text = row.get("text") or ""
            if not text:
                continue
            ids = [bos] + tok.encode(text, add_special_tokens=False) + [eos]
            np.asarray(ids, dtype=np.uint32).tofile(f)
            total += len(ids)
    os.replace(tmp, out)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", type=Path,
                    default=ROOT / "data/models/qwen3-0.6b-base")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "data/tokenized/qwen3-0.6b-base/replay")
    ap.add_argument("--dataset", default="wikimedia/wikipedia",
                    help="HF dataset name. Default: wikimedia/wikipedia "
                         "(the maintained parquet mirror; the legacy "
                         "`wikipedia` builder only ships 20220301.* configs).")
    ap.add_argument("--config", default="20231101.zh",
                    help="Dataset config. Default: 20231101.zh — the most "
                         "recent snapshot available on wikimedia/wikipedia "
                         "as of pinning. Pin once, all four runs see the "
                         "same articles.")
    ap.add_argument("--n-train", type=int, default=10_000,
                    help="Articles in the replay/train slice.")
    ap.add_argument("--n-val", type=int, default=500,
                    help="Articles in the replay/val slice.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--force", action="store_true",
                    help="Re-fetch + re-tokenize even when fingerprint matches.")
    args = ap.parse_args()

    fp = _fingerprint(args.dataset, args.config, args.n_train, args.n_val,
                      args.seed, args.base_model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "fingerprint.json"
    if not args.force and meta_path.exists():
        try:
            info = json.loads(meta_path.read_text())
            if info.get("fingerprint") == fp:
                print(f"replay cache hit  ({info.get('train_tokens', '?'):>11,} "
                      f"train + {info.get('val_tokens', '?'):>7,} val tokens)")
                return
        except (json.JSONDecodeError, ValueError):
            pass

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.base_model))
    bos = tok.convert_tokens_to_ids("<|im_start|>")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    if bos is None or eos is None:
        raise SystemExit("Qwen tokenizer does not expose <|im_start|>/<|im_end|>; aborting")
    print(f"tokenizer: {args.base_model}  (<|im_start|>={bos}, <|im_end|>={eos})")

    print(f"loading {args.dataset}:{args.config} ...")
    ds = load_dataset(args.dataset, args.config, split="train")
    print(f"  {len(ds):,} articles available")

    # Shuffle once with a fixed seed; the same N articles go to every run.
    ds = ds.shuffle(seed=args.seed)
    n = args.n_train + args.n_val
    if n > len(ds):
        raise SystemExit(f"need {n:,} articles, dataset has only {len(ds):,}")
    train = ds.select(range(args.n_train))
    val = ds.select(range(args.n_train, args.n_train + args.n_val))

    tokens = {}
    for name, subset in [("train", train), ("val", val)]:
        out = args.out_dir / f"{name}.bin"
        total = _write_atomic(out, subset, tok, bos, eos)
        tokens[name] = total
        print(f"  replay/{name}: {len(subset):>6,} articles -> {total:>12,} tokens -> {out.relative_to(ROOT)}")

    meta_path.write_text(json.dumps({
        "fingerprint": fp, "dataset": args.dataset, "config": args.config,
        "n_train_articles": args.n_train, "n_val_articles": args.n_val,
        "seed": args.seed, "dtype": "uint32",
        "train_tokens": tokens["train"], "val_tokens": tokens["val"],
        "vocab_size": tok.vocab_size,
    }))


if __name__ == "__main__":
    main()
