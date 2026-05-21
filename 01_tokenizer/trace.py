#!/usr/bin/env python3
"""Stage 01 — trace the merge path of a term.

For each term, print every merge that builds its encoding, in the order BPE
learned them: raw bytes -> characters -> the final token. Made for following
Arknights proper nouns (专有名词) down to their bytes.

    python3 01_tokenizer/trace.py --name vocab_32k 罗德岛 凯尔希 PRTS
    python3 01_tokenizer/trace.py --name vocab_32k            # a default sample
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import bpe, corpus  # noqa: E402

# Well-known Arknights proper nouns, traced when no terms are given.
DEFAULT_TERMS = ["罗德岛", "凯尔希", "阿米娅", "整合运动", "源石技艺",
                 "感染者", "特蕾西娅", "切尔诺伯格", "罗德岛制药", "PRTS"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True,
                    help="tokenizer name under data/tokenizers/")
    ap.add_argument("terms", nargs="*",
                    help="terms to trace (default: a sample of Arknights 专有名词)")
    args = ap.parse_args()

    path = corpus.DATA_DIR / "tokenizers" / args.name / "tokenizer.json"
    tok = bpe.ByteBPE.load(path)
    terms = args.terms or DEFAULT_TERMS
    print(f"tokenizer '{args.name}'  ({tok.vocab_size:,} tokens)\n")

    for term in terms:
        ids, merge_path = tok.trace(term)
        pieces = " | ".join(tok.decode([i]) for i in ids)
        print(f"{term}  ->  {len(ids)} token(s):  {pieces}")
        for r in merge_path:
            count = f"{r['count']:,}" if r["count"] is not None else "-"
            print(f"  #{r['rank']:<6} count {count:>10}   "
                  f"{r['token']!r} = {r['left']!r} + {r['right']!r}")
        if not merge_path:
            print("  (no merges — encodes straight to byte/character tokens)")
        print()


if __name__ == "__main__":
    main()
