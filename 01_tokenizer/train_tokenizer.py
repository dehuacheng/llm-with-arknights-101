#!/usr/bin/env python3
"""Stage 01 — train a from-scratch byte-level BPE tokenizer.

Trains lib.bpe.ByteBPE on the stage-00 *train* split and saves it under
data/tokenizers/<name>/ (git-ignored — the merge table holds corpus byte
fragments). Configs are YAML, one per vocab size; clone, never edit in place.

    python3 01_tokenizer/train_tokenizer.py --config 01_tokenizer/configs/vocab_16k.yaml
    python3 01_tokenizer/train_tokenizer.py --config <any> --smoke-test
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import bpe, corpus  # noqa: E402

# --smoke-test: a tiny vocab on a handful of files for a <1-minute end-to-end run.
SMOKE_NAME, SMOKE_VOCAB, SMOKE_FILES = "smoke", 512, 25


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML experiment config")
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny vocab on a few files for a quick end-to-end run")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name, vocab_size = cfg["name"], cfg["vocab_size"]

    files = corpus.split_files("train")
    if args.smoke_test:
        name, vocab_size = SMOKE_NAME, SMOKE_VOCAB
        files = files[:SMOKE_FILES]

    specials = list(bpe.CONTROL_TOKENS) + corpus.structural_tag_tokens()
    texts = [f.read_text(encoding="utf-8") for f in files]
    total_chars = sum(len(t) for t in texts)
    print(f"train tokenizer '{name}'")
    print(f"  vocab size : {vocab_size:,}  ({len(specials)} special tokens reserved)")
    print(f"  corpus     : {len(files)} train files, {total_chars:,} chars")

    t0 = time.time()
    tok = bpe.ByteBPE.train(texts, vocab_size, specials, verbose=True)
    elapsed = time.time() - t0

    out = corpus.DATA_DIR / "tokenizers" / name / "tokenizer.json"
    tok.save(out)
    print(f"  saved      : {out}  ({tok.vocab_size:,} tokens, {elapsed:.1f}s)")

    # The merge sequence is a teaching artifact — BPE builds the vocabulary
    # bottom-up, most-frequent pair first. See 01_tokenizer/README.md.
    rows = list(tok.merge_log())
    log_path = out.parent / "merge_log.txt"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# '{name}': {len(rows)} merges, in the order BPE learned them.\n")
        fh.write("# rank  count  token = left + right   (Python repr; a b'..'\n")
        fh.write("# literal is a partial byte run, shown until its bytes form a char)\n")
        for r in rows:
            fh.write(f"{r['rank']:>8} {r['count']:>11}  "
                     f"{r['token']!r} = {r['left']!r} + {r['right']!r}\n")
    print(f"  merge log  : {log_path}")
    print("  first merges (most frequent pairs first):")
    for r in rows[:15]:
        print(f"    #{r['rank']:<4} count={r['count']:>9}  "
              f"{r['token']!r} = {r['left']!r} + {r['right']!r}")

    print("  sanity checks:")
    samples = [t[:3000] for t in texts[:5]]
    failed = False
    for cname, passed, detail in tok.sanity_check(sample_texts=samples):
        print(f"    [{'PASS' if passed else 'FAIL'}] {cname:<28} {detail}")
        failed = failed or not passed
    if failed:
        sys.exit("ERROR: sanity check failed")


if __name__ == "__main__":
    main()
