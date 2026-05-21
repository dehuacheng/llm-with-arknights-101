#!/usr/bin/env python3
"""Stage 00 · sub-step 2 — apply the split (run by everyone).

Reads the committed 00_data_prep/split_manifest.json and the cleaned corpus
(data/clean/cn/), and writes data/clean/splits.json — the concrete train / val
/ test file lists that stages 01+ consume.

The split is a date RULE, not a frozen file list:

    operator profile  -> train         (always; every entity stays fit-able)
    story in manifest -> train / val   (pre-cutoff; val is a seeded holdout)
    story not in it   -> test          (post-cutoff — including content newer
                                         than the manifest's own snapshot)

Test sets are nested T0 ⊂ T1 ⊂ T2 (see 00_data_prep/README.md):
    T1 = post-cutoff events (whole)
    T2 = T1 + post-cutoff operator records
    T0 = a seeded subset of <章节> blocks inside T1's events

Needs only the cleaned corpus + the manifest — no game data, no git. Run from
the repo root, after clean_corpus.py:

    python3 00_data_prep/apply_split.py
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import corpus  # noqa: E402

VAL_FRACTION = 0.10       # held-out share of pre-cutoff stories, per doc type
VAL_SEED = 20260101       # cutoff date as the seed — reproducible, memorable

MANIFEST = corpus.REPO_ROOT / "00_data_prep" / "split_manifest.json"
_SUBSTORY = re.compile(r"act\d+d\d+\.txt$")


def doc_type(name):
    """Stratification key — doc type only (00_data_prep/README.md)."""
    if name.startswith("main_"):
        return "main"
    if "_set_" in name:
        return "record"
    if name.endswith("side.txt"):
        return "side"
    if name.endswith("mini.txt"):
        return "mini"
    if _SUBSTORY.match(name):
        return "substory"
    return "other"  # 1stact.txt


def stratified_holdout(names, fraction, seed):
    """A seeded, doc-type-stratified subset of `names`."""
    by_type = {}
    for n in sorted(names):
        by_type.setdefault(doc_type(n), []).append(n)
    rng = random.Random(seed)
    held = []
    for _, group in sorted(by_type.items()):
        k = round(fraction * len(group))
        held.extend(rng.sample(group, k))
    return set(held)


def _rel(directory, name):
    """A cleaned file's path as stored in splits.json — relative to data/clean/
    (the form lib.corpus.split_files reverses)."""
    return str((directory / name).relative_to(corpus.CLEAN_DIR))


def main():
    if not MANIFEST.exists():
        sys.exit(f"apply_split: {MANIFEST} not found — run derive_split.py first.")
    stories_dir = corpus.CLEAN_CN_DIR / "stories"
    operators_dir = corpus.CLEAN_CN_DIR / "operators"
    if not stories_dir.is_dir():
        sys.exit(f"apply_split: {stories_dir} not found — run clean_corpus.py first.")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pre_cutoff = set(manifest["pre_cutoff_stories"])

    corpus_stories = sorted(p.name for p in stories_dir.glob("*.txt"))
    corpus_operators = sorted(p.name for p in operators_dir.glob("*.txt"))

    # The date rule applied to *this* snapshot's story files.
    pre = [n for n in corpus_stories if n in pre_cutoff]
    post = [n for n in corpus_stories if n not in pre_cutoff]
    post_events = [n for n in post if "_set_" not in n]
    post_records = [n for n in post if "_set_" in n]

    # Edge case (00_data_prep/README.md): an upstream rename can drop a
    # pre-cutoff file out of the corpus and mis-send it to test.
    missing = sorted(pre_cutoff - set(corpus_stories))
    if missing:
        print(f"WARNING: {len(missing)} manifest pre-cutoff files absent from "
              f"the corpus (rename upstream?): {missing[:5]}...", file=sys.stderr)

    # Snapshot diagnostics — informational; the rule does not depend on a pin.
    snap = manifest["post_cutoff_at_snapshot"]
    pinned = set(snap["events"]) | set(snap["records"])
    if set(post) == pinned:
        print("snapshot: matches the manifest's pinned snapshot (test set is "
              "bit-identical).")
    else:
        print(f"snapshot: differs from the manifest pin — test set has "
              f"{len(post)} files vs {len(pinned)} pinned (newer game content "
              f"correctly falls to test).")

    # val: a seeded holdout of pre-cutoff *stories*. Every operator profile
    # stays in train — same 'entity always fit-able' guarantee as the test sets.
    val = stratified_holdout(pre, VAL_FRACTION, VAL_SEED)
    train = [_rel(operators_dir, n) for n in corpus_operators]
    train += [_rel(stories_dir, n) for n in pre if n not in val]
    val_paths = [_rel(stories_dir, n) for n in sorted(val)]

    test_t1 = [_rel(stories_dir, n) for n in sorted(post_events)]
    test_t2 = [_rel(stories_dir, n) for n in sorted(post_events + post_records)]
    # T0: keep only selections whose event is actually held out in this snapshot.
    test_t0 = {_rel(stories_dir, f): idx
               for f, idx in sorted(manifest["t0"]["selection"].items())
               if f in post_events}

    splits = {
        "schema_version": 1,
        "source_manifest": "00_data_prep/split_manifest.json",
        "meta": {
            "cutoff": manifest["provenance"]["cutoff"],
            "val_fraction": VAL_FRACTION,
            "val_seed": VAL_SEED,
            "t0_unit": manifest["t0"]["unit"],
            "paths": "relative to data/clean/",
            "nesting": "test_t0 chapters ⊂ test_t1 ⊂ test_t2",
        },
        "counts": {
            "train": len(train), "val": len(val_paths),
            "test_t1": len(test_t1), "test_t2": len(test_t2),
            "test_t0_chapters": sum(len(v) for v in test_t0.values()),
        },
        "train": sorted(train),
        "val": val_paths,
        "test_t1": test_t1,
        "test_t2": test_t2,
        "test_t0": test_t0,
    }
    corpus.SPLITS_FILE.write_text(
        json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    c = splits["counts"]
    print(f"wrote {corpus.SPLITS_FILE}")
    print(f"  train   : {c['train']:>4}  ({len(corpus_operators)} operator "
          f"profiles + {c['train'] - len(corpus_operators)} pre-cutoff stories)")
    print(f"  val     : {c['val']:>4}  (seeded {VAL_FRACTION:.0%} holdout, "
          f"doc-type stratified)")
    print(f"  test_t1 : {c['test_t1']:>4}  post-cutoff events")
    print(f"  test_t2 : {c['test_t2']:>4}  events + post-cutoff records")
    print(f"  test_t0 : {c['test_t0_chapters']:>4}  <章节> blocks inside T1's events")


if __name__ == "__main__":
    main()
