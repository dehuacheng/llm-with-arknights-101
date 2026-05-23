#!/usr/bin/env python3
"""Stage 03 — re-tokenize the Stage-00 splits with Qwen3-0.6B-Base's tokenizer.

Stage 02 trained from-scratch ByteBPE tokenizers; Qwen3 has its own
151,936-token tokenizer baked into the checkpoint, so the Arknights cn/ corpus
has to be re-tokenized before it can feed continued pretraining.

Output layout (mirrors Stage 02 — see 02_pretrain/README.md §3):

    data/tokenized/qwen3-0.6b-base/arknights/{train,val,test_t0,test_t1,test_t2}.bin

Each .bin is a flat uint32 stream of token ids (151,936 > 65K, so uint16 is
not wide enough). Documents are bracketed by Qwen's <|im_start|> / <|im_end|>
structural tokens — the same packing convention as Stage 02 §2. A sidecar
.json records the source fingerprint so a stale cache (after a tokenizer
swap or split re-derivation) is detected instead of silently reused.

The general-Chinese replay stream (data/tokenized/qwen3-0.6b-base/replay/*)
is prepared by 03_cpt/prepare_replay.py.

Usage:
    python3 03_cpt/prepare_data.py
    python3 03_cpt/prepare_data.py --base-model data/models/qwen3-0.6b-base
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

# splits.json keys are lowercase (`train`, `val`, `test_t0`, `test_t1`,
# `test_t2`); test_t0 is a *dict* {relpath: [chapter_idx, ...]} carving
# specific <章节> blocks out of files that mostly live in train/val.
LIST_SPLITS = ["train", "val", "test_t1", "test_t2"]
DICT_SPLIT = "test_t0"


def load_tokenizer(base_model: Path):
    """Defer the transformers import so this script can be lint-loaded without
    the stage-03 deps installed."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=False)


def _fingerprint(tok_dir: Path, files: list[Path]) -> str:
    """A digest of the tokenizer config + every input file's path and size.
    Changes if the tokenizer is swapped or a split's file set is edited, so a
    stale .bin is detected instead of silently reused. Matches the cache shape
    used by 02_pretrain/train.py."""
    h = hashlib.sha1()
    # Hash the tokenizer file itself (small JSON) so an unrelated weight swap
    # in the same dir doesn't flip the fingerprint.
    tok_json = tok_dir / "tokenizer.json"
    if tok_json.exists():
        h.update(tok_json.read_bytes())
    for f in sorted(files):
        h.update(f"{f}:{f.stat().st_size}".encode("utf-8"))
    return h.hexdigest()


def _write_stream(out: Path, items: list[tuple[Path, list[int] | None]],
                  tok, bos: int, eos: int, fingerprint: str) -> int:
    """Tokenize each (path, chapter_idx) item, pack with BOS/EOS, write
    atomically. Returns total token count.

    `chapter_idx` is None for full-file splits and a list of <章节> block
    ordinals for test_t0 (one .bin row per selected chapter)."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    total = 0
    with tmp.open("wb") as f:
        for path, chapters in items:
            text = path.read_text(encoding="utf-8")
            if chapters is None:
                docs = [text]
            else:
                # Stage 00 splits a file into <章节>...</章节> blocks; test_t0
                # selects a subset by 0-based ordinal.
                blocks = _split_chapters(text)
                docs = [blocks[i] for i in chapters if i < len(blocks)]
            for doc in docs:
                ids = [bos] + tok.encode(doc, add_special_tokens=False) + [eos]
                np.asarray(ids, dtype=np.uint32).tofile(f)
                total += len(ids)
    os.replace(tmp, out)  # atomic — no partial .bin on crash mid-write
    out.with_suffix(".json").write_text(json.dumps({
        "fingerprint": fingerprint, "dtype": "uint32",
        "n_tokens": total, "vocab_size": tok.vocab_size,
    }))
    return total


def _split_chapters(text: str) -> list[str]:
    """Cut a cleaned-corpus file along <章节>...</章节> boundaries — one doc
    per chapter block, in source order. Bytes outside chapter blocks (intros,
    file headers) are dropped; the same convention Stage 00 used to derive
    test_t0."""
    out, depth, buf = [], 0, []
    for line in text.splitlines(keepends=True):
        if "<章节>" in line:
            depth += 1
            buf = [line]
        elif "</章节>" in line:
            buf.append(line)
            out.append("".join(buf))
            depth, buf = 0, []
        elif depth:
            buf.append(line)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", type=Path,
                    default=ROOT / "data/models/qwen3-0.6b-base",
                    help="Qwen3-0.6B-Base checkpoint dir (holds tokenizer.json).")
    ap.add_argument("--splits-json", type=Path,
                    default=ROOT / "data/clean/splits.json",
                    help="Stage-00 output: {split: [file_path, ...]}.")
    ap.add_argument("--clean-root", type=Path,
                    default=ROOT / "data/clean",
                    help="Directory the split file paths are relative to.")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "data/tokenized/qwen3-0.6b-base/arknights",
                    help="Where the .bin streams land.")
    ap.add_argument("--force", action="store_true",
                    help="Re-tokenize even when fingerprint matches cache.")
    args = ap.parse_args()

    print(f"loading tokenizer: {args.base_model}")
    tok = load_tokenizer(args.base_model)

    # Qwen3 uses <|im_start|> / <|im_end|> as structural sentinels in its chat
    # template; we reuse them as document boundaries during CPT. Verify they
    # exist in the vocabulary before relying on them. (Qwen3 has unk_token=None,
    # so a missing token resolves to None, not unk_id — guard accordingly.)
    bos = tok.convert_tokens_to_ids("<|im_start|>")
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    if bos is None or eos is None or bos == tok.unk_token_id == eos:
        raise SystemExit("Qwen tokenizer does not expose <|im_start|>/<|im_end|>; aborting")
    print(f"  document boundary: <|im_start|>={bos}  <|im_end|>={eos}")

    splits = json.loads(args.splits_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build (split_name, items, file_list_for_fingerprint) tuples.
    work: list[tuple[str, list[tuple[Path, list[int] | None]], list[Path]]] = []
    for split in LIST_SPLITS:
        rels = splits.get(split) or []
        items = [(args.clean_root / r, None) for r in sorted(rels)]
        work.append((split, items, [p for p, _ in items]))

    t0 = splits.get(DICT_SPLIT) or {}
    if t0:
        items = [(args.clean_root / r, list(chs)) for r, chs in sorted(t0.items())]
        work.append((DICT_SPLIT, items, [p for p, _ in items]))

    for split, items, files in work:
        if not items:
            print(f"  skip {split}: no files in splits.json")
            continue
        out = args.out_dir / f"{split}.bin"
        fp = _fingerprint(args.base_model, files)
        meta = out.with_suffix(".json")
        if not args.force and out.exists() and meta.exists():
            try:
                info = json.loads(meta.read_text())
                if info.get("fingerprint") == fp:
                    print(f"  {split:>10}: cache hit  ({info.get('n_tokens', '?'):>11,} tokens)")
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
        total = _write_stream(out, items, tok, bos, eos, fp)
        print(f"  {split:>10}: {len(items):4d} docs  -> {total:>11,} tokens  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
