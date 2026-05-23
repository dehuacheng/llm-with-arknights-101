#!/usr/bin/env python3
"""Sanity-check the shared evaluation set.

Runs on every change to questions.yaml. Catches schema drift before the
hand-grading scripts encounter it. Not a grader — just a structural check.

Usage:
    python3 eval/validate.py
    python3 eval/validate.py eval/questions.yaml   # explicit path
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

REQUIRED = {
    "id", "category", "answer_type", "difficulty",
    "question_zh", "question_en",
    "gold_zh", "gold_en",
    "key_facts", "traps", "tags", "source",
}
OPTIONAL = {"notes"}

CATEGORIES = {"character", "faction", "world", "event", "relationship", "refusal"}
ANSWER_TYPES = {"factoid", "open_ended", "refusal"}
DIFFICULTIES = {"easy", "medium", "hard"}
ID_RE = re.compile(r"^Q\d{3}$")


def fail(msg: str) -> None:
    print(f"FAIL  {msg}", file=sys.stderr)


def main(path: Path) -> int:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        fail(f"{path}: top-level must be a YAML list")
        return 1

    errors = 0
    seen_ids: set[str] = set()
    n_by_cat: dict[str, int] = {c: 0 for c in CATEGORIES}

    for i, q in enumerate(raw):
        # Guard the dict check before calling .get on it — a stray scalar in
        # the YAML list would otherwise AttributeError on the loc string itself.
        if not isinstance(q, dict):
            fail(f"item #{i}: not a mapping ({type(q).__name__})")
            errors += 1
            continue
        loc = f"item #{i} (id={q.get('id', '<missing>')})"

        missing = REQUIRED - set(q.keys())
        if missing:
            fail(f"{loc}: missing fields {sorted(missing)}")
            errors += 1

        extra = set(q.keys()) - REQUIRED - OPTIONAL
        if extra:
            fail(f"{loc}: unknown fields {sorted(extra)}")
            errors += 1

        qid = q.get("id")
        if not isinstance(qid, str) or not ID_RE.match(qid):
            fail(f"{loc}: id must match Q\\d\\d\\d, got {qid!r}")
            errors += 1
        elif qid in seen_ids:
            fail(f"{loc}: duplicate id {qid!r}")
            errors += 1
        else:
            seen_ids.add(qid)

        cat = q.get("category")
        if cat not in CATEGORIES:
            fail(f"{loc}: category={cat!r} not in {sorted(CATEGORIES)}")
            errors += 1
        else:
            n_by_cat[cat] += 1

        at = q.get("answer_type")
        if at not in ANSWER_TYPES:
            fail(f"{loc}: answer_type={at!r} not in {sorted(ANSWER_TYPES)}")
            errors += 1

        if q.get("difficulty") not in DIFFICULTIES:
            fail(f"{loc}: difficulty={q.get('difficulty')!r} not in {sorted(DIFFICULTIES)}")
            errors += 1

        # traps must be a list everywhere (possibly empty for non-refusal,
        # non-empty for refusal). `traps: null` would silently pass the
        # not-q.get("traps") and `if q.get("traps"):` checks below, so
        # require the list shape up front.
        traps = q.get("traps")
        if not isinstance(traps, list):
            fail(f"{loc}: traps must be a list (got {type(traps).__name__})")
            errors += 1
            traps = []  # downstream checks rely on list shape; substitute

        # refusal items: gold_* empty, key_facts empty, traps non-empty
        if at == "refusal":
            if q.get("gold_zh") or q.get("gold_en"):
                fail(f"{loc}: refusal items must have empty gold_zh / gold_en")
                errors += 1
            if q.get("key_facts"):
                fail(f"{loc}: refusal items must have empty key_facts")
                errors += 1
            if not isinstance(traps, list) or not traps:
                fail(f"{loc}: refusal items must list at least one trap")
                errors += 1
        else:
            if not q.get("gold_zh") or not q.get("gold_en"):
                fail(f"{loc}: non-refusal items must have non-empty gold_zh / gold_en")
                errors += 1
            if not q.get("key_facts"):
                fail(f"{loc}: non-refusal items must list at least one key_fact")
                errors += 1
            # traps on factoid/open_ended are allowed (common-misconception capture).

        # category / answer_type consistency
        if cat == "refusal" and at != "refusal":
            fail(f"{loc}: category=refusal requires answer_type=refusal")
            errors += 1
        if at == "refusal" and cat != "refusal":
            fail(f"{loc}: answer_type=refusal requires category=refusal")
            errors += 1

        # tags must be a list of strings
        tags = q.get("tags")
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            fail(f"{loc}: tags must be a list of strings")
            errors += 1

    print(f"\n{len(raw)} items checked across categories:")
    for c in sorted(CATEGORIES):
        print(f"  {c:>14}: {n_by_cat[c]:3d}")

    if errors:
        print(f"\n{errors} schema error(s) — see above.", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "questions.yaml"
    sys.exit(main(path))
