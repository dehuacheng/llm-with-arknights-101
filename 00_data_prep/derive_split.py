#!/usr/bin/env python3
"""Stage 00 · sub-step 2 — derive the train/test split (ONE-TIME, provenance).

Writes 00_data_prep/split_manifest.json: the committed, immutable *pre-cutoff
manifest*. Repo users do NOT run this — they consume the committed manifest via
apply_split.py. It stays committed only so the derivation is auditable.

The split is a date rule — train = pre-cutoff content, test = post-cutoff —
applied to whatever ArknightsGameData snapshot a user builds from. Dating, per
00_data_prep/README.md, needs two sources:

  events  — story_review_table.json `startTime` (authoritative in-game date)
  records — the arknights_lore_wiki git delta since the cutoff (a proxy: the
            operator records carry no in-game date at all)

Needs the sibling repos ../ArknightsGameData and ../arknights_lore_wiki (the
latter with git history). Run from the repo root:

    python3 00_data_prep/derive_split.py
"""
import json
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import corpus  # noqa: E402

BEIJING = timezone(timedelta(hours=8))
CUTOFF = datetime(2026, 1, 1, tzinfo=BEIJING)
CUTOFF_TS = int(CUTOFF.timestamp())

# Last arknights_lore_wiki commit before the cutoff (2025-12-06). The next
# wiki commits (2026-01-10) add the first post-cutoff content, so this is a
# clean boundary — see 00_data_prep/README.md.
WIKI_BOUNDARY_COMMIT = "3cacb8f8"

T0_SEED = 20260101            # cutoff date as the seed — reproducible, memorable
T0_FRACTION = 1 / 3           # share of each post-cutoff event's <章节> blocks

PROJECTS = corpus.REPO_ROOT.parent
GAME_DATA = PROJECTS / "ArknightsGameData"
WIKI = PROJECTS / "arknights_lore_wiki"
REVIEW_TABLE = GAME_DATA / "zh_CN" / "gamedata" / "excel" / "story_review_table.json"
MANIFEST = corpus.REPO_ROOT / "00_data_prep" / "split_manifest.json"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def wiki_added_files(head):
    """Story files added to arknights_lore_wiki between the boundary and `head`."""
    out = _git(WIKI, "diff", "--diff-filter=A", "--name-only",
               WIKI_BOUNDARY_COMMIT, head, "--", "data/stories")
    return sorted(Path(p).name for p in out.splitlines() if p.strip())


def chapter_count(story_file):
    """Number of <章节> blocks in a raw story file (cleaning preserves these)."""
    text = (corpus.RAW_CN_DIR / "stories" / story_file).read_text(encoding="utf-8")
    return sum(1 for line in text.splitlines() if line.strip() == "<章节>")


def main():
    for name, path in (("ArknightsGameData", GAME_DATA), ("arknights_lore_wiki", WIKI)):
        if not path.is_dir():
            sys.exit(f"derive_split: missing sibling repo '{name}' at {path}")

    table = json.loads(REVIEW_TABLE.read_text(encoding="utf-8"))
    # Two populations: calendar events vs operator records (entryType NONE,
    # which unlock by operator trust and carry no date at all).
    events = {k: e for k, e in table.items() if e["entryType"] != "NONE"}
    records = {k: e for k, e in table.items() if e["entryType"] == "NONE"}

    # --- events: dated from the authoritative game data ----------------------
    # ACTIVITY / MINI events carry a real startTime; MAINLINE events carry
    # startTime -1 (the main story is undated in this table). Undated events
    # are placed pre-cutoff and reported — at this snapshot they are the long-
    # released main-story chapters, but a re-derivation must re-verify that.
    dateable = {k: e for k, e in events.items() if e["startTime"] > 0}
    undateable = sorted(k for k, e in events.items() if e["startTime"] <= 0)
    post_event_ids = sorted(k for k, e in dateable.items()
                            if e["startTime"] >= CUTOFF_TS)
    post_event_files = [f"{k}.txt" for k in post_event_ids]

    # --- records: dated from the wiki git delta (proxy oracle) ---------------
    wiki_head = _git(WIKI, "rev-parse", "HEAD")
    added = wiki_added_files(wiki_head)
    post_record_files = sorted(f for f in added if "_set_" in f)
    wiki_added_events = sorted(f for f in added if "_set_" not in f)

    unknown = [f for f in post_record_files if Path(f).stem not in records]
    if unknown:
        print(f"WARNING: wiki delta names records absent from this game-data "
              f"snapshot (skew): {unknown}", file=sys.stderr)

    # --- cross-check: the wiki delta should re-surface the game-dated events --
    # The wiki proxy can also list pre-cutoff events it happened to backfill
    # late (e.g. main_17); those must NOT be treated as post-cutoff. This is
    # exactly why events are dated from the game data, not the wiki proxy.
    wiki_event_ids = {Path(f).stem for f in wiki_added_events}
    backfills = sorted(wiki_event_ids - set(post_event_ids))
    if backfills:
        parts = []
        for eid in backfills:
            if eid in undateable:
                parts.append(f"{eid} (undated MAINLINE, placed pre-cutoff)")
            elif eid in dateable:
                d = datetime.fromtimestamp(dateable[eid]["startTime"], BEIJING).date()
                parts.append(f"{eid} (startTime {d}, pre-cutoff)")
            else:
                parts.append(f"{eid} (absent from this game-data snapshot)")
        backfill_note = ("wiki delta also lists " + "; ".join(parts) +
                         " — wiki backfills, not post-cutoff game content.")
    else:
        backfill_note = "wiki delta and game data agree on the post-cutoff events."

    # --- the manifest: the immutable pre-cutoff side -------------------------
    all_story_files = [f"{k}.txt" for k in table]
    post_cutoff = set(post_event_files) | set(post_record_files)
    pre_cutoff_stories = sorted(f for f in all_story_files if f not in post_cutoff)
    operator_profiles = sorted(p.name for p in
                               (corpus.RAW_CN_DIR / "operators").glob("*.txt"))

    # --- T0: a seeded subset of <章节> blocks inside the post-cutoff events ---
    chapters = {f: chapter_count(f) for f in sorted(post_event_files)}
    rng = random.Random(T0_SEED)
    t0_selection = {}
    for story_file, n in chapters.items():
        k = max(1, min(round(n * T0_FRACTION), n - 1))  # proper subset: partly-seen
        t0_selection[story_file] = sorted(rng.sample(range(n), k))

    manifest = {
        "schema_version": 1,
        "description": "Pre-cutoff manifest for the Track A train/test split. "
                       "See 00_data_prep/README.md.",
        "rule": "story file in pre_cutoff_stories => train; not in it => test. "
                "operator profiles => always train.",
        "provenance": {
            "cutoff": CUTOFF.isoformat(),
            "cutoff_unix": CUTOFF_TS,
            "derived_on": datetime.now(BEIJING).date().isoformat(),
            "game_data_commit": _git(GAME_DATA, "rev-parse", "HEAD"),
            "wiki_boundary_commit": _git(WIKI, "rev-parse", WIKI_BOUNDARY_COMMIT),
            "wiki_head_commit": wiki_head,
            "event_dating": "story_review_table.json startTime "
                             "(MAINLINE events have startTime -1 — see "
                             "undateable_events)",
            "record_dating": "arknights_lore_wiki git delta "
                              "(wiki_boundary_commit..wiki_head_commit)",
        },
        "counts": {
            "events_total": len(events),
            "records_total": len(records),
            "operator_profiles": len(operator_profiles),
            "undateable_events": len(undateable),
            "pre_cutoff_stories": len(pre_cutoff_stories),
            "post_cutoff_events": len(post_event_files),
            "post_cutoff_records": len(post_record_files),
        },
        "undateable_events": {
            "ids": undateable,
            "note": "MAINLINE events carry startTime -1. Placed pre-cutoff: at "
                    "this snapshot every one is a long-released main-story "
                    "chapter. A re-derivation MUST re-verify that none of "
                    "these released after the cutoff.",
        },
        "pre_cutoff_stories": pre_cutoff_stories,
        "operator_profiles_always_train": operator_profiles,
        "post_cutoff_at_snapshot": {
            "events": sorted(post_event_files),
            "records": post_record_files,
            "note": "Reference only. The rule is 'not in pre_cutoff_stories "
                    "=> test', so a newer snapshot's extra content also falls "
                    "to test — the test set grows with the game.",
        },
        "t0": {
            "unit": "0-based <章节> block ordinal within the cleaned event file",
            "seed": T0_SEED,
            "fraction": T0_FRACTION,
            "selection": t0_selection,
        },
        "wiki_cross_check": {
            "wiki_added_events": wiki_added_events,
            "game_dated_post_cutoff_events": sorted(post_event_files),
            "note": backfill_note,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    c = manifest["counts"]
    print(f"wrote {MANIFEST}")
    print(f"  cutoff              : {CUTOFF.isoformat()}")
    print(f"  events / records    : {c['events_total']} / {c['records_total']} "
          f"(+ {c['operator_profiles']} operator profiles)")
    print(f"  pre-cutoff stories  : {c['pre_cutoff_stories']}  -> train/val pool")
    print(f"  post-cutoff events  : {c['post_cutoff_events']}  {sorted(post_event_files)}")
    print(f"  post-cutoff records : {c['post_cutoff_records']}")
    print(f"  undated events      : {len(undateable)} MAINLINE "
          f"(startTime -1) -> pre-cutoff; verify on any re-derivation")
    print(f"  T0 <章节> selection : "
          f"{ {f: len(v) for f, v in t0_selection.items()} } of {chapters}")
    print(f"  cross-check         : {backfill_note}")


if __name__ == "__main__":
    main()
