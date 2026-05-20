#!/usr/bin/env python3
"""Deterministically rebuild data/raw/ from the sibling source repos.

data/ is intentionally gitignored and never committed (Hypergryph game IP +
size). THIS SCRIPT is the committed, reproducible way to regenerate it: given
the three sibling repos at the same commits, it produces identical output.

Sources — siblings of this repo under ~/projects/:
    ArknightsGameData/        CN game-data dump          -> data/raw/cn/
    arknights_lore_wiki/      LLM-summarized wiki         -> data/raw/wiki/
    arknights_lore_wiki_lib/  deterministic parser        (build tool only)

`arknights_lore_wiki_lib` is used ONLY by this script, as a build-time parser.
Nothing else in llm-with-arknights-101 imports it — the project depends on the
two data repos as inputs and on the parser solely to (re)generate data/raw/.

The parser needs its venv (provides pypinyin). Set it up once:
    python3 -m venv ../arknights_lore_wiki_lib/.venv
    ../arknights_lore_wiki_lib/.venv/bin/pip install -r ../arknights_lore_wiki_lib/requirements.txt

Then run from the repo root with ANY python — the script re-execs itself under
the parser venv automatically:
    python3 tools/build_raw_data.py
"""
import os
import shutil
import subprocess
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # llm-with-arknights-101/
PROJECTS = os.path.dirname(REPO)                   # ~/projects/
GAME_DATA = os.path.join(PROJECTS, "ArknightsGameData")
WIKI_SRC = os.path.join(PROJECTS, "arknights_lore_wiki")
LIB = os.path.join(PROJECTS, "arknights_lore_wiki_lib")
LIB_VENV_PY = os.path.join(LIB, ".venv", "bin", "python")
OUT = os.path.join(REPO, "data", "raw")

_REEXEC_FLAG = "_BUILD_RAW_DATA_REEXEC"


def _die(msg):
    sys.exit(f"build_raw_data: {msg}")


def _ensure_lib_venv():
    """Make the lib's parser importable by re-exec'ing under its venv.

    A venv python is a symlink to the base interpreter, so comparing resolved
    executable paths is unreliable — an env-var sentinel guards the re-exec
    against looping instead.
    """
    try:
        import pypinyin  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get(_REEXEC_FLAG):
        _die(f"pypinyin still missing after switching to the parser venv — run:\n"
             f"  {LIB}/.venv/bin/pip install -r {LIB}/requirements.txt")
    if not os.path.isfile(LIB_VENV_PY):
        _die(f"parser venv not found at {LIB_VENV_PY}\n"
             f"  create it: python3 -m venv {LIB}/.venv && "
             f"{LIB}/.venv/bin/pip install -r {LIB}/requirements.txt")
    os.environ[_REEXEC_FLAG] = "1"
    os.execv(LIB_VENV_PY, [LIB_VENV_PY, os.path.abspath(__file__)] + sys.argv[1:])


def _check_sources():
    for name, path in (("ArknightsGameData", GAME_DATA),
                        ("arknights_lore_wiki", WIKI_SRC),
                        ("arknights_lore_wiki_lib", LIB)):
        if not os.path.isdir(path):
            _die(f"missing sibling repo '{name}' — expected at {path}\n"
                 f"  clone it under {PROJECTS}/ first")


def _git_commit(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001 - provenance is best-effort
        # A bare "unknown" in the manifest would defeat its reproducibility
        # purpose, so make the degradation visible rather than silent.
        print(f"build_raw_data: warning: no git commit for {path} ({e})",
              file=sys.stderr)
        return "unknown"


def _fresh_dir(path):
    """(Re)create an empty directory so a rebuild leaves no stale files."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _extract(kind, items, out_dir, render):
    """Write render(payload) to <out_dir>/<id>.txt for each (id, payload).

    Failures are collected as (kind, id, error) rather than aborting the batch.
    """
    _fresh_dir(out_dir)
    n_ok, failures = 0, []
    for item_id, payload in items:
        try:
            _write(os.path.join(out_dir, f"{item_id}.txt"), render(payload))
            n_ok += 1
        except Exception as e:  # noqa: BLE001 - batch job, collect and report
            failures.append((kind, item_id, repr(e)))
    return n_ok, failures


def build_cn():
    """Parse ArknightsGameData CN -> data/raw/cn/ via the lib's parser."""
    sys.path.insert(0, LIB)  # import the parser straight from the sibling repo
    from libs.core.game_data import (
        extract_data_from_story_review_table, get_all_text_from_event,
        get_all_char_info, get_char_info_text_prompt,
    )
    print("--- stories ---")
    events = extract_data_from_story_review_table(GAME_DATA)
    n_story, story_fail = _extract(
        "story", events.items(), os.path.join(OUT, "cn", "stories"),
        lambda ev: get_all_text_from_event(GAME_DATA, ev))

    print("--- operators ---")
    char_info, _ = get_all_char_info(GAME_DATA)
    # char ids present only in voice/skin tables have no profile to write
    profiled = [(cid, v) for cid, v in char_info.items() if v.get("name")]
    n_ops, op_fail = _extract(
        "operator", profiled, os.path.join(OUT, "cn", "operators"),
        get_char_info_text_prompt)

    return {"stories": n_story, "operators": n_ops,
            "skipped": len(char_info) - len(profiled),
            "failures": story_fail + op_fail}


def build_wiki():
    """Copy arknights_lore_wiki/data/ verbatim -> data/raw/wiki/."""
    print("--- wiki ---")
    src = os.path.join(WIKI_SRC, "data")
    if not os.path.isdir(src):
        _die(f"expected {src} — is arknights_lore_wiki checked out?")
    dst = os.path.join(OUT, "wiki")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    return sum(len(files) for _, _, files in os.walk(dst))


def write_manifest(cn, n_wiki):
    gd, lib, wiki = (_git_commit(GAME_DATA), _git_commit(LIB),
                     _git_commit(WIKI_SRC))
    text = f"""# data/raw — provenance (generated by tools/build_raw_data.py)

Generated {date.today().isoformat()}. All of data/ is gitignored and never
committed; rebuild deterministically with:

    python3 tools/build_raw_data.py

## cn/ — CN ground-truth corpus
- source : ArknightsGameData @ {gd}
- parser : arknights_lore_wiki_lib @ {lib} (libs/core/game_data.py)
- cn/stories/    {cn['stories']} event story scripts (dialogue + narration)
- cn/operators/  {cn['operators']} operator profiles \
({cn['skipped']} char ids skipped: no profile)

## wiki/ — LLM-summarized wiki (NOT ground truth; eval/test only)
- source : arknights_lore_wiki @ {wiki} (data/ copied verbatim)
- {n_wiki} files

## IP
cn/ is derived from Hypergryph (明日方舟) game data. Per plan.md stage 8, raw
in-game text is never redistributed — only this regen script and the source
commit hashes above are committed.
"""
    _write(os.path.join(OUT, "MANIFEST.md"), text)


def main():
    _ensure_lib_venv()
    _check_sources()
    os.makedirs(OUT, exist_ok=True)
    print(f"game data : {GAME_DATA}\nwiki src  : {WIKI_SRC}\n"
          f"parser    : {LIB}\noutput    : {OUT}\n")

    cn = build_cn()
    n_wiki = build_wiki()
    write_manifest(cn, n_wiki)

    print("\n" + "=" * 52)
    print(f"cn/stories   : {cn['stories']}")
    print(f"cn/operators : {cn['operators']} (skipped {cn['skipped']})")
    print(f"wiki/        : {n_wiki} files")
    for kind, name, err in cn["failures"]:
        print(f"  FAILED {kind} {name}: {err}")
    if cn["failures"]:
        sys.exit(1)
    print("data/raw rebuilt OK")


if __name__ == "__main__":
    main()
