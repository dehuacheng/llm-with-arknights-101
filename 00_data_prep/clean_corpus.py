#!/usr/bin/env python3
"""Stage 00 · sub-step 1 — corpus cleaning.

Strips engine markup from data/raw/cn/ and writes the result to data/clean/cn/.
The keep/strip/rewrite decisions are the table in 00_data_prep/README.md:

  keep      structural lore tags (<正文>, <干员名称>, …)            — verbatim
            speaker labels  ('凯尔希:' …)                          — verbatim
            HEADER / animtext / spellsticker *text* (stage titles, captions)
  rewrite   {@nickname} / {@Nickname}  -> 博士     {@nbs} -> removed
            [multiline(name="X")] dialogue  -> 'X:dialogue'  (canonical speaker)
  strip     [HEADER(...)] / [name=...] / [animtext(...)] / … directive syntax
            pure-engine directive lines  ([Character(...)], [delay(...)], …)
            rich-text markup  (<i>, <b>, <color=…>, <p=N>, </>)

Deterministic: identical input -> identical output. Run from the repo root:

    python3 00_data_prep/clean_corpus.py
"""
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import corpus  # noqa: E402

# Engine directives, written [Name], [Name(args)] or [Name=val] at the start
# of a line, each mapped to whether it carries human-readable text. NOTE: only
# genuine directives belong here — bracketed strings like [DWDB-221E] or
# [MULTICORE-TACTICS|…] are in-fiction lore text, not directives.
#   text-carrying : HEADER/animtext/spellsticker hold captions, name/multiline
#                   hold a speaker — that text is kept (see clean_line).
#   content-less  : on this snapshot they only ever appear alone on a line, so
#                   the whole line is dropped.
DIRECTIVE_CARRIES_TEXT = {
    "HEADER": True, "name": True, "multiline": True, "animtext": True,
    "spellsticker": True, "character": False, "charslot": False,
    "delay": False, "Background": False, "stopmusic": False,
    "CameraShake": False, "Blocker": False,
}
DIRECTIVE_NAMES = tuple(DIRECTIVE_CARRIES_TEXT)
CONTENT_LESS = {n.lower() for n, has_text in DIRECTIVE_CARRIES_TEXT.items()
                if not has_text}

_NAMES = "|".join(DIRECTIVE_NAMES)
# A directive's args take three forms: parenthesised ([HEADER(...)]),
# equals ([name=""]), or none ([character]).
_ARGS = r'(?:\([^\[\]]*\)|=[^\[\]]*)?'
LEADING_DIRECTIVE = re.compile(rf'^\s*\[({_NAMES}){_ARGS}\]', re.I)
ANY_DIRECTIVE = re.compile(rf'\[(?:{_NAMES}){_ARGS}\]', re.I)
NAME_ARG = re.compile(r'name\s*=\s*"([^"]*)"')

# Rich-text markup: <p=N> and </> separate on-screen panels (-> space); the
# rest is inline emphasis with no break (-> removed).
RICHTEXT_TO_SPACE = re.compile(r'<p=\d+>|</>')
RICHTEXT_TO_EMPTY = re.compile(r'</?i>|</?b>|<color=[^>]*>|</color>')

NICKNAME = re.compile(r'\{@[Nn]ickname\}')
NBS = re.compile(r'\{@nbs\}')

HAS_CJK = re.compile(r'[一-鿿]')
MULTISPACE = re.compile(r'[ \t]{2,}')


def clean_line(line, stats):
    """Clean one line. Returns the text, '' for a blank line, or None to drop."""
    if line.strip() == "":
        return ""  # genuine blank — kept as a paragraph break

    m = LEADING_DIRECTIVE.match(line)
    lead = m.group(1).lower() if m else None
    speaker = ""
    if lead in ("name", "multiline"):
        # name="X" on these directives is the speaker; '' means the narrator.
        m_name = NAME_ARG.search(m.group(0))
        speaker = m_name.group(1).strip() if m_name else ""

    s, n_dir = ANY_DIRECTIVE.subn("", line)
    stats["directives_stripped"] += n_dir
    s = RICHTEXT_TO_SPACE.sub(" ", s)
    s = RICHTEXT_TO_EMPTY.sub("", s)
    s, n_nick = NICKNAME.subn("博士", s)
    s, n_nbs = NBS.subn("", s)
    stats["nickname_mapped"] += n_nick
    stats["nbs_removed"] += n_nbs
    s = MULTISPACE.sub(" ", s).strip()

    if speaker:
        return f"{speaker}:{s}" if s else None  # speaker form, drop if empty
    if not s:
        return None  # directive-only line, or emptied by stripping
    if lead in CONTENT_LESS and not HAS_CJK.search(s):
        return None  # pure-engine residue (e.g. the stray ']' of '[delay(...)]]')
    return s


def clean_text(text, stats):
    """Clean a whole file: drop engine lines, collapse runs of blank lines."""
    out, prev_blank = [], True  # prev_blank=True trims leading blank lines
    for line in text.splitlines():
        r = clean_line(line, stats)
        if r is None:
            stats["lines_dropped"] += 1
            continue
        if r == "":
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(r)
            prev_blank = False
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def main():
    if not corpus.RAW_CN_DIR.is_dir():
        sys.exit(f"clean_corpus: {corpus.RAW_CN_DIR} not found — "
                 f"run tools/build_raw_data.py first.")

    stats = Counter()
    for sub in ("stories", "operators"):
        src = corpus.RAW_CN_DIR / sub
        dst = corpus.CLEAN_CN_DIR / sub
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)
        for f in sorted(src.glob("*.txt")):
            raw = f.read_text(encoding="utf-8")
            cleaned = clean_text(raw, stats)
            (dst / f.name).write_text(cleaned, encoding="utf-8")
            stats["files"] += 1
            stats["chars_in"] += len(raw)
            stats["chars_out"] += len(cleaned)

    if not stats["files"]:
        sys.exit(f"clean_corpus: no .txt files under {corpus.RAW_CN_DIR}")

    print(f"cleaned {stats['files']} files -> {corpus.CLEAN_CN_DIR}")
    print(f"  chars               : {stats['chars_in']:>10,} -> "
          f"{stats['chars_out']:>10,} "
          f"({stats['chars_out'] / stats['chars_in']:.1%} kept)")
    print(f"  directives stripped : {stats['directives_stripped']:>10,}")
    print(f"  engine lines dropped: {stats['lines_dropped']:>10,}")
    print(f"  {{@nickname}} -> 博士 : {stats['nickname_mapped']:>10,}")
    print(f"  {{@nbs}} removed     : {stats['nbs_removed']:>10,}")


if __name__ == "__main__":
    main()
