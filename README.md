# llm-with-arknights-101

**English | [简体中文](README.zh-CN.md)**

> An end-to-end, educational walkthrough of continued-pretraining and
> post-training a sub-1B language model — using *Arknights* lore as the corpus.

This repo teaches the full small-LLM training pipeline by example. The "student"
model is **Qwen3-0.6B**; the domain is the lore of the game *Arknights* (明日方舟),
chosen because it is rich, naturally multilingual (CN/EN/JP), and has a clear
notion of ground truth to evaluate against.

It is a learning project — every stage is meant to be read, run, and modified.

> [!IMPORTANT]
> **Not affiliated with Hypergryph (鹰角网络).** *Arknights* and all related text
> and characters are the property of Hypergryph. This project is non-commercial
> and for educational use only. It does **not** redistribute raw in-game text —
> see [Data & IP](#data--ip).

## Roadmap

The project is split into independent stages; each produces a checkpoint or
artifact the next consumes.

| Stage | Status |
|---|---|
| Setup & model pick — Qwen3-0.6B base, `transformers + peft + trl` stack | decided |
| **Raw-data generation** — build the lore corpus | ✅ |
| Continued pretraining (CPT) — full-FT vs LoRA, with a replay mix | ✅ Stage 03 |
| SFT distillation — closed-book Q&A distilled from a teacher LLM | ✅ Stage 04 |
| DPO/IPO — preference pairs against plausible hallucinations | ✅ Stage 05 |
| **RLVR / GRPO** — verifiable rewards close the preference-vs-argmax gap | ✅ Stage 06 (scaffolded) |
| Refusal training — "I don't know" on out-of-canon questions | planned (Stage 07) |
| Thinking distillation — optional `<think>` traces | planned (Stage 08) |
| Evaluation & publish — ~200-question hand-graded test set | planned (Stage 09) |

The evaluation set is authored *before* training starts, so every stage can be
re-scored against the same questions. Detailed rationale for each decision lives
in `plan.md` (planning workspace, not committed here).

Only **raw-data generation** is in scope for the current commit. Later stages
will each land in their own top-level folder (`tokenizer/`, `pretrain/`, …).

## Repository layout

```
llm-with-arknights-101/
├── README.md             # English overview (this file)
├── README.zh-CN.md       # 简体中文版
├── AGENTS.md             # durable decisions for AI coding agents
├── CLAUDE.md             # imports AGENTS.md for Claude Code
├── requirements.txt      # Python deps (minimal for now)
├── tools/
│   └── build_raw_data.py # regenerates the corpus
└── data/                 # generated; git-ignored; never committed
    └── raw/
        ├── cn/stories/   # CN event story scripts (ground truth)
        ├── cn/operators/ # CN operator profiles (ground truth)
        ├── wiki/         # LLM-summarized wiki (eval/test only)
        └── MANIFEST.md   # provenance: source repos + commit hashes
```

## External data dependencies

This repo holds *code*, not data. The corpus is rebuilt from three upstream
repositories, which must be cloned as **siblings of this repo** (same parent
directory):

| Repo | Role | Used by `build_raw_data.py` as |
|------|------|----|
| [`Kengxxiao/ArknightsGameData`](https://github.com/Kengxxiao/ArknightsGameData) | Raw CN game-data dump — **ground truth** | input → `data/raw/cn/` |
| [`littlepangding/arknights_lore_wiki`](https://github.com/littlepangding/arknights_lore_wiki) | LLM-summarized lore wiki — *not* ground truth | input → `data/raw/wiki/` |
| [`littlepangding/arknights_lore_wiki_lib`](https://github.com/littlepangding/arknights_lore_wiki_lib) | Deterministic parser for the game-data dump | build-time parser only |

> [!NOTE]
> **Dependency boundary.** `arknights_lore_wiki_lib` is used **only** by
> `tools/build_raw_data.py`, purely as a build-time parser. Nothing else in this
> project imports it, and it is not vendored or pip-installed. The project's
> standing dependencies are the two *data* repos.

### Clone the dependencies

```bash
# from the parent directory that contains this repo
git clone --depth 1 https://github.com/Kengxxiao/ArknightsGameData.git
git clone https://github.com/littlepangding/arknights_lore_wiki.git
git clone https://github.com/littlepangding/arknights_lore_wiki_lib.git
```

Resulting layout:

```
projects/
├── llm-with-arknights-101/   # this repo
├── ArknightsGameData/
├── arknights_lore_wiki/
└── arknights_lore_wiki_lib/
```

### Pinned commits (for byte-identical output)

`build_raw_data.py` is deterministic *given the source repos at fixed commits*.
The corpus this repo is committed against was built from:

| Repo | Commit |
|------|--------|
| ArknightsGameData | `de85be497656612992f25092d3df2045476d6dc1` |
| arknights_lore_wiki | `9c2f8cb64f2453802989808890bb031333b17cca` |
| arknights_lore_wiki_lib | `fbf68f3eaf14ee7032b8b9941280c9f4f9b6a777` |

To reproduce exactly, `git checkout` each commit (a full clone is needed — a
`--depth 1` clone only fetches current `HEAD`). Each run also writes the commits
it actually used to `data/raw/MANIFEST.md`.

## Environment setup

**Raw-data generation needs almost nothing.** `tools/build_raw_data.py` uses
only the Python standard library. Its one requirement is the parser repo's
virtualenv, which supplies the parsing dependencies (`pypinyin`, …).

```bash
# 1. Python 3.10+
python3 --version

# 2. Create the parser venv, once (run from this repo's root)
python3 -m venv ../arknights_lore_wiki_lib/.venv
../arknights_lore_wiki_lib/.venv/bin/pip install -r ../arknights_lore_wiki_lib/requirements.txt
```

`build_raw_data.py` re-execs itself under that venv automatically — you can
launch it with any Python 3.

`requirements.txt` in this repo is intentionally minimal for now; later stages
(CPT, SFT, …) will pin their own training dependencies (`torch`, `transformers`,
`peft`, `trl`, …) as they land.

## Regenerate the training data

From the repo root:

```bash
python3 tools/build_raw_data.py
```

This writes, all under the git-ignored `data/`:

| Output | Contents | Count |
|--------|----------|-------|
| `data/raw/cn/stories/` | CN event story scripts (dialogue + narration) | 461 |
| `data/raw/cn/operators/` | CN operator profiles | 444 |
| `data/raw/wiki/` | LLM-summarized wiki pages (eval/test only) | 3000 files |
| `data/raw/MANIFEST.md` | provenance — source repos + commit hashes | — |

`cn/` is the **ground-truth** corpus for training; `wiki/` is *model-generated*
summaries and should only be used for evaluation/testing, never as a
training target of record.

## Data & IP

- The entire `data/` directory is **git-ignored and never committed** — it is
  large and derived from Hypergryph game IP.
- This repo ships the **regeneration script + source commit hashes** so the
  corpus is reproducible without redistributing it.
- *Arknights* / 明日方舟 and its text and characters © Hypergryph (鹰角网络).
  This project has **no affiliation** with Hypergryph, is **non-commercial**,
  and is for **educational use only**.

## License

- Code: **Apache-2.0** (planned).
- The hand-graded evaluation set, when it lands: **CC-BY-4.0**.
- Game-derived text is **not** licensed by this project and is never
  redistributed here.
