# AGENTS.md — guidance for AI coding agents

Shared by all coding agents working on this repo (Claude Code, Codex, …).
Claude Code also loads it via `CLAUDE.md`, which imports this file. Keep it
terse and current — when a decision changes, update this file.

## What this project is

An educational, end-to-end project that continued-pretrains and post-trains a
sub-1B LLM (**Qwen3-0.6B base**) on *Arknights* (明日方舟) lore. The
reader-facing overview and staged roadmap are in `README.md`. The detailed
rationale for each decision lives in the planning workspace `plan.md`, kept
alongside this repo but **not committed here** (a sibling `arknights-llm/`
folder is the planning workspace).

This repo is the source of truth for **how**; the planning workspace is the
source of truth for **why**.

## Hard rules

1. **`data/` is never committed.** The whole directory is git-ignored. It is
   large and derived from Hypergryph game IP. The corpus is reproduced from the
   committed `tools/build_raw_data.py` plus the source commit hashes recorded
   in `README.md` — never by checking data in. Do not redistribute raw in-game
   text.

2. **Do not take a runtime dependency on `arknights_lore_wiki_lib`.** It is a
   build-time parser, imported *only* by `tools/build_raw_data.py`. It is not
   vendored, not in `requirements.txt`, and nothing in a future `tokenizer/`,
   `pretrain/`, etc. may import it. The project depends on the two *data* repos
   (`ArknightsGameData`, `arknights_lore_wiki`) as inputs.

3. **The three source repos are siblings**, cloned under the same parent
   directory as this repo. `build_raw_data.py` locates them via `../<name>`.
   Don't hardcode absolute paths.

4. **`cn/` is ground truth; `wiki/` is not.** `data/raw/cn/` is parsed straight
   from the game data and is the training target of record. `data/raw/wiki/` is
   LLM-summarized — use it for evaluation/testing only.

## Conventions (carry into future stages)

- Each stage gets its own top-level folder (`tokenizer/`, `pretrain/`, `sft/`,
  …), independently runnable, producing the artifact the next stage consumes.
- Each stage pins its own deps in a per-stage requirements file; the root
  `requirements.txt` stays minimal.
- Stack: transparent `transformers + peft + trl` — **not** Unsloth (chosen for
  learning clarity over speed; revisit only if training time blocks progress).
- Long training runs go in `tmux`; never block the foreground for hours.
- Configs as YAML, one file per experiment; clone-and-rename, never edit in
  place.
- Document failures, not just successes (a `docs/RESULTS.md` per stage).
- Every training stage gets a `--smoke-test` flag for a <10-minute end-to-end
  run.
- The ~200-question evaluation set is authored *before* training begins, so
  every stage can be re-scored against it.

## Fixed decisions

- Model: **Qwen3-0.6B base** (best sub-1B CJK tokenizer; Apache-2.0; has
  base / instruct / thinking variants for ablations).
- Code license target: **Apache-2.0**; evaluation set: **CC-BY-4.0**.
- Default branch: `main`.

## Current status

- Raw-data generation complete: `tools/build_raw_data.py`.
- No training code yet — `tokenizer/`, `pretrain/`, … not created.
