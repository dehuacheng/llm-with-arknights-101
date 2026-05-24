# AGENTS.md — guidance for AI coding agents

Shared by all coding agents working on this repo (Claude Code, Codex, …).
Claude Code also loads it via `CLAUDE.md`, which imports this file. Keep it
terse and current — when a decision changes, update this file.

## What this project is

An educational, end-to-end project about training a small LLM on *Arknights*
(明日方舟) lore. It runs **two tracks and compares them**: Track A builds a
tokenizer and a small model *from scratch* (for learning the mechanics);
Track B continued-pretrains and post-trains **Qwen3-0.6B base**. The
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
   vendored, not in `requirements.txt`, and nothing in a future `01_tokenizer/`,
   `02_…/`, etc. may import it. The project depends on the two *data* repos
   (`ArknightsGameData`, `arknights_lore_wiki`) as inputs.

3. **The three source repos are siblings**, cloned under the same parent
   directory as this repo. `build_raw_data.py` locates them via `../<name>`.
   Don't hardcode absolute paths.

4. **`cn/` is ground truth; `wiki/` is not.** `data/raw/cn/` is parsed straight
   from the game data and is the training target of record. `data/raw/wiki/` is
   LLM-summarized — use it for evaluation/testing only.

## Conventions (carry into future stages)

- Stages are **numbered top-level folders** (`00_data_prep/`, `01_tokenizer/`,
  `02_…/`). Each folder owns that stage's guide/notes, scripts, experiment
  attempts, and results — kept together, independently runnable, producing the
  artifact the next stage consumes.
- Shared, reusable code lives in a top-level `lib/` (an importable package),
  not inside a stage folder. Stage folders hold stage-specific scripts; `lib/`
  holds what more than one stage needs.
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
- Learning mode: in from-scratch (Track A) code, the pedagogically central
  regions are wrapped in `# === EXERCISE START/END: <slug> ===` comment blocks
  carrying a Concept/Given/Produce/Steps spec. The committed file stays the
  working reference; a learner reimplements within the markers (`git diff`
  recovers the reference). Markers only — no generated stub file. Carry this
  into every Track A stage.
- Track A model code (PyTorch) follows the **Stanford CS336** convention:
  named-axis `einops` (`rearrange` / `einsum`) instead of `.view()` /
  `.transpose()` / `.reshape()`, plus `jaxtyping` shape annotations. Keeps the
  from-scratch model aligned with that course's idioms.

## Fixed decisions

- **Two tracks, compared at the end.** Track A — build a tokenizer and a small
  LM *from scratch* on the Arknights corpus; for learning the full mechanics,
  not expected to be strong (small data). Track B — continued-pretrain /
  fine-tune Qwen3-0.6B. The payoff is the A-vs-B comparison on a shared eval.
- Track B base model: **Qwen3-0.6B base** (best sub-1B CJK tokenizer;
  Apache-2.0; has base / instruct / thinking variants for ablations).
- Track A tokenizer: **byte-level BPE implemented from scratch** — pure Python,
  **no HuggingFace dependency**. `lib/bpe.py` (`ByteBPE`: train / encode /
  decode / save / load, custom JSON format), trained on the train split only.
  Stage 02 loads it via `ByteBPE.load`, not an HF tokenizer. See `01_tokenizer/`.
- **Nested test sets.** Track A's split is a *family* of growing test sets
  T0 ⊂ T1 ⊂ T2, so models trained with different held-out content are scored on
  a common yardstick (T0). Design in `00_data_prep/`.
- Code license target: **Apache-2.0**; evaluation set: **CC-BY-4.0**.
- Default branch: `main`.

## Current status

- Raw-data generation complete: `tools/build_raw_data.py`.
- `00_data_prep/` **implemented** — `clean_corpus.py` (cleaning), `derive_split.py`
  (one-time split derivation → committed `split_manifest.json`), `apply_split.py`
  (manifest → `data/clean/splits.json`). Shared code in `lib/corpus.py`.
- `01_tokenizer/` **implemented** — `lib/bpe.py` (from-scratch `ByteBPE`),
  `train_tokenizer.py`, `fertility.py`, `trace.py`, `configs/vocab_*.yaml`.
  Vocab sweep run; experiment report (EN + 中文) is `01_tokenizer/README*.md`,
  raw sweep data in `01_tokenizer/docs/RESULTS.md`. `lib/bpe.py` core regions
  carry `EXERCISE` learning-mode markers (see README §7).
- `02_pretrain/` **implemented** — hand-rolled GPT in `lib/model.py`; `train.py`
  (with gradient accumulation + early stopping), `sample.py`, `eval_probes.py`
  (inference probes, reading the editable `probes.txt`), `configs/` for a
  nine-run three-axis sweep (scale × vocab × context: `{tiny,small,large}_32k`,
  `small_{8k,16k}`, `ctx_{256,1024,2048,4096}`). All nine runs complete. Design
  + metrics in `02_pretrain/README.md`, result tables in
  `02_pretrain/docs/RESULTS.md`, in-universe inference write-up in
  `02_pretrain/docs/FIELD_REPORT.md` (EN + 中文). `lib/model.py`, `train.py`,
  `sample.py`, `eval_probes.py` carry `EXERCISE` markers. Repo venv at
  `.venv/` (git-ignored); `data/tokenized/` + `data/checkpoints/` git-ignored.
- `eval/` **scaffolded** — the shared, hand-graded ~200-question evaluation
  set every later stage is scored against; **CC-BY-4.0** (rest of the repo is
  Apache-2.0). Design + schema + a 15-item seed authored before Stage 03
  training; full set grows in dedicated authoring sessions. `validate.py`
  guards the schema. cn/-traceable sources only; `wiki/` allowed only when the
  underlying `cn/` fact has been verified.
- `03_cpt/` **scaffolded (Track B begins)** — Stage 03 continued pretraining
  of Qwen3-0.6B-Base. `README.md` (2×2 ablation design: full-FT vs LoRA × no-
  replay vs replay), `requirements.txt`, four configs, `prepare_data.py`
  (re-tokenize Stage-00 splits with Qwen's tokenizer to
  `data/tokenized/qwen3-0.6b-base/`), `prepare_replay.py` (fetch + tokenize
  the zh-wiki replay slice, fixed-seed shuffle for cross-run comparability),
  `train_cpt.py` skeleton with `EXERCISE` markers (`lora-injection`,
  `replay-mix`, `pack-sequences`, `eval-loss`, `train-loop`). Base weights
  pre-fetched to `data/models/qwen3-0.6b-base/` (git-ignored, 1.2 GB).
  Replay corpus pinned: **Chinese Wikipedia**
  (`wikimedia/wikipedia:20231101.zh`), 10k train + 500 val articles at
  seed 1337.
- `data_gen/` **brief authored** — spec for an external Arknights-knowledge
  agent that produces the project's training/eval/preference data.
  `AGENT_BRIEF.md` is the full schema reference (eval items, SFT pairs,
  DPO bulk + curated pairs, RL prompts; bilingual `中文 / English` fact
  convention; cn/-only sourcing rule; ~200 / 5000 / 5000+500 / 2000 volume
  targets; judge-mode invocation spec). `examples/` holds gold-standard
  rows in each format. Output paths under `data/sft/`, `data/dpo/`,
  `data/rl/` (all git-ignored).
- `04_sft/` **implemented (sft_full only; sft_lora skipped)** — Stage 04 SFT
  distillation: teach the Stage-03 CPT checkpoint to answer questions in
  Qwen3 chat format. `train_sft.py` (`sft-format` + `train-loop` EXERCISE
  filled in; gradient checkpointing on by default — same Qwen3 151K-vocab
  fp32-logits-cast constraint as Stage 03), `chat.py` (probe battery + REPL),
  `derive_val_split.py` (stratified-by-category one-time val split from the
  agent-shipped JSONL; seed 1337). One real run: `sft_full` from the Stage-03
  `full_ft_replay` winner, 21 min wall, best val ppl 19.29. Smoke-test
  probes recorded in `04_sft/docs/RESULTS.md` — mechanics work but content
  surfaces all four named hallucination modes, motivating Stage 05.
- `05_dpo/` **scaffolded** — Stage 05 DPO against plausible hallucinations.
  `README.md` (2×2 ablation: DPO vs IPO × bulk vs curated; the project's
  named failure mode finally addressed), `requirements.txt`, four configs,
  `train_dpo.py` with `EXERCISE` markers (`dpo-format`, `dpo-logprobs`,
  `dpo-loss`, `ipo-loss`, `train-loop`). Holds policy + frozen reference
  (both starting from the Stage 04 SFT winner). Consumes agent-produced
  JSONL from `data/dpo/{bulk,curated}_{train,val}.jsonl`. Hand-rolled
  DPO/IPO losses; `trl.DPOTrainer` not used.
