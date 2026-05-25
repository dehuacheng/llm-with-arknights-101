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
- `05_dpo/` **implemented — all 4 cells of the 2×2 complete** —
  Stage 05 preference optimisation on top of `sft_full`. All five
  `train_dpo.py` EXERCISE blocks (`dpo-format`, `dpo-logprobs`,
  `dpo-loss`, `ipo-loss`, `train-loop`) implemented. `derive_val_split.py`
  produced stratified-by-`fault_type` val splits (seed 1337);
  `04_sft/chat.py` extended with `--adapter` to layer Stage-05 LoRA on the
  Stage-04 base. Four runs done: `dpo_curated` (val_loss 0.064 / drift
  +0.31), `ipo_curated` (val_loss 6.04 / drift +0.013), `dpo_bulk`
  (val_loss 0.064 / drift +0.98), `ipo_bulk` (val_loss 6.89 / drift
  +0.018). **Headline finding (confirmed at scale):** every cell fully
  satisfies the preference objective (val_acc ≥ 0.92), but T=0 argmax
  probes are largely unchanged from the SFT baseline — the gap between
  "prefer chosen over rejected" and "emit chosen at argmax" is the
  central result, and 5× more pairs does not close it. Two textbook
  effects materialised cleanly: **DPO is unbounded (bulk tripled drift
  for zero benefit), IPO ceilings at `1/(2β)` (bulk = curated for drift).**
  Details in `05_dpo/docs/RESULTS.md`. Hand-rolled DPO/IPO losses;
  `trl.DPOTrainer` not used.
- `06_rlvr/` **implemented; two cells run, both informative** — Stage
  06 RLVR / GRPO with verifiable rewards on `data/rl/prompts_train.jsonl`
  (agent-shipped, 925 factoid prompts; 30% with `must_not_contain`
  post-enrichment; val derived to 98 via `derive_val_split.py` with
  hash-by-id stratification on `category`). All five `train_rlvr.py`
  EXERCISE blocks implemented. Verifiable reward (`reward.py`, 15 unit
  tests pass) is asymmetric `(matched_facts / |K|) − 0.5·(traps / |M|)`
  with optional length + fluency penalties; substring matcher splits
  bilingual `中文 / English` facts on `/`. Starting policy =
  `data/checkpoints/sft_full`. Single-step GRPO, ε=0.2, N=8, on-policy.
  Three hard-cap tripwires (mean KL > 0.5, max KL > 2.0, SFT-CE drift >
  +1.0). **Two runs**: (1) `grpo_baseline` (β=0.04, LR=5e-6) — 17.3
  min wall, mode-collapsed at step 100, KL hard-cap fired; (2)
  `grpo_v2` (β=0.10, LR=1e-6, fluency_penalty_cap=0.3) — 128.9 min
  wall, **early-stopped on patience after val_reward plateaued; best
  +0.078 at step 350; KL stayed bounded at 0.140 vs cap 0.5; no
  collapse**. The three named fixes worked exactly as predicted.
  **But the Stage-06 headline is darker:** v2's T=0 argmax probes
  show **Kal'tsit is still 萨卡兹** — same as after SFT, DPO×4, IPO×2,
  GRPO. **Six different training mechanisms; six different headline
  numbers; one argmax answer.** The argmax gap Stage 05 documented
  also holds for on-policy RL with verifiable rewards. The reason is
  mechanistic (gradient never sees model's own top-1; reference KL
  anchor outweighs reward at this scale; LoRA r=16 caps first-token
  redistribution). Two follow-up levers identified but deferred:
  rejection-sampling-at-inference and self-corrected-DPO (sample from
  model, label its argmax as rejected). Full diagnosis +
  six-mechanism comparison table in `06_rlvr/docs/RESULTS.md`.
