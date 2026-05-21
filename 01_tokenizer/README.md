# Stage 01 — Tokenizer (from scratch)

Track A builds its own tokenizer instead of reusing Qwen3's. This stage trains
that tokenizer on the **train split** produced by `00_data_prep/`.

> Status: **design only.** No implementation code yet.

## Decision

- **GPT-2-style byte-level BPE.** Byte-level fallback ⇒ no out-of-vocabulary
  characters ever, even for rare CJK. Trained on the **train split only**
  (never val/test), so the vocabulary is not fit to held-out text.

## Inputs

- The cleaned **train** split from `00_data_prep/` — `data/clean/splits.json`
  `train`, read via `lib.corpus.split_files("train")`. Never val/test.

## Special tokens

BPE does **not** designate special tokens on its own — it only learns frequent
merges. The structural tags kept in sub-step 1 of stage 00 — the 14 names in
`lib.corpus.STRUCTURAL_TAGS` (`<干员名称>`, `<活动名称>`, `<章节名称>`, …) — are
registered **explicitly** as added/special tokens before training, so each
becomes one reserved, atomic token ID — never split, never merged into a
neighbour. `lib.corpus.structural_tag_tokens()` returns their open + close
forms. Plus the usual BOS / EOS / PAD.

## Knobs to explore (this is a learning stage)

- **Vocab size** — the key trade-off for a *small* model: a larger vocab
  shortens sequences but its embedding table can dominate total parameters.
  Sweep candidates and plot params-vs-fertility.
- Fertility (tokens per CN character) on train vs. val/test — the gap is itself
  a lesson.

## Outputs

- The trained tokenizer, plus a short report: fertility, vocab stats, example
  segmentations of operator/faction names. Aggregate stats and a few example
  tokenizations of public names are IP-safe to commit.
