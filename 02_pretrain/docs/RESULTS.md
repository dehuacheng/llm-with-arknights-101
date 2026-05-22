# Stage 02 — results

Pretraining experiments. Each row is one `configs/*.yaml` run. Per the project
convention, record surprises and failures here, not only the clean numbers.

> Status: **not yet run.** The tables below are the experiment plan with empty
> result columns — fill them as runs complete. Parameter counts are exact;
> perplexity / bits-per-character come from `train.py`'s final report.

## Setup

Trained on the Stage 00 `train` split (838 files, 11.66M chars), tokenized by a
Stage 01 tokenizer. Every run sees the same token budget: `max_steps × batch ×
block` = 8000 × 32 × 512 ≈ 131M tokens. `train.py` keeps the checkpoint with
the lowest val loss. Hardware: RTX 4090.

## Scale axis — vocab fixed at 32k

How model size trades against a fixed, small dataset. Perplexity is comparable
down this column (same tokenizer throughout). Watch the train → val gap: it is
the overfitting signal.

| Run         | Scale | Total params | Train ppl | Val ppl | Best step | Wall time |
|-------------|-------|--------------|-----------|---------|-----------|-----------|
| `tiny_32k`  | tiny  | 11.7M        | —         | —       | —         | —         |
| `small_32k` | small | 23.4M        | —         | —       | —         | —         |
| `large_32k` | large | 42.3M        | —         | —       | —         | —         |

## Vocab axis — model fixed at `small`

How vocabulary size changes the model. Perplexity is **not** comparable across
this column — a token covers a different amount of text under each tokenizer —
so compare **bits-per-character**.

| Run         | Tokenizer | Total params | Val ppl | Val bits/char | Best step |
|-------------|-----------|--------------|---------|---------------|-----------|
| `small_8k`  | vocab_8k  | 14.0M        | —       | —             | —         |
| `small_16k` | vocab_16k | 17.1M        | —       | —             | —         |
| `small_32k` | vocab_32k | 23.4M        | —       | —             | —         |

(`small_32k` is the shared cell — it appears in both tables.)

## Observations

_To be filled after the runs. Questions the sweep should answer:_

- **Scale vs. data.** Does `large_32k` overfit — train ppl falling while val
  ppl stalls or climbs? At what step does each run's best val checkpoint land?
- **Vocab vs. model.** Bigger vocab = shorter sequences (more context per
  block) but a bigger embedding table to learn on the same data. Which way
  does bits-per-character move across 8k → 16k → 32k?
- **Embedding budget.** At 32k vocab, `small` is 55% embedding table. Does the
  model spend its capacity well, or is much of that table undertrained?

## Failures / surprises

_None recorded yet._
