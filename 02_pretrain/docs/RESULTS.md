# Stage 02 — results

Pretraining experiments. Each row is one `configs/*.yaml` run. Per the project
convention, record surprises and failures here, not only the clean numbers.

> Status: **all three axes complete** (9 runs). Parameter counts are exact;
> perplexity / bits-per-character are `train.py`'s best-val report. The
> inference probes and the in-universe discussion are in
> [`docs/FIELD_REPORT.md`](FIELD_REPORT.md). Hardware: RTX 4090.

## Setup

Trained on the Stage 00 `train` split (838 files, 11.66M chars → 6.20M tokens
at vocab_32k), tokenized by a Stage 01 tokenizer. Every run sees the same token
budget — **`effective_batch × block_size × max_steps` ≈ 131M tokens** (~21
passes over the corpus) — so the comparison is fair however batch/block/steps
are traded off. `train.py` keeps the checkpoint with the lowest val loss; train
and val loss are both averaged over 100 batches with dropout off.

## Scale axis — vocab fixed at 32k

How model size trades against a fixed, small dataset. Perplexity is comparable
down this column (same tokenizer). `train ppl` is the final-step estimate.

| Run         | Scale | Total params | Train ppl | Val ppl | Best step | Wall time |
|-------------|-------|--------------|-----------|---------|-----------|-----------|
| `tiny_32k`  | tiny  | 11.7M        | 93.2      | 148.7   | 8000      | 9.7 min   |
| `small_32k` | small | 23.4M        | 56.8      | 118.4   | 8000      | 18.3 min  |
| `large_32k` | large | 42.3M        | 34.5      | 116.4   | 7500      | 30.3 min  |

## Vocab axis — model fixed at `small`

How vocabulary size changes the model. Perplexity is **not** comparable across
this column (a token means a different amount of text under each vocab) — so
compare **bits-per-character**.

| Run         | Tokenizer | Total params | Val ppl | Val bits/char | Best step |
|-------------|-----------|--------------|---------|---------------|-----------|
| `small_8k`  | vocab_8k  | 14.0M        | 54.7    | 3.918         | 8000      |
| `small_16k` | vocab_16k | 17.1M        | 81.8    | 3.904         | 8000      |
| `small_32k` | vocab_32k | 23.4M        | 118.4   | 3.932         | 8000      |

(`small_32k` is the shared cell — it appears in both tables.)

## Context axis — `small`, vocab 32k

How context length helps, at a fixed token budget. Effective batch is held at
32; `max_steps` scales so every run still sees ~131M tokens.

| Run        | block | micro × accum | max_steps | Train ppl | Val ppl | Val bits/char | Wall time |
|------------|-------|---------------|-----------|-----------|---------|---------------|-----------|
| `ctx_256`  | 256   | 32 × 1        | 16000     | 47.3      | 120.08  | 3.944         | 14.0 min  |
| `small_32k`| 512   | 32 × 1        | 8000      | 56.8      | 118.40  | 3.932         | 18.3 min  |
| `ctx_1024` | 1024  | 16 × 2        | 4000      | 84.5      | 143.29  | 4.089         | 24.4 min  |
| `ctx_2048` | 2048  | 4 × 8         | 2000      | 145.3     | 199.39  | 4.361         | 34.8 min  |
| `ctx_4096` | 4096  | 2 × 16        | 1000      | 262.9     | 326.23  | 4.767         | 57.4 min  |

## Observations

### Scale axis — overfitting, made visible

- **Diminishing returns.** Val perplexity falls 148.7 → 118.4 → 116.4. The
  tiny→small step buys 30 points; small→large buys only **2**. On a 6M-token
  corpus, capacity past `small` barely helps generalization.
- **The train→val gap widens with scale.** Train ppl keeps plunging — 93 → 57
  → 34 — while val stalls. The val/train ratio goes 1.6× → 2.1× → **3.4×**. The
  `large` model has the capacity to memorize the train set; that capacity does
  not transfer. This is the headline result, and it is textbook overfitting.
- **`large_32k` crossed the line mid-run.** Its best val (116.38) was at step
  **7500**, and step 8000 was *worse* (116.43) — val perplexity turned upward
  while train kept dropping. `tiny` and `small` were still improving at step
  8000 (every eval saved a new best). So `large` is past its useful size here;
  `small` is about right.
- **Cost.** 9.7 / 18.3 / 30.3 min — `large` costs 3× `tiny` for a 1.6% val gain
  over `small`.

### Vocab axis — vocabulary size barely matters here

- **Perplexity is not comparable across vocabs.** It rises 54.7 → 81.8 → 118.4
  purely because a larger vocab has more classes per token (and each token
  carries more text). The honest metric is bits-per-character.
- **Bits-per-character is nearly flat: 3.918 / 3.904 / 3.932** — a 0.7% spread
  across a 4× vocab range. The three tokenizers model the corpus essentially
  equally well; 16k is marginally best.
- **32k is marginally the *worst* in bits/char**, despite being the largest
  model. Its embedding table is 12.8M params (55% of the model) and many of its
  32k tokens are rare — undertrained on 6M tokens. A bigger vocab buys shorter
  sequences (Stage 01: ~10% shorter at 32k vs 16k) but spends capacity on a
  table the data cannot fill. For a corpus this small, **8k–16k is the sweet
  spot** — same modeling quality, smaller and faster.

### Context axis — longer context, at a fixed budget, made it worse

- **Bits-per-character rose monotonically with the window**: 3.944 → 3.932 →
  4.089 → 4.361 → 4.767 for block 256 / 512 / 1024 / 2048 / 4096. The shortest
  windows won; block 4096 came in **21% worse** than block 512.
- **Why — the long runs are step-starved, not overfit.** Every run sees the
  same ~131M tokens, so a longer window buys proportionally fewer optimiser
  steps: 16000 / 8000 / 4000 / 2000 / **1000**. *Train* perplexity climbs right
  alongside val (47 → 57 → 85 → 145 → 263) — the long-context models fit the
  *training* set worse too. That is the signature of undertraining, not
  overfitting. And every context run saved its best checkpoint at its **final**
  step: all four were still improving when they ran out of steps.
- **So this is a result about budget, not about context.** It does not say a
  long window is useless — it says a 4096-token window needs far more than
  1000 updates to pay off. Held to a fixed *data* budget on a small corpus,
  the compute is better spent on more steps over a short window. block 256–512
  is the sweet spot here; the design (constant token budget) is what forces
  the trade.
- **Cost compounds the verdict.** Wall time ran 14 → 18 → 24 → 35 → 57 min —
  block 4096 cost 3× block 512 to land 21% worse.

### The corpus is the bottleneck

Across the whole sweep, every model from `small` up sits at **3.90–3.93
bits/char** regardless of scale or vocabulary; only `tiny` is materially worse
(4.12). The corpus has a hard floor around ~3.9 bits/char that a `small` model
already reaches — more parameters do not lower it. The 6M-token corpus, not the
model, is the limit. (This is exactly why the project also runs Track B —
continued-pretraining Qwen3-0.6B — for comparison.)

## Failures / surprises

- No run crashed; all nine completed at exit 0.
- _Surprise:_ the vocab axis was expected to favor the larger vocabulary
  (shorter sequences, more context per block). It did not — bits/char was flat
  and 32k was slightly worst. The embedding-table-undertraining effect on a
  small corpus outweighed the sequence-length benefit.
- _Surprise:_ the context axis was expected to show a longer window helping,
  or at least overfitting the way the scale axis did. Instead every
  longer-context run was simply **undertrained** — the fixed token budget
  starved it of steps. The fix is not a longer window but more compute.
