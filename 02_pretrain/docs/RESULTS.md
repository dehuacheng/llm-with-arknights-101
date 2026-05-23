# Stage 02 — results

Pretraining experiments. Each row is one `configs/*.yaml` run. Per the project
convention, record surprises and failures here, not only the clean numbers.

> Status: **all three axes complete** (9 runs). Numbers are `train.py`'s best-val
> report — every run was **trained to its own validation minimum** (constant
> LR + early stopping, patience 10 evals without improvement before halt; the
> `eval_interval` varies per run so the *step* count of "10 evals" is run-specific).
> The inference probes and the in-universe discussion are in
> [`docs/FIELD_REPORT.md`](FIELD_REPORT.md). Hardware: RTX 4090.

## Setup

Trained on the Stage 00 `train` split (838 files, 11.66M chars → 6.20M tokens
at vocab_32k), tokenized by a Stage 01 tokenizer. **Effective batch = 32** is
the only quantity held constant across the sweep; `max_steps` is only a safety
cap, and every run stops when val loss stalls. `train.py` keeps the checkpoint
with the lowest val loss; train and val loss are both averaged over 100 batches
with dropout off.

An earlier design instead fixed the *token budget* (the same ≈131M tokens for
every run). That made the context axis unfair — a wider window forced fewer
steps and starved those runs. Early stopping is the fix.

## Scale axis — vocab fixed at 32k

How model size trades against a fixed, small dataset. Perplexity is comparable
down this column (same tokenizer). `train ppl` is the best-val-step estimate.

| Run         | Scale | Total params | Train ppl | Val ppl | Val bits/char | Best step | Wall time |
|-------------|-------|--------------|-----------|---------|---------------|-----------|-----------|
| `tiny_32k`  | tiny  | 11.7M        | 50.75     | 135.47  | 4.043         | 15500     | 25.3 min  |
| `small_32k` | small | 23.4M        | 49.38     | 124.82  | 3.976         |  6500     | 29.2 min  |
| `large_32k` | large | 42.3M        | 43.41     | 124.71  | 3.975         |  5000     | 42.4 min  |

## Vocab axis — model fixed at `small`

How vocabulary size changes the model. Perplexity is **not** comparable across
this column (a token means a different amount of text under each vocab) — so
compare **bits-per-character**.

| Run         | Tokenizer | Total params | Val ppl | Val bits/char | Best step | Wall time |
|-------------|-----------|--------------|---------|---------------|-----------|-----------|
| `small_8k`  | vocab_8k  | 14.0M        |  52.55  | **3.879**     | 13000     | 32.6 min  |
| `small_16k` | vocab_16k | 17.1M        |  82.74  |   3.914       |  9000     | 28.5 min  |
| `small_32k` | vocab_32k | 23.4M        | 124.82  |   3.976       |  6500     | 29.2 min  |

(`small_32k` is the shared cell — it appears in both tables.)

## Context axis — `small`, vocab 32k

How context length helps. Every run early-stops at its own val minimum, so the
comparison is at each run's own peak.

| Run         | block | micro × accum | Train ppl | Val ppl | Val bits/char | Best step | Wall time |
|-------------|-------|---------------|-----------|---------|---------------|-----------|-----------|
| `ctx_256`   |  256  | 32 × 1        | 51.16     | 132.45  | 4.024         | 12000     |  22.5 min |
| `small_32k` |  512  | 32 × 1        | 49.38     | 124.82  | 3.976         |  6500     |  29.2 min |
| `ctx_1024`  | 1024  | 16 × 2        | 53.71     | 127.72  | **3.995**     |  4000     |  42.3 min |
| `ctx_2048`  | 2048  |  4 × 8        | 46.75     | 129.52  | 4.006         |  3500     |  84.7 min |
| `ctx_4096`  | 4096  |  2 × 16       | 48.91     | 140.88  | 4.075         |  2700     | 203.6 min |

## Observations

### Scale axis — early stopping erased the overfitting headline

- **Val perplexity is now flat across scale.** 135.5 (tiny) → 124.8 (small) →
  124.7 (large). The small → large step is **0.1 ppl** — within noise. The old
  fixed-budget sweep showed a 30-point train→val widening that screamed
  overfitting; early stopping makes that gap go away. The large model halts
  at step 5000 (the earliest of the three) because its val loss flattens
  fastest — exactly the moment a fixed-step run would have *kept going and
  overfit*.
- **The savant gave up early.** `large_32k` stopped at step 5000, `small_32k`
  at 6500, `tiny_32k` at 15500 — bigger models reach their personal best
  faster, then plateau. Early stopping keeps each at *its* peak; running them
  longer at a constant LR would only have hurt.
- **Cost.** Large still costs ~40% more wall-time than small for no
  generalisation gain. The `small` size is the right one here, but for a
  different reason than before — not because `large` overfits, but because
  `large` plateaus at the same val ppl with more parameters.

### Vocab axis — now the dominant axis

- **Vocab 8k wins by a wide margin.** Val ppl: 52.55 / 82.74 / 124.82 for
  8k / 16k / 32k. Per token, the smaller vocab is **2.4×** less surprised
  than the larger one. This collapses to a smaller but real gap on the fair
  metric: bits-per-character **3.879 / 3.914 / 3.976** — a 2.5% reduction.
- **The previous sweep (fixed-token-budget) hid this.** Under the old design
  every vocab ran for exactly 8000 steps and landed at 3.918 / 3.904 / 3.932
  BPC — flat. Under early stopping each tokenizer takes a different number of
  steps to converge — 8k goes to step **13000**, 32k stops at **6500** — and
  the small vocab pulls ahead. The fixed step count masked the difference;
  letting each one run to its own optimum exposed it.
- **Why.** The 32k embedding table is 12.8M params, 55% of the `small_32k`
  model, and 6M training tokens cannot fill it — many of the 32k tokens stay
  effectively untrained. The 8k tokenizer spends 3.1M on its embedding (22% of
  `small_8k`) and puts the rest of the budget into the transformer. **For a
  corpus this small, the embedding table is the bottleneck.**

### Context axis — flat under early stopping

- **Bits-per-character is essentially constant** across 256–4096: 4.024 / 3.976
  / 3.995 / 4.006 / 4.075. The 4096 run is only 2.5% worse than the 512
  baseline, vs **21% worse** under the old fixed-token-budget design.
- **So the old "long context hurts" result was a budgeting artefact.** When
  every run is allowed to train until its val stalls, the wider windows
  catch up. `ctx_1024` lands marginally below `small_32k` (3.995 vs 3.976) —
  noise-level, not a real win, but no longer the regression the old design
  reported.
- **Cost still compounds the verdict.** Wall time: 22 / 29 / 42 / 85 / **204**
  min — the 4096 run costs 7× the 512 baseline for the same val BPC. On a
  corpus this small, more context is not worth more time; the lesson stands,
  just for a different reason (compute, not undertraining).

### The corpus is still the bottleneck

The eight non-tiny runs land in **3.879–4.075 bits/char** — a 5% spread across
4× variation in vocab, 16× in context, and 4× in parameters. Only the smallest
tokenizer at the standard scale (`small_8k`, 3.879) materially lowers it. The
ceiling has moved a little — from ~3.9 (old) to ~3.88 (new) — but the shape is
unchanged: a 6M-token corpus has a hard floor that none of the three knobs
crosses by much. Continued-pretraining a larger pretrained model (Track B) is
what changes that.

## Failures / surprises

- No run crashed; all nine completed at exit 0 (early-stopped).
- _Surprise:_ on a tokenizer-fair metric, **the vocab axis is the only one
  that materially matters here**. The fixed-token-budget design hid this
  result for half the sweep; early stopping made it visible. The 32k vocab
  carried a 12.8M embedding table that 6M tokens cannot fill — undertrained
  embeddings cost val ppl that bits-per-character only partly normalises out.
- _Surprise:_ the old design's headline result — `large_32k` *overfits past*
  `small_32k` — does not survive early stopping. Under the new methodology
  large and small land at the **same** val perplexity (124.7 vs 124.8) and
  large simply stops earlier. The overfitting was an artefact of training
  every run to a fixed step count past the small model's natural stopping
  point.
- _Methodology note:_ the early-stopping patience (10 evals without
  improvement) is generous; reducing it would shorten the long runs
  (`ctx_4096` ran 203 min mostly waiting out patience after the best-val
  step at 2700) at the cost of a small risk of premature halt on a flat
  plateau. The current setting errs toward the safer side, which costs
  compute but not quality.

## Zero-shot Track B baseline — `qwen3-0.6b-base`

Stage 02 also runs the same probe set against the **un-tuned** Qwen3-0.6B-Base
to fix a reference point before continued pretraining. The cloze BPC is
directly comparable with the Track A cloze numbers in
[`FIELD_REPORT.md`](FIELD_REPORT.md) (Station 3). Reproduce with

```bash
.venv/bin/python 02_pretrain/eval_probes_qwen.py
```

### Cloze, bits-per-character (6 items, lower = better)

Comparing Qwen against three Track A anchors: the cloze winner (small-N
artefact, `tiny_32k` — see FIELD_REPORT footnote), the recommended candidate
(`small_8k`), and the cloze last-place (`ctx_4096`).

| Run                              |  c1<br>阿米娅 | c2<br>主要问题 | c3<br>保持全神贯注 | c4<br>公开领导人 | c5<br>感染者 | c6<br>制药公司 | **MEAN** |
|----------------------------------|------:|------:|------:|------:|------:|------:|---------:|
| `tiny_32k`  (Track A cloze #1¹)  | 4.421 | 3.024 | 4.980 | 5.676 | 2.175 | 0.781 | **3.509** |
| `small_8k`  (Track A recommended)| 3.482 | 3.653 | 4.686 | 6.076 | 1.966 | 1.985 | **3.641** |
| `ctx_4096`  (Track A cloze last) | 5.732 | 3.886 | 6.609 | 6.835 | 2.887 | 1.625 | **4.596** |
| `qwen3-0.6b-base` (zero-shot)    | **7.271** | 4.955 | **2.292** | 5.281 | 4.622 | 3.014 | **4.572** |

¹ `tiny_32k` won the 6-item cloze on a small-N artefact (over-trained on
exactly the high-frequency tokens these prompts target); on held-out val it
is *last*. The FIELD_REPORT Station 3 footnote has the full explanation.

**Read this row-by-row, not just on the mean.** Qwen's mean (4.572) lands
between `ctx_2048` and `ctx_4096` — worse than 8 of 9 Track A runs. But the
per-cell shape tells a different story: Qwen's surprise is **bimodal**, not
flat. Compare to `small_8k`:

| Cell | gold      | small_8k | Qwen  | Δ (Qwen − A) | what the cell tests |
|------|-----------|---------:|------:|-------------:|---------------------|
| c1   | 阿米娅       | 3.482 | **7.271** | **+3.79** | a corpus-specific name after a corpus-specific title |
| c2   | 主要问题      | 3.653 | 4.955 | +1.30 | corpus-specific paraphrase |
| c5   | 感染者       | 1.966 | 4.622 | +2.66 | corpus-specific term Qwen knows in COVID sense |
| c4   | 公开领导人 | 6.076 | **5.281** | −0.80 | generic Chinese noun — Qwen's prior wins |
| c6   | 制药公司      | 1.985 | 3.014 | +1.03 | generic noun after corpus-specific prefix |
| c3   | 保持全神贯注  | 4.686 | **2.292** | **−2.39** | high-frequency Chinese phrase — Qwen's prior wins big |

The Track A runs have **flat** per-cell surprise (every cell within ±2.5 of the
row mean, all in the same regime). Qwen has **spiky** per-cell surprise — c1
to c3 spans 5 bits/char. Same mean BPC, different shape, and the shape is
exactly "strong priors that happen to be wrong on this corpus".

### Why this is the right reference point for Stage 03

The Track A runs are 11.7M – 42.3M params trained for 22 – 204 minutes on 6M
tokens. Qwen3-0.6B-Base is 596M params trained by Alibaba on trillions of
tokens of the open web. On Arknights-specific cloze cells, the latter is
**worse**. Stage 03 (continued pretraining) is the experiment of converting
some of Qwen's 596M-param world prior into Arknights-specific knowledge while
preserving the Q&A register Qwen has and Track A lacks (see
`FIELD_REPORT.md` *Visiting consultant* appendix).

The expected Stage 03 trajectory, against this baseline:

- `c1`, `c4`: should fall by 2–4 bits as Qwen learns Amiya's title.
- `c3`, `c6`: should stay near zero-shot levels — already-known Chinese phrases.
- Q&A: the **register** stays, the **facts** rewrite.

The number to beat on cloze mean is `small_8k`'s **3.879 val bits/char** /
**3.641 cloze mean**. If a CPT'd Qwen lands above that on cloze, the 596M
params bought nothing and the experiment is a no.
