# Stage 02 — Pretrain: a from-scratch language model

The tokenizer from Stage 01 can *read* — it chops text into a fixed menu of
tokens. This stage builds the thing that *writes*: a small **language model**
that, given some tokens, predicts the next one. Stack enough of those
predictions and the model is generating text.

Track A builds it **from scratch** — a hand-rolled GPT in `lib/model.py`, no
`transformers` library. Every tensor moves in the open. The tensor ops follow
the [Stanford CS336](https://stanford-cs336.github.io/) convention: named-axis
`einops` (`rearrange` / `einsum`) instead of `.view()` / `.transpose()`, so a
reshape reads as the equation it is.

> Status: **all three axes complete** (9 runs). Result tables are in
> [`docs/RESULTS.md`](docs/RESULTS.md); the inference probes are written up,
> in-universe, in [`docs/FIELD_REPORT.md`](docs/FIELD_REPORT.md).

The corpus is tiny — the Stage 00 `train` split is **11.66M characters**, only
~6–9M tokens depending on the vocabulary. That is far too little to train a
good language model, and that is *fine*: this stage is not chasing a good
model. It is an **experiment** about what happens when you change three knobs —
the model size, the tokenizer's vocabulary, and the context length — on a
corpus this small. The lesson is the shape of the curves, not the final number.

---

## 1. The experiment — a three-axis sweep

Three things are worth varying. We vary them one at a time, holding the rest
fixed — a full grid would tangle the effects together; controlling one variable
at a time is the point.

- **Scale axis** — fix tokenizer 32k and block 512, train all three model
  sizes. *Teaches:* how model size trades against a fixed, small dataset.
- **Vocab axis** — fix `small` and block 512, train on the 8k / 16k / 32k
  tokenizers. *Teaches:* how vocabulary size changes the model.
- **Context axis** — fix `small` and tokenizer 32k, train at block size
  256 / 512 / 1024 / 2048 / 4096. *Teaches:* how much context length helps,
  against its quadratic memory cost.

That is **9 runs** — `small_32k` (scale `small`, vocab 32k, block 512) is the
shared centre of all three axes. Configs are `configs/<run>.yaml`:

| Axis    | Run         | What varies     | Total params |
|---------|-------------|-----------------|--------------|
| scale   | `tiny_32k`  | scale tiny      | 11.7M |
| scale   | `small_32k` | *shared centre* | 23.4M |
| scale   | `large_32k` | scale large     | 42.3M |
| vocab   | `small_8k`  | vocab 8k        | 14.0M |
| vocab   | `small_16k` | vocab 16k       | 17.1M |
| context | `ctx_256`   | block 256       | 23.3M |
| context | `ctx_1024`  | block 1024      | 23.6M |
| context | `ctx_2048`  | block 2048      | 24.0M |
| context | `ctx_4096`  | block 4096      | 24.8M |

Every run sees the **same token budget** — `effective_batch × block_size ×
max_steps` ≈ 131M tokens — so the comparison is fair. The context axis trades
those factors off (a longer window → smaller batch → fewer steps) while holding
the product *and* the effective batch constant; **gradient accumulation** lets
a long-context run keep an effective batch of 32 even when only a micro-batch
of 1–4 sequences fits in memory.

**What to watch.** On a corpus this small, the bigger models drive *train* loss
down while *validation* perplexity stalls or climbs — that gap is
**overfitting**, and making it visible is the headline result of the scale
axis. On the context axis, watch whether a longer window actually lowers
perplexity or just burns memory. `train.py` keeps the checkpoint with the best
val loss, so each run records its own best moment even if it overfits after.

## 2. The model (`lib/model.py`)

A GPT-style decoder-only transformer, classic GPT-2 shape:

- Token embedding + **learned absolute** positional embedding.
- Pre-norm transformer blocks: `LayerNorm → attention → +residual`, then
  `LayerNorm → MLP → +residual`.
- Multi-head **causal** self-attention (a position attends only to the past).
- **Weight-tied** head: the output projection reuses the embedding matrix.
- GELU MLP with a 4× hidden expansion.

**A note on document packing — a deliberate simplification.** The corpus is
concatenated into one flat token stream, each document bracketed by
`<|bos|>` … `<|eos|>`, and `get_batch` samples training windows at random
offsets — so a `block_size`-token window can straddle a document boundary.
The causal mask only forbids attending to the *future*; it does **not** reset
at `<|eos|>`. So when a window spans `… docA <|eos|> <|bos|> docB …`, tokens in
docB *can* attend back into docA. We do not mask that out — instead the model
is expected to learn, from the `<|bos|>` / `<|eos|>` markers (which are real,
predicted tokens), that content before a boundary is no longer relevant. This
is the GPT-2 / nanoGPT convention. The stricter alternative — **intra-document
(block-diagonal) attention masking**, which resets attention at every `<|eos|>`
— is left as a possible extension; its value grows with longer context windows,
which straddle more boundaries.

Three scale presets (`lib/model.SCALE_PRESETS`) — each fixes only depth and
width:

| Scale | layers | d_model | heads | Transformer params |
|-------|--------|---------|-------|--------------------|
| tiny  | 4 | 256 | 4 | 3.2M  |
| small | 6 | 384 | 6 | 10.6M |
| large | 8 | 512 | 8 | 25.2M |

**The embedding table is separate — and it is big.** The transformer column
above is fixed, but the embedding table is `vocab_size × d_model`, set by the
tokenizer. At a 32k vocab it can *outweigh the whole transformer*:

| Scale | params @ 8k | @ 16k | @ 32k |
|-------|-------------|-------|-------|
| tiny  | 5.4M  | 7.5M  | 11.7M (73% embedding) |
| small | 14.0M | 17.1M | 23.4M (55% embedding) |
| large | 29.7M | 33.9M | 42.3M (40% embedding) |

So "model scale" really means the *transformer* column; total size depends on
the vocabulary. That coupling is exactly what the vocab axis probes — and a
real lesson about small from-scratch models: much of the parameter budget can
go to an embedding table that 10M tokens struggle to fill.

## 3. Metrics

- **Perplexity** `exp(loss)` — the model's average "surprise" per token, lower
  is better. Comparable *within* the scale axis (same 32k vocab throughout).
- **Bits-per-character** — perplexity is **not** comparable across the vocab
  axis: a token covers a different amount of text under each tokenizer.
  Bits-per-character normalises that out, so it is the fair metric for the
  vocab sweep. `train.py` reports both.

Later stages re-score the model on the nested test sets T0 ⊂ T1 ⊂ T2 from
Stage 00 and against the shared evaluation set — that is where Track A finally
meets Track B.

## 4. How to run

```bash
# one-time: virtualenv + deps (Ubuntu's system Python is PEP-668 locked)
python3 -m venv .venv
.venv/bin/pip install -r 02_pretrain/requirements.txt

# a <2-minute end-to-end check — tiny model, a few files
.venv/bin/python 02_pretrain/train.py --config 02_pretrain/configs/small_32k.yaml --smoke-test

# a real run (use tmux — the full sweep is hours, not minutes)
.venv/bin/python 02_pretrain/train.py --config 02_pretrain/configs/small_32k.yaml

# generate from a checkpoint
.venv/bin/python 02_pretrain/sample.py --name small_32k --prompt '罗德岛'

# probe every trained run — generation, temperature dial, cloze, Q&A
.venv/bin/python 02_pretrain/eval_probes.py
```

The probe set `eval_probes.py` reads — free-generation prompts, cloze
sentences, and questions — lives in the editable `probes.txt`, not the code.

The token stream is encoded once and cached under `data/tokenized/<tokenizer>/`;
checkpoints land in `data/checkpoints/<run>/`. Both are git-ignored.

## 5. A note for the learner — EXERCISE blocks

Like the tokenizer, the model code carries **`EXERCISE` markers** around its
pedagogically central regions (see [`01_tokenizer/README.md`](../01_tokenizer/README.md)
§7 for the convention). The committed code is the working reference; in
learning mode you **delete the body between the markers and rewrite it** from
the `Concept / Given / Produce / Steps` spec in the block header.

| Block | File | What you implement |
|-------|------|--------------------|
| `attention`         | `lib/model.py` | scaled-dot-product causal self-attention |
| `transformer-block` | `lib/model.py` | the pre-norm residual wiring |
| `gpt-forward`       | `lib/model.py` | embed → blocks → logits → loss |
| `get-batch`         | `02_pretrain/train.py` | sampling (input, target) windows |
| `train-step`        | `02_pretrain/train.py` | gradient accumulation + the optimiser step |
| `sample-loop`       | `02_pretrain/sample.py` | autoregressive decoding |
| `cloze-nll`         | `02_pretrain/eval_probes.py` | scoring a known answer in bits-per-char |

```python
# === EXERCISE START: attention ========================================
# Concept: ...
# Given:   ...
# Produce: ...
# Steps:   1) ...
# Learning mode: delete the body below and rewrite it from the spec ...
# ----------------------------------------------------------------------
    <reference implementation — delete and redo this>
# === EXERCISE END: attention ==========================================
```

`git diff` shows your version against the reference; `--smoke-test` is your
grader — if it runs end-to-end and the loss falls, your code works.

## Files

```
02_pretrain/
  README.md            this guide
  train.py             training loop          (EXERCISE: get-batch, train-step)
  sample.py            generate from a checkpoint   (EXERCISE: sample-loop)
  eval_probes.py       probe trained runs           (EXERCISE: cloze-nll)
  probes.txt           the probe set — prompts, cloze items, questions
  requirements.txt     torch, einops, jaxtyping, numpy, pyyaml
  configs/             one YAML per run — clone, never edit in place
  docs/RESULTS.md      perplexity tables — the three-axis sweep
  docs/FIELD_REPORT.md inference probes, written up in-universe (EN + 中文)
lib/model.py           the GPT   (EXERCISE: attention, transformer-block, gpt-forward)
```
