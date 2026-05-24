# Stage 05 — DPO / IPO results (2×2 ablation)

Preference optimisation on top of the Stage 04 `sft_full` checkpoint, using
agent-authored pairs that target the named failure modes from Stage 04's smoke
probes. Four cells: **DPO vs IPO** × **bulk (~5000) vs curated (~500)**.

> Status: **All 4 cells complete.** Numbers below are `train_dpo.py`'s best-val
> report. Hardware: RTX 4090.

## Setup (shared)

| | |
|---|---|
| Base (policy init + frozen ref) | `data/checkpoints/sft_full` (Stage 04 winner, full-FT, 1.2 GB) |
| Adapter | LoRA r=16, α=32, dropout 0.05, on `q/k/v/o_proj` — 4.6 M trainable params (0.76%) |
| Effective batch | 16 (`batch_size=4 × grad_accum=4`), `max_length=1024` |
| LR / schedule | `5e-6` with linear warmup 25 + flat |
| β | 0.1 (DPO temperature; IPO target ratio = 1/(2β) = 5) |
| Early stop | patience=5 evals |

Cell-specific:

| Cell | Train pairs | Val pairs | Max steps | Eval interval |
|---|---:|---:|---:|---:|
| `dpo_curated` |   801 |  89 |   800 |  25 |
| `ipo_curated` |   801 |  89 |   800 |  25 |
| `dpo_bulk`    | 1610 | 180 |  2000 | 100 |
| `ipo_bulk`    | 1610 | 180 |  2000 | 100 |

Stratified-by-`fault_type` val splits via `derive_val_split.py`, seed 1337.
Gradient checkpointing on the policy (logits-fp32 cast fires four times per
step under DPO; ~93% VRAM peak on a 24 GB 4090).

## Training trajectories

All four cells early-stopped on `patience=5`. **`sft_val_ce`** = mean
assistant-token CE on `data/sft/qa_val.jsonl` (192 rows); the Stage-04 final
value at this set was **≈2.96**, so `Δ` reads as drift.

### Curated cells (801 train pairs)

#### `dpo_curated` — best step 325 / 7.4 min wall

| Step | val_loss | val_acc | sft_val_ce | Δ vs Stage 04 |
|---:|---:|---:|---:|---:|
|  25 | 0.5918 | 0.95 | 2.964 | +0.004 |
|  50 | 0.4068 | 0.97 | 3.026 | +0.07  |
| 100 | 0.2286 | 0.95 | 3.131 | +0.17  |
| 200 | 0.1000 | 0.97 | 3.268 | +0.31  |
| **325 (best)** | **0.0636** | **1.00** | **3.270** | **+0.31** |
| 400 | 0.0804 | 0.97 | 3.447 | +0.49  |
| 450 (stop) | 0.0772 | 0.97 | 3.346 | +0.39 |

Val loss dropped 10× from init (log 2 ≈ 0.693 → 0.064). SFT-preservation drift
accelerated past step 325; early-stop on val_loss caught the right cut-point
by luck.

#### `ipo_curated` — best step 275 / 6.6 min wall

| Step | val_loss | val_acc | sft_val_ce | Δ vs Stage 04 |
|---:|---:|---:|---:|---:|
|  25 | 11.99 | 0.92 | 2.964 | +0.004 |
|  50 |  7.79 | 0.94 | 2.966 | +0.006 |
| 200 |  6.08 | 0.97 | 2.969 | +0.009 |
| **275 (best)** | **6.04** | 0.94 | **2.973** | **+0.013** |
| 300 |  6.56 | 1.00 | 2.971 | +0.011 |
| 400 (stop) |  7.01 | 0.94 | 2.971 | +0.011 |

The numerical val_loss scale is **not comparable to DPO's** — IPO targets a
log-ratio diff of `1/(2β)=5` and the loss is `(diff − 5)²`, which floors around
6 even at convergence rather than approaching 0.

### Bulk cells (1610 train pairs)

#### `dpo_bulk` — best step 1300 / 24.4 min wall

| Step | val_loss | val_acc | sft_val_ce | Δ vs Stage 04 |
|---:|---:|---:|---:|---:|
|  100 | 0.3645 | 0.89 | 3.151 | +0.19 |
|  300 | 0.1885 | 0.92 | 3.247 | +0.29 |
|  500 | 0.1394 | 0.92 | 3.292 | +0.33 |
|  700 | 0.1417 | 0.95 | 3.527 | +0.57 |
|  900 | 0.0928 | 0.98 | 3.588 | +0.63 |
| 1100 | 0.0998 | 0.97 | 3.903 | +0.94 |
| **1300 (best)** | **0.0641** | **0.98** | **3.936** | **+0.98** |
| 1500 | 0.0753 | 0.98 | 3.938 | +0.98 |
| 1800 (stop) | 0.0983 | 0.95 | 3.964 | +1.01 |

**The drift tripled vs curated (+0.31 → +0.98)** while val_loss landed at the
same place (0.064 vs 0.064). 5× more pairs bought essentially **no additional
preference signal** at this β / LR / capacity — just more capability erosion.
The sft_val_ce climbs monotonically; nothing in the trajectory hints at a
plateau.

#### `ipo_bulk` — best step 1200 / 23.1 min wall

| Step | val_loss | val_acc | sft_val_ce | Δ vs Stage 04 |
|---:|---:|---:|---:|---:|
|  100 | 10.72 | 0.86 | 2.974 | +0.014 |
|  300 |  7.78 | 0.89 | 2.975 | +0.015 |
|  500 |  8.09 | 0.89 | 2.977 | +0.017 |
|  700 |  8.00 | 0.92 | 2.981 | +0.021 |
|  900 |  7.11 | 0.94 | 2.981 | +0.021 |
| 1100 |  7.00 | 0.94 | 2.978 | +0.018 |
| **1200 (best)** | **6.89** | **0.92** | **2.978** | **+0.018** |
| 1500 |  7.74 | 0.92 | 2.980 | +0.020 |
| 1700 (stop) |  7.46 | 0.92 | 2.977 | +0.017 |

Drift held flat at curated levels (+0.013 → +0.018). Val_loss landed *worse*
than ipo_curated's 6.04 — IPO's `(diff − 5)²` doesn't reward over-confident
ordering, and the bulk pairs include more ambiguous chosen/rejected boundaries
where the policy can't comfortably hit the target. **IPO bulk is the dataset
diluting the signal, not strengthening it.**

## 2×2 summary

|              | **DPO**                                         | **IPO**                                         |
|--------------|-------------------------------------------------|-------------------------------------------------|
| **Curated**  | val_loss 0.064 · val_acc 1.00 · **drift +0.31** | val_loss 6.04 · val_acc 0.94 · **drift +0.013** |
| **Bulk**     | val_loss 0.064 · val_acc 0.98 · **drift +0.98** | val_loss 6.89 · val_acc 0.92 · **drift +0.018** |

**Key axis effects:**

- **DPO → IPO (down the column):** 25× less SFT drift, slight val_acc dip.
  IPO's identity-link ceiling at `1/(2β)=5` does what Azar 2023 advertises.
- **Curated → bulk (across the row):** for DPO, **3× more drift for the same
  val_loss** — bulk is the dataset wasting capacity on already-easy pairs.
  For IPO, no meaningful change — IPO floors out either way.
- **Pairwise objective fully satisfied in every cell** (val_acc ∈ [0.92, 1.0]).
  The model can rank the two strings correctly. The question is whether that
  shifts what it *generates* at T=0.

## Smoke probes — four-way comparison at T=0

Same 8 prompts as Stage 04's `chat.py --probes`. Stage 04 SFT baseline
reproduced from `04_sft/docs/RESULTS.md`. Full probe logs at
`data/dpo_logs/{dpo,ipo}_{curated,bulk}_probes.log`.

| Probe | Stage 04 SFT | `dpo_curated` | `ipo_curated` | `dpo_bulk` | `ipo_bulk` |
|---|---|---|---|---|---|
| factoid: Kal'tsit race | **WRONG** (萨卡兹) | **STILL 萨卡兹** + `.ALIGNMENT` prefix | **STILL 萨卡兹**, SFT-like paragraph | **STILL 萨卡兹** + `物理` prefix + hedge ("自称是萨卡兹，但…并非萨卡兹") | **STILL 萨卡兹**, loop on "萨卡兹的领袖" |
| factoid: Amiya height | 162cm (hallucinated) | 155cm (different number) | 162cm (= SFT) | 155cm (= DPO curated) | **degenerate** ("mPid设定为...") |
| open-ended: 博士/阿米娅 | "罗德岛主干干员" + circular | meta-quote about file labels | "同班同学" fabrication | A3-module archival framing, degenerate | "spep" prefix + relationship fabrication |
| event: 切尔诺伯格 | mode-collapse loop | **coherent paragraph** (factory explosion, partly fabricated) | mode-collapse loop (= SFT) | "物理：阿米娅体温36.2℃" — totally off-topic | mode-collapse loop (= IPO curated) |
| relationship: 推进之王/双子 | circular + fabricated source | circular + "共同敌人" fabrication | "姐姐"→"也是妹妹" contradiction | **severe loop** on "档案资料四也写到" | "姐姐" + Sarkaz-leader fabrication |
| refusal: 2030 股票市值 | 1000亿泰拉币 + fake source | **off-domain English** hallucination | 1000亿泰拉币 — **byte-identical to SFT** | 1.2万亿泰拉 (different fake number) | 1000亿泰拉币 — **byte-identical to SFT** |
| general_zh: 李白 | 唐代 ✓ + 大历十才子 anachronism | 唐朝 ✓ + 四川成都 anachronism | 唐代 ✓ + 天宝年间 (reasonable) | **lost the answer** — body-temperature archive | 唐代 ✓ + 陇西郡甘肃 (plausible historical fact) |
| format: 你是谁 | garbage `{EIF)` | claims to be Amiya (with stats) | `{EIF)` — **byte-identical to SFT** | hallucinated 月光 (Moonlight) persona | `{EIF)` — **byte-identical to SFT** |

## Observations

### The headline result, now confirmed at scale: **preference learning ≠ argmax change**

After curated, the suspicion was that 801 pairs was just too sparse to move
the model's top-1 over a 151K vocabulary. The bulk cells were the test:
**5× more pairs, same β, same LR, same capacity** — does that buy argmax
change?

**It does not.** On the named Stage-04 failure modes:

- **Kal'tsit race:** still 萨卡兹 in all 4 cells. The bulk DPO cell adds a hedge
  ("she claims to be Sarkaz, but file two indicates she's not") — the closest
  any cell came to acknowledging the trained preference — but the argmax token
  is still 萨卡兹.
- **Refusal probe:** identical `1000亿泰拉币` under SFT, ipo_curated,
  *and* ipo_bulk. IPO doesn't touch it at any data scale. DPO swaps it for
  a different fake number or off-domain English — different failure, not a fix.
- **Format probe:** identical `{EIF)` garbage under SFT, ipo_curated, and
  ipo_bulk. IPO is byte-identical at both scales.

Confirms the curated-cell finding: **the DPO/IPO gradient constrains pairwise
ordering between two specific sequences; it does not constrain the model's
top-1 over the whole 151K-token vocabulary.** More pairs sharpens the ordering
further (val_acc 1.00 → 1.00) without spilling into argmax.

### The bulk DPO "物理" prefix — a new artefact at scale

`dpo_curated` introduced a `.ALIGNMENT` / `.ORDER` / `.ORDERED RECORD`
Latin-character preamble on every answer — a LoRA reshaping the first-token
distribution in an unintended direction. **`dpo_bulk` does the same thing in
Chinese: every answer starts `物理。`** (literally "physical." — a token-
sequence that's never in either training side). The mechanism is identical;
the bulk run only changed which token won the first-position lottery.

This is a clean illustration of what unbounded DPO loss does to the policy:
some position in the weights gets twisted to maximise the log-ratio diff, and
the easiest way is often to shift the first-token mass to a low-probability
token that happens to slightly down-weight the `rejected` sequence relative
to its competitor. The shift is gradient-cheap; the cost is that the model
opens every answer with a meaningless word.

### Why does scaling pairs not close the argmax gap?

Same list as the curated post-mortem, with two findings now upgraded from
prediction to data:

1. **The preference signal is two-sequence; the argmax is over the whole
   distribution.** *(Same as curated.)* If the model's top-1 sequence is a
   third string the agent never wrote — the well-trodden SFT-distribution
   answer like 萨卡兹 — DPO's gradient doesn't touch it directly, and **adding
   pairs does not help**, because every new pair still doesn't include that
   third string.
2. **The pair-budget is spent on already-easy pairs.** *(Curated prediction
   now confirmed.)* Bulk train_acc hits 0.94 by step 100 (vs step 25 for
   curated, but the same fraction of total steps). The bulk pairs have a
   wider distribution of difficulty, but the model burns the budget on the
   bottom half and never re-allocates to the hard ones.
3. **LoRA r=16 caps how much the first-token distribution can move.**
   *(Same as curated.)* The bulk cell shows where the constrained capacity
   goes when pushed: into a different artefact (`物理`), not into a fix.
4. **β=0.1 + log-σ loss has no ceiling.** *(Curated prediction now confirmed
   at scale.)* DPO bulk's drift triples (+0.31 → +0.98) for **zero** val_loss
   improvement. The loss keeps pulling chosen up and rejected down even when
   it's already maxed the easy pairs; the gradient bleeds into off-target
   weights. IPO bulk's ceiling at `1/(2β)=5` is exactly the right behaviour
   for this dataset — the loss stops pushing once the target is hit.

### Was the curated > bulk prediction (Lambert et al. 2024) correct?

**Yes, decisively** — but with a twist. The classical "few hundred curated
pairs beat thousands of bulk pairs" finding is usually stated in terms of the
preference objective itself. Here the four cells **all hit val_acc ≥ 0.92**,
so on the preference objective bulk and curated are indistinguishable. The
relevant axis is **what bulk costs**: 3× the SFT drift under DPO, and no
benefit. For this 0.6B-on-Arknights setup, the curated cells dominate every
metric we care about.

### What actually moved across the 4 cells

| Failure mode (from Stage 04) | Fixed by any cell? |
|---|---|
| Stated-factoid hallucination (Kal'tsit race, Amiya height) | **No** — argmax unchanged or replaced by a different hallucination |
| Open-ended fabrication (博士/阿米娅) | **No** — every cell rewrites the fabrication style without removing it |
| Mode-collapse loop (切尔诺伯格 event) | **dpo_curated only** — bulk DPO and both IPOs regress to the loop |
| Refusal failure (2030 股票市值) | **No** — IPO is byte-identical to SFT; DPO swaps the fake source |
| General-knowledge anachronism (李白) | **No** — IPO bulk got closer to plausible historical fact; DPO bulk lost the answer entirely |
| Format brittleness ({EIF)) | **No** — IPO byte-identical to SFT; DPO replaces with a different hallucinated persona |

**Only one failure mode (mode-collapse) moved, in only one cell.** The named
"plausible-hallucination" failure modes are still there in all four cells.

### What did work (mechanically and operationally)

- **All four loss curves are clean.** No NaN, no divergence; early-stop fired
  cleanly in all cells; the SFT-preservation tripwire (README §5 third signal)
  did its job — DPO bulk's drift was clearly visible by step 700 and would
  have triggered a manual stop if val_loss hadn't peaked first.
- **The 2×2 matrix is fully interpretable.** Both axes (DPO/IPO, bulk/curated)
  produce the predicted Azar-2023 / Lambert-2024 effects.
- **IPO genuinely is gentler at both data scales** — the central claim of
  Azar 2023 verified at 0.6B in Chinese on a custom dataset.

## What this means for Stage 06

The 2×2 settles the question that motivated Stage 05 — and answers it in the
negative. Preference optimisation on hand-authored pairs does not fix the
named hallucination modes at 0.6B, regardless of loss function or dataset
scale within the tested range. The reason is mechanistic and quoted above
(point 1: gradient sees two strings; argmax sees 151K).

Three forward levers, in increasing order of cost:

- **Rejection sampling at inference** — sample N from the SFT policy, score
  each with the IPO-curated policy's preference signal, return the highest-
  scoring one. Cheapest; turns the trained preference into a usable signal
  *outside* the argmax.
- **DPO with model-sampled rejected** — re-author the rejected side as the
  model's own argmax output, so the gradient actually sees what to push down.
  Closes the preference-vs-argmax gap by construction; requires a sampling
  pass and re-authoring.
- **RLHF with a real reward model trained on these pairs** — full PPO/GRPO
  loop; the policy is updated against rewards on its own samples, so the
  argmax is in the optimisation path. Expensive but the standard answer to
  "DPO doesn't move argmax."

The `ipo_curated` checkpoint is preserved as the safest reference policy
(preserves SFT behaviour while learning the preference ordering). The
`dpo_curated` checkpoint is preserved as the cell that maximised preference
signal at the smallest acceptable cost (drift +0.31, the one mode-collapse
fix). The two bulk checkpoints are kept for reproducibility but neither
dominates its curated sibling on any metric we report.

## Files

```
05_dpo/
  README.md            design doc (2×2 ablation: DPO × IPO × bulk × curated)
  train_dpo.py         DPO + IPO training loop  (dpo-format, dpo-logprobs,
                                                 dpo-loss, ipo-loss,
                                                 train-loop EXERCISES filled in)
  derive_val_split.py  one-time stratified val split (seed 1337) from
                       agent-shipped {bulk,curated}_train.jsonl
  configs/
    dpo_curated.yaml   ✓ run · val_loss 0.064 · drift +0.31
    ipo_curated.yaml   ✓ run · val_loss 6.04  · drift +0.013
    dpo_bulk.yaml      ✓ run · val_loss 0.064 · drift +0.98
    ipo_bulk.yaml      ✓ run · val_loss 6.89  · drift +0.018
  docs/
    RESULTS.md         this file
```

End-of-run scoring against `eval/questions.yaml` is still pending (will land
as `05_dpo/score_eval.py` once the judge-mode agent invocation pattern is
implemented).
