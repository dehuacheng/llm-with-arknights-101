# Stage 05 — DPO / IPO results (curated cell)

Preference optimisation on top of the Stage 04 `sft_full` checkpoint, using
the agent-curated preference pairs that target the named failure modes from
Stage 04's smoke probes.

> Status: **`dpo_curated` and `ipo_curated` complete.** Bulk cells
> (`dpo_bulk`, `ipo_bulk`) scaffolded; not yet run — see §7. Numbers are
> `train_dpo.py`'s best-val report. Hardware: RTX 4090.

## Setup

| | |
|---|---|
| Base (policy init + frozen ref) | `data/checkpoints/sft_full` (Stage 04 winner, full-FT, 1.2 GB) |
| Train data | `data/dpo/curated_train.jsonl` (801 pairs) |
| Val data | `data/dpo/curated_val.jsonl` (89 pairs, **stratified by `fault_type`** via `derive_val_split.py`, seed 1337) |
| Source | Agent-generated 890-row pair JSONL; 10% held out per fault_type so `wrong_date` (n=11) and `should_refuse` (n=48) keep eval signal |
| Adapter | LoRA r=16, α=32, dropout 0.05, on `q/k/v/o_proj` — 4.6 M trainable params (0.76%) |

Effective batch = 16 (`batch_size=4 × grad_accum=4`), `max_length=1024`,
LR `5e-6` with linear warmup 25 + flat, β=0.1. Gradient checkpointing on
the policy (logits-fp32 cast fires four times per step under DPO; ~93%
VRAM peak on a 24 GB 4090).

## Training trajectories

Both cells early-stopped on `patience=5` after the val-loss plateau.

### `dpo_curated`

| Step | val_loss | val_acc | sft_val_ce | Δ vs Stage 04 |
|---:|---:|---:|---:|---:|
|  25 | 0.5918 | 0.95 | 2.964 | +0.004 |
|  50 | 0.4068 | 0.97 | 3.026 | +0.07 |
| 100 | 0.2286 | 0.95 | 3.131 | +0.17 |
| 200 | 0.1000 | 0.97 | 3.268 | +0.31 |
| **325 (best)** | **0.0636** | **1.00** | **3.270** | **+0.31** |
| 400 | 0.0804 | 0.97 | 3.447 | +0.49 |
| 450 (stop) | 0.0772 | 0.97 | 3.346 | +0.39 |

7.4 min wall; best at step 325. Val loss dropped 10× from init (log 2 ≈
0.693 → 0.064). **SFT-preservation tripwire reported +0.31 by best step,
+0.49 by 5 evals past — the drift accelerated after step 325**, and
early-stop on val_loss caught the right cut-point by luck. Adapter:
18 MB at `data/checkpoints/dpo_curated/`.

### `ipo_curated`

| Step | val_loss | val_acc | sft_val_ce | Δ vs Stage 04 |
|---:|---:|---:|---:|---:|
|  25 | 11.99 | 0.92 | 2.964 | +0.004 |
|  50 |  7.79 | 0.94 | 2.966 | +0.006 |
| 200 |  6.08 | 0.97 | 2.969 | +0.009 |
| **275 (best)** | **6.04** | 0.94 | **2.973** | **+0.013** |
| 300 |  6.56 | 1.00 | 2.971 | +0.011 |
| 400 (stop) |  7.01 | 0.94 | 2.971 | +0.011 |

6.6 min wall; best at step 275. The numerical val_loss scale is **not
comparable to DPO's** — IPO's target log-ratio diff is 1/(2β) = 5 and the
loss is `(diff - 5)²`, which floors around 6 even at convergence rather
than approaching 0. The right comparison is val_acc (both ~1.0 at best) and
SFT preservation (DPO +0.31 vs IPO +0.013 — IPO is ~25× gentler on the
policy). Adapter: same size, at `data/checkpoints/ipo_curated/`.

## Smoke probes — three-way comparison at T=0

Same 8 prompts as Stage 04's `chat.py --probes`. The Stage 04 SFT baseline
column is reproduced verbatim from `04_sft/docs/RESULTS.md`.

| Probe | Stage 04 SFT | `dpo_curated` | `ipo_curated` |
|---|---|---|---|
| factoid: Kal'tsit race | **WRONG** (萨卡兹) | **STILL 萨卡兹**, plus fake citations; `.ALIGNMENT` prefix | **STILL 萨卡兹**, fluent paragraph similar to SFT |
| factoid: Amiya height | 162cm (hallucinated) | 155cm (hallucinated, different number) | 162cm (same as SFT) |
| open-ended: 博士/阿米娅 | "罗德岛主干干员"+ circular self-quote | Incoherent meta-quote about file labels | "罗德岛干员"+ circular + new "同班同学" fabrication |
| event: 切尔诺伯格 | mode-collapse loop on 感染者难民 | **coherent paragraph** about factory explosion (partly fabricated) | mode-collapse loop (back to SFT behaviour) |
| relationship: 推进之王/双子 | circular + fabricated source | circular + "共同敌人" fabrication | "姐姐"→"也是妹妹" — single-answer contradiction |
| refusal: 2030 股票市值 | 1000亿泰拉币 + fake source | **off-domain English** Wikipedia hallucination | **1000亿泰拉币 + fake source — byte-identical to SFT** |
| general_zh: 李白 | 唐代 ✓ + 大历十才子 anachronism | 唐朝 ✓ + forced archival framing | 唐代 ✓ + 唐玄宗天宝年间 (reasonable) |
| format: 你是谁 | garbage token `{EIF)` | confidently claims to be Amiya (with stats) | **`{EIF)` — byte-identical to SFT** |

## Observations

### The headline result: **preference learning ≠ argmax change**

Both cells achieved val pair_acc ≈ **1.0** — i.e., on every held-out pair
the policy assigns higher log-prob to `chosen` than to `rejected`. The
DPO objective is fully satisfied; the IPO objective is at its target. And
yet on the named Stage-04 failure modes (Kal'tsit race, Amiya height,
refusal, mode collapse) the **argmax-decoded output is largely unchanged**.

Kal'tsit's race is the cleanest demonstration. The curated training data
contains the explicit pair:

```jsonc
{"chosen":  "凯尔希博士的种族是菲林（Feline）, 档案中明确登记。",
 "rejected": "凯尔希博士的种族是萨卡兹（Sarkaz）……"}
```

After DPO and IPO, `logπ_pol(菲林…) > logπ_pol(萨卡兹…)` on this prompt —
but the global argmax over the whole 151K-vocab token space at each
position **still produces a sequence beginning with 萨卡兹**, because the
DPO/IPO gradient only constrains the *ordering of two specific sequences*,
not the model's top-1 over everything else.

The gap between **"prefer chosen over rejected"** (DPO trains this) and
**"emit chosen at T=0"** (what the probes test) is the central finding.

### DPO vs IPO — the trade-off Azar 2023 named

- **DPO is aggressive.** SFT-preservation drift +0.31 absolute (11% rise
  in mean CE on the Stage-04 val set). Introduces visible artefacts:
  every answer is prefixed with `.ALIGNMENT` / `.ORDER` / `.ORDERED
  RECORD` (a Latin-character preamble that wasn't in either training side
  — the LoRA reshaping the first-token distribution in an unintended
  direction). Refusal failure switched modes from Chinese fabrication
  to off-domain English. The cure for one failure mode came with new ones.
- **IPO is conservative.** SFT-preservation drift +0.013 (essentially
  zero). On half the probes the output is byte-for-byte identical to
  Stage 04 (refusal probe: identical fake number; format probe: identical
  `{EIF)` garbage). On the other half the difference is subtle wording.
  No new artefacts; no register collapse.

This is the textbook IPO trade-off: the identity-link loss caps the
log-ratio diff at `1/(2β)=5` and stops pushing once that target is hit,
where DPO's log-σ loss has no natural ceiling and keeps pulling chosen
up / rejected down indefinitely.

For this project's goal — fixing named hallucinations without trashing
the broader capability — **IPO is the safer choice but neither cell
fixes the named hallucinations**. The improvement is asymmetric:

- **What moved:** the event probe (mode-collapse → coherent paragraph
  under DPO; unchanged under IPO).
- **What didn't move:** the stated-factoid hallucination (Kal'tsit) under
  either loss; the refusal failure under either loss; the format-probe
  garbage under IPO.

### Why does T=0 argmax fail to track DPO's gradient?

A small list of contributing factors, in decreasing order of likely impact:

1. **The preference signal is two-sequence, the argmax is over the whole
   distribution.** DPO never sees the model's actual top-1 generation; it
   only sees the two strings the agent shipped. If the model's
   highest-probability sequence is a *third* string (the well-trodden
   SFT-distribution answer like "萨卡兹"), DPO's gradient doesn't touch
   it directly.
2. **β=0.1 is permissive.** The "regularisation toward the reference"
   that β controls is loose at this value, but the loss saturates fast on
   easy pairs (val_acc 0.95 by step 25), so most of the gradient is spent
   on already-easy pairs rather than the hard ones where argmax could
   shift.
3. **LoRA rank 16 caps how much the first-token distribution can move.**
   A higher-rank or full-FT DPO might rewrite the argmax behaviour at the
   cost of much more SFT drift.
4. **890 curated pairs is small.** At one pair per failure-mode-instance,
   the model sees each "right answer" once or twice — sparse signal vs
   the 1,727 SFT rows whose patterns the policy memorised in Stage 04.

### What did work

- **SFT preservation behaved exactly as the README §5 tripwire predicted.**
  DPO drift was real and bounded; IPO drift was negligible. The signal
  caught the named failure mode #1 (refuse-everything collapse) wasn't
  triggered — pair_acc stayed ≥0.94 throughout both runs, no sign of
  pushing both sides down.
- **Loss curves were clean.** No NaN, no divergence, well-behaved early
  stop. The mechanical loop works; the loop is not where the failure is.
- **IPO genuinely is gentler.** 25× less SFT drift than DPO at the same
  β, same data, same step budget.
- **The mode-collapse failure mode is movable.** DPO did fix the 切尔诺伯格
  loop into a coherent paragraph. Pattern (4) from Stage 04's taxonomy
  is the only one that visibly responded to either preference loss.

### What this means for Stage 05 / Stage 06

The two unrun cells (`dpo_bulk`, `ipo_bulk`) will test whether **5×
more pairs** (1,610 vs 801) shifts argmax behaviour or just sharpens the
preference ranking further. Two predictions worth recording before the
runs:

- **DPO bulk:** more drift (the loss has no ceiling) → maybe argmax does
  shift on the targeted facts, but at higher SFT-preservation cost.
- **IPO bulk:** still capped at the 1/(2β) target → broader coverage
  but likely the same "doesn't change argmax" pattern.

If neither bulk cell closes the argmax gap, the project's named "fix
plausible hallucinations" goal needs a different lever entirely — RLHF
with a real reward model, or **rejection sampling at inference** (sample
N, pick the highest-reward one), or further-curated data that *includes
the model's own argmax output as the rejected side* (so DPO actually sees
what to push down).

The `dpo_curated` checkpoint is preserved as the policy that **maximises
the preference-loss signal at the cost of broader behaviour**; the
`ipo_curated` checkpoint is preserved as the policy that **preserves
SFT behaviour while learning the preference ordering**. Either can serve
as a Stage-06 RL starting point depending on whether the next stage
prioritises preference-shaping or capability-preservation.

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
    dpo_curated.yaml   the dpo run reported here
    ipo_curated.yaml   the ipo run reported here
    dpo_bulk.yaml      scaffolded; not run
    ipo_bulk.yaml      scaffolded; not run
  docs/
    RESULTS.md         this file
```

End-of-run scoring against `eval/questions.yaml` is still pending (will
land as `05_dpo/score_eval.py` once the judge-mode agent invocation
pattern is implemented).
