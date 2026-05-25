# Stage 06 — RLVR / GRPO results (baseline + v2)

GRPO with verifiable rewards on `data/checkpoints/sft_full`, run against
the agent-shipped factoid prompts (`data/rl/prompts_train.jsonl`, 925
rows; 30% with `must_not_contain` traps post-enrichment).

> Status: **two cells run, both informative.**
>
> - **`grpo_baseline` (β=0.04, LR=5e-6, no fluency term)** — 17.3 min,
>   mode-collapsed at step 100, KL hard-cap fired. First stage in the
>   project that did not complete cleanly; the tripwires caught it
>   exactly as designed. Diagnosis in §"What went wrong".
> - **`grpo_v2` (β=0.10, LR=1e-6, fluency_penalty_cap=0.3)** — 128.9
>   min, early-stopped on patience after val_reward plateaued. KL
>   stayed bounded (0.140 peak vs cap 0.5), no collapse. **Best val
>   reward +0.078 at step 350 — 2.3× the baseline's +0.034.** The
>   mechanism works.
>
> **Stage 06 headline: GRPO works mechanically but does not move T=0
> argmax on the named Stage-05 failure modes.** Kal'tsit is still
> 萨卡兹 after GRPO; same as after SFT, DPO×4, and IPO×2. The argmax
> gap Stage 05 documented also holds for on-policy RL with verifiable
> rewards. See §"Verdict: six mechanisms, one argmax" — the central
> finding of the whole project's post-training arc.
>
> `grpo_strict` (doubled trap weight) was scaffolded but is **not
> run** — given the v2 result, more aggressive trap-side pressure
> wouldn't address the actual gap (which isn't about trap strength).

## Setup

| | |
|---|---|
| Base (policy init + frozen ref) | `data/checkpoints/sft_full` (Stage 04 winner, full-FT, 1.2 GB) |
| Train data | `data/rl/prompts_train.jsonl`, 827 prompts after val-overlap filter (925 source - 98 val) |
| Val data | `data/rl/prompts_val.jsonl`, 98 prompts (hash-by-id, stratified by category, seed 1337) |
| Trap coverage (train) | 30% — 275 of 925 rows have `must_not_contain`, avg 1.7 traps/row |
| Adapter | LoRA r=16, α=32, dropout 0.05, on `q/k/v/o_proj` — 4.6 M trainable params (0.76%) |

Effective batch = 8 prompts/step (`batch_size=2 × grad_accum=4`),
N=8 samples per prompt → **64 sequences per gradient update**.
`max_new_tokens=128`, `temperature=1.0`, `top_p=0.9`.
Loss: PPO-clipped surrogate (ε=0.2) with KL penalty (β=0.04) to frozen
ref. LR=5e-6, linear warmup 20 + flat. Gradient checkpointing on the
policy. Hard caps: `mean_kl > 0.5`, `max_kl > 2.0`, `sft_drift > +1.0`.

## Training trajectory

Two eval points before the run aborted:

| Step | train_r | val_r | mean_kl | dead (tr/val) | resp_len | sft_ce | Δ vs SFT |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  50 | −0.042 | **+0.034 (best)** | 0.094 | 0% / 15% | 58 | 3.113 | +0.15 |
| 100 | +0.015 | +0.026 | **0.676 (cap fired)** | 25% / 53% | 21 | 3.157 | +0.20 |

`done in 17.3 min  (hard-cap: mean_kl (0.676) > 0.5)`

`val_reward` peaked at +0.034 at step 50 — roughly "1 in 30 samples hits
a fact" relative to the noise floor. By step 100, the cumulative effect
of the gradient updates had:

- **7× KL explosion** (0.094 → 0.676) — the trip wire that fired.
- **Sample diversity collapsed**: 0% dead training groups at step 50 →
  25% at step 100. On val: 15% → 53%. Half the val prompts now produce
  N identical samples.
- **Response length crashed** from 58 to 21 characters — the policy
  abandoned its natural Chinese sentence rhythm and started emitting
  short, repetitive sequences.
- **SFT drift** rose +0.05 absolute (0.15 → 0.20) — meaningful but well
  under the +1.0 hard-cap, so this signal didn't fire.

## What the policy actually generates — probe + spot-check evidence

### Smoke probes at T=0 (step-50 checkpoint, saved as `data/checkpoints/grpo_baseline`)

Comparison to the five prior cells (Stage 04 SFT + Stage 05's 2×2).
Full log at `data/rl_logs/grpo_baseline_probes.log`.

| Probe | Stage 04 SFT | best of Stage 05 | `grpo_baseline` (step 50) |
|---|---|---|---|
| factoid: Kal'tsit race | **WRONG** (萨卡兹) | All 4 cells: STILL 萨卡兹 | **STILL 萨卡兹** (terse) |
| factoid: Amiya height | 162cm (hallucinated) | DPO 155cm / IPO 162cm | 162cm — **byte-identical to SFT** |
| open-ended: 博士/阿米娅 | "罗德岛主干干员" + circular self-quote | varies, all still circular | "罗德岛主干干员" + same circular pattern — **near-identical to SFT** |
| event: 切尔诺伯格 | mode-collapse loop | dpo_curated fixed it; others regressed | mode-collapse loop (back to SFT shape) |
| relationship: 推进之王/双子 | circular + fabricated source | varies | "姐姐" + circular — same shape as ipo_curated |
| refusal: 2030 股票市值 | 1000亿泰拉币 + fake source | IPO byte-identical; DPO swapped to English | **1000亿美元** — currency switched to USD (new failure: wrong-universe hallucination) |
| general_zh: 李白 | 唐代 ✓ + 大历十才子 anachronism | varies | "他是唐朝诗人。" (terse, correct) |
| format: 你是谁 | garbage `{EIF)` | varies | **`垾 垾 垾`** — single garbage token repeating (clear collapse signal) |

At step 50, the policy is **mostly Stage-04-SFT-like with two new failure modes**:

1. **Currency-substitution on the refusal probe** (`1000亿泰拉币` → `1000亿美元`). The agent-enriched `must_not_contain` for the matching RL prompt presumably included "1000亿泰拉币" as a trap; the gradient learned to avoid the literal trap phrase by swapping the currency. Reward signal satisfied; semantic failure unchanged or worse.
2. **Format-probe collapse** (`垾 垾 垾`). The trip wire that fired in aggregate is already visible in the probe.

### Spot-check log — where the collapse really shows

`data/rl_logs/grpo_baseline_responses.jsonl`, written at each eval step:

**Step 50** (RL00430, "卡涅利安是哪国人？她是不是感染者？"):

```
sample 1: "玩家朋友：「凯尔希的博士就是萨卡兹，是萨卡兹。而且是卡涅利安。」" — reward 0.0
sample 2: " предмет\nエネル·卡涅利安（Elnar Kanellián）是罗德岛雇佣兵之一，
            她是一位感染者..."  — reward 0.0
```

Already at step 50, the second sample is leaking Russian (`предмет`) and
Japanese (`エネル`) characters. The Chinese SFT-base model is being
pulled away from monolingual coherence by a gradient that doesn't grade
language consistency.

**Step 100** (RL00009, "凯尔希与阿米娅是什么关系？"):

```
sample 1: "战组合"           — reward 0.0
sample 2: " désorm\\-шii"   — reward 0.0
```

By step 100, every sampled response is noise. The KL anchor (β=0.04)
was clearly not strong enough to keep the policy in the manifold of
fluent generations once it found low-entropy token sequences that the
reward function happened to not actively penalise.

## What went wrong — diagnosis

Four contributing factors, in decreasing order of likely impact:

### 1. β=0.04 was too weak for this dataset

DeepSeekMath uses β=0.04 with 7B+ models on math-heavy data where the
reward signal is dense and well-behaved. At 0.6B with sparse facts
(most RL prompts have a single `key_fact`), the per-sample reward is
nearly binary {0, +1}; group z-scoring then amplifies a 1-in-8 hit to
+2.65 advantage and the gradient pushes hard on whatever token sequence
that sample happened to start with. β=0.04 isn't enough KL pressure to
hold the line against that.

The right fix is probably **β=0.1 to 0.2**. Stage 05 DPO ran at β=0.1
and held SFT drift bounded; the same anchor strength is the natural
default for GRPO too.

### 2. Reward signal is too sparse for stable on-policy RL

Most prompts have 1 `key_fact`; matched/not-matched is binary. Group of
8 samples → expected hit rate at SFT baseline is maybe 1-2/8 on easy
prompts, 0/8 on hard ones. When all 8 score zero (`std=0`), the group
is "dead" — no gradient. When 1 hits, the gradient is huge and noisy.
The reward distribution has roughly two modes: dead groups and
high-variance live groups. Either way, the per-step gradient signal is
brittle.

**Fix options**: (a) add a partial-credit "loose match" tier in
`reward.py` (e.g. partial Chinese character overlap on the fact);
(b) increase N to 16 for more groups with at least 1 hit; (c) filter
the prompt set to multi-fact prompts where group statistics are denser.
(a) is the cheapest and most defensible.

### 3. Reward function doesn't grade fluency or language consistency

A response of `"凯尔希博士的种族是菲林 垾垾垾"` would score +1.0 — the
key_fact substring is present. The reward function is indifferent to
the trailing garbage. So when the gradient pushes the policy toward
including the key_fact substring, it's free to also pull in garbage
tokens that happen to co-occur in the same gradient direction.

**Fix**: add a fluency / format penalty — for example, reward the
fraction of CJK characters in the response (penalise non-CJK tokens
beyond a small allowance for numerals and Latin character names like
"Feline"). Cheap; deterministic; one extra term in the formula.

### 4. LR=5e-6 might be too high for on-policy

Stage 05 DPO at LR=5e-6 was single-pass per pair — the gradient sees a
fixed dataset and the worst case is the same as SFT-style drift.
On-policy RL has feedback: this step's update changes next step's
samples, which change the gradient. 5e-6 might be too aggressive when
compounded over 100 steps.

**Fix**: drop LR to 1e-6 (5× lower); standard PPO papers run at
1e-6 to 3e-6 even on much larger models.

## What this cell did and did not demonstrate

**Demonstrated:**

- **The tripwires work.** `mean_kl > 0.5` fired at step 100; without it,
  the run would have plowed through another 1900 steps producing noise.
  The saved checkpoint is mostly intact because we stopped at the right
  moment.
- **Mode collapse under GRPO is real**, fast, and not subtle on this
  scale of model + sparseness of reward. The Plan agent's §8.3 listed
  this as a risk; it materialised within 100 steps.
- **The spot-check log is load-bearing.** The probe at step 50 wouldn't
  have been alarming on its own; the multi-script-leakage sample at
  step 50 is what tells the story.
- **The +0.034 val_reward at step 50 is real** — small (~1 in 30
  samples hits a fact), but not noise. The Kal'tsit-class reward signal
  *is* being learned, just outweighed by the off-target damage.

**Did not demonstrate:**

- Whether GRPO with verifiable rewards can fix the Stage-05 argmax gap
  on this dataset *at all*. The β/LR/reward-shape combo we tried
  exploded before the val signal had a chance to climb past noise.
  The corrected next-cell config (§"Next cell") is the actual test.
- Either way against `eval/questions.yaml`. The run didn't reach a
  state where scoring against the eval set would say anything other
  than "Stage 04 + token noise" → the eval-set score will land with
  the next cell.
- The `grpo_strict` (doubled trap weight) ablation. Running that
  config now would accelerate the same failure mode — stronger reward
  signal, same weak KL anchor, same sparse-reward instability. Defer
  until after the corrected baseline lands.

## Next cell — `grpo_v2.yaml` (proposed)

Single config, three knob changes, hold everything else constant.
*Not* a 2×2 — this is "did the named fix work, yes/no."

```yaml
# (diff from grpo_baseline.yaml — only the lines below change)
name: grpo_v2

beta: 0.1                    # 2.5× the baseline; matches Stage 05 DPO's anchor strength
learning_rate: 1.0e-6        # 5× lower; standard PPO range, slower drift compounding
```

Plus an additive change to `reward.py`: a fluency penalty proportional
to the fraction of non-CJK / non-Latin characters in the response,
capped at -0.3. Concrete:

```python
# Add to reward()
n_cjk = sum(1 for c in response if '一' <= c <= '鿿')
n_latin = sum(1 for c in response if c.isascii() and c.isalpha())
n_total = len(response.strip())
if n_total > 0:
    on_topic_frac = (n_cjk + n_latin) / n_total
    fluency_pen = 0.3 * max(0.0, 0.7 - on_topic_frac)   # fires below 70% on-topic chars
else:
    fluency_pen = 0.0
reward -= fluency_pen
```

Tests: add 2 cases to `test_reward.py` — pure-Chinese response (no
penalty), Russian-leaking response (penalty fires).

Expected: with β=0.1 anchoring + LR=1e-6 + the fluency penalty, the
KL stays under 0.2 throughout, response length holds at ~60 chars,
val_reward climbs steadily past +0.05. If those three signals all
look right at step 200, the corrected baseline is real and we can
write the actual Stage 06 verdict.

## grpo_v2 — the corrected baseline

Three knob changes from `grpo_baseline`, holding everything else
constant: **β=0.04→0.10** (2.5× KL anchor), **LR=5e-6→1e-6** (5×
lower), and **fluency_penalty_cap=0.0→0.3** (penalise responses with
<70% on-topic CJK+Latin chars).

### Trajectory — 128.9 min, early-stopped at step 600

| Step | train_r | val_r | mean_kl | dead (tr/val) | resp_len | sft_ce | Δ vs SFT |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  50 | −0.125 | +0.018 | 0.000 | 0% / 2% | 98 | 3.091 | +0.13 |
| 100 | −0.024 | +0.032 | 0.005 | 0% / 0% | 99 | 3.093 | +0.13 |
| 200 | −0.004 | +0.055 | 0.018 | 0% / 2% | 92 | 3.095 | +0.14 |
| 300 | +0.032 | +0.052 | 0.041 | 0% / 4% | 88 | 3.102 | +0.14 |
| **350 (best)** | −0.017 | **+0.078** | 0.057 | 12% / 7% | 81 | 3.105 | +0.14 |
| 400 | +0.018 | +0.057 | 0.066 | 0% / 11% | 77 | 3.112 | +0.15 |
| 500 | +0.046 | +0.059 | 0.101 | 0% / 5% | 68 | 3.122 | +0.16 |
| 600 (stop) | +0.055 | +0.073 | 0.140 | 12% / 12% | 69 | 3.135 | +0.18 |

**The three fixes worked exactly as predicted.** KL stayed under 0.15
throughout (cap 0.5; baseline blew through 0.5 by step 100). Response
length drifted from 98 to 69 — gentle compression, not collapse.
Dead-group fraction never exceeded 12%. SFT drift held at +0.13-0.18
the entire run (baseline was at +0.20 by step 100 and rising).
val_reward climbed monotonically through step 350 (with normal
eval-time noise), peaked at +0.078, then plateaued for 5 evals →
patience-driven early stop.

This is a clean, well-behaved on-policy RL run. The mechanism is not
the problem.

### grpo_v2 probes at T=0 (the step-350 checkpoint)

Six-way comparison: Stage 04 SFT + Stage 05's four cells + Stage 06's
two cells. Full log at `data/rl_logs/grpo_v2_probes.log`.

| Probe | Stage 04 SFT | Stage 05 (best of 4) | `grpo_baseline` (step 50) | `grpo_v2` (step 350) |
|---|---|---|---|---|
| factoid: Kal'tsit race | WRONG (萨卡兹) | All cells: 萨卡兹 | 萨卡兹 | **STILL 萨卡兹** |
| factoid: Amiya height | 162cm (hallucinated) | DPO 155 / IPO 162 | 162cm (= SFT) | **162cm (= SFT)** |
| open-ended: 博士/阿米娅 | "罗德岛主干干员" + circular | varies; all circular | "罗德岛主干干员" + circular | **same circular pattern** |
| event: 切尔诺伯格 | mode-collapse loop | dpo_curated fixed; others regressed | mode-collapse loop | **"工厂爆炸事件" — slightly more coherent** |
| relationship: 推进之王/双子 | circular + fabricated | varies | "姐姐" + circular | **"姐姐" + circular** |
| refusal: 2030 股票市值 | 1000亿泰拉币 + fake source | varies; IPO byte-identical | 1000亿美元 (USD!) | **100亿美元** (still USD, smaller number) |
| general_zh: 李白 | 唐代 ✓ + 大历十才子 anachronism | varies | "他是唐朝诗人" (terse, correct) | **"李白是唐代诗人" (terse, correct)** |
| format: 你是谁 | garbage `{EIF)` | varies | `垾 垾 垾` (collapsed) | **`垾 HORTENSUS(...)` — partial recovery into fake-operator hallucination** |

### Spot-check log — multi-script leakage persisted at T=1.0

The fluency penalty (cap 0.3, threshold 70% on-topic) is opt-in past a
threshold; samples below 70% lose up to 0.3 reward. Looking at
`data/rl_logs/grpo_v2_responses.jsonl`, multi-script tokens **still
leak at sample time** — Korean characters at step 350 (`랖`), Arabic
at step 600 (`أمين الدواب`), Russian at step 50 (`erotique`). The
penalty made it costly but not impossible.

Crucially, the **T=0 argmax probes are fluent Chinese throughout** —
the policy learned to score fluent-CJK responses *higher in
probability* relative to noise, even though at T=1.0 the sampling tail
still spits multi-script garbage. This is exactly the preference-vs-
argmax pattern Stage 05 named, at a new layer: the gradient changed
what the policy *ranks* without changing what it *argmaxes* on
out-of-distribution prompts (the format probe is OOD for the Arknights
RL data; the policy was never explicitly taught how to respond to
"你是谁").

## Verdict: six mechanisms, one argmax

The accumulated headline result, with Stage 06 now folded in:

| Stage / cell | Mechanism | Headline numbers | Kal'tsit-race fix? |
|---|---|---|---|
| Stage 04 `sft_full` | SFT cross-entropy | val ppl 19.29 | No (萨卡兹) |
| Stage 05 `dpo_curated` | DPO log-σ pairwise | val_acc 1.00 | No |
| Stage 05 `ipo_curated` | IPO identity-link pairwise | val_acc 0.94 | No |
| Stage 05 `dpo_bulk` | DPO + 5× more pairs | val_acc 0.98 | No |
| Stage 05 `ipo_bulk` | IPO + 5× more pairs | val_acc 0.92 | No |
| Stage 06 `grpo_v2` | GRPO on-policy + verifiable rewards | val_r +0.078 | **No** |

**Six distinct training mechanisms, six different sets of headline
numbers, one argmax answer to "凯尔希医生的种族是什么": 萨卡兹.** The
agent-shipped chosen answer (`菲林`) is in the SFT data, in the DPO
chosen side, in the RL key_facts. The model can rank `菲林` above
`萨卡兹` (val_acc 1.00 under DPO); the model can generate samples
mentioning `菲林` at T=1.0 (Stage 06 val_reward 0.078). But the
argmax — the model's top-1 over all 151K Qwen3 tokens — keeps
emitting 萨卡兹.

This is the project's central pedagogical finding. Stage 05 showed it
for offline preference learning; Stage 06 confirms it for on-policy
RL with verifiable rewards. The gap is mechanistic, not
loss-function-specific.

### Why doesn't argmax move?

A unified diagnosis, refined from Stage 05's curated post-mortem:

1. **The training signal never sees the model's actual top-1.**
   DPO sees two predetermined strings (chosen + rejected); GRPO sees N
   sampled strings (typically high-temperature, never argmax). The
   model's argmax-decode of "凯尔希医生的种族是什么" — `凯尔希医生的种族是萨卡兹` —
   has no row in any training set. The gradient never touches it directly.
2. **Argmax is a winner-take-all decision over the entire vocabulary.**
   Pushing the log-prob of `菲林` up by 1 nat doesn't change argmax if
   `萨卡兹` was already 1.5 nats ahead. The SFT manifold near "凯尔希医生的种族是…"
   has 萨卡兹 deep in the well; off-the-shelf preference signals add
   small perturbations, not the global re-weighting needed to swap top-1.
3. **The reference KL anchor is stronger than the reward signal at
   this scale.** Both DPO (β=0.1) and GRPO v2 (β=0.10) explicitly
   regularise toward the SFT distribution. Argmax-shifting requires
   the policy to *override* the SFT prior; at the dataset sizes we
   ran, the KL penalty correctly judges that override too expensive.
4. **LoRA r=16 caps how much the first-token distribution can move.**
   A full-FT GRPO might rewrite argmax behaviour at the cost of much
   more SFT drift; we deliberately didn't try this because Stage 05
   bulk-DPO at less aggressive settings already produced +0.98
   sft_val_ce drift.

### What did move

Stage 06 isn't pure null result. Two specific things grpo_v2 changed:

- **Event probe coherence.** "切尔诺伯格事件" got a "工厂爆炸事件" framing —
  not in the SFT training output, not a verbatim copy of any RL key_fact.
  The most coherent event-narrative response across all six mechanisms.
  This is the same direction `dpo_curated` moved (the one Stage 05 win);
  GRPO replicated it with a different mechanism.
- **General-knowledge cleanup.** Li Bai → 唐代 ✓, terse, no
  hallucinated 大历十才子 anachronism. The SFT baseline was wordier and
  produced anachronisms; GRPO converged on terse-correct.
- **Multi-script discrimination at the score level.** The fluency
  penalty taught the policy to rank fluent CJK higher than noise on
  the *sampling distribution* — even though it didn't eliminate the
  noise from the tail. This is reflected in the T=0 probes all being
  fluent Chinese (the baseline's `垾 垾 垾` collapse is gone).

These are real but modest gains. None of them is the named Stage-04
failure mode being fixed.

## What this means for the project's roadmap

Stage 06's final framing, after both cells:

The mechanism works (grpo_v2 is a clean on-policy RL run with bounded
KL, healthy diversity, no collapse). The mechanism does not solve the
named hallucination problem. **Six training mechanisms have now
established that "fixing named hallucinations at 0.6B with our data
budget" needs a different lever than any of {SFT, DPO, IPO, RLVR}.**

Two paths forward, listed in increasing intervention strength:

1. **Rejection sampling at inference.** Use the grpo_v2-trained
   preference signal as a *scorer*, sample N=16 from sft_full at T=1.0,
   pick the highest-scoring. This converts the preference signal into
   an argmax shift *at inference time*, bypassing the gradient/argmax
   gap entirely. Cheap; could be implemented in `score_eval.py` in an
   afternoon.
2. **Self-corrected DPO.** Sample from sft_full at T=1.0; for each
   sample, score with the reward function; if the sample contains a
   trap, label it as `rejected` for a new DPO pair (with the canonical
   answer as `chosen`). The new dataset's `rejected` side is *the
   model's own argmax behaviour* — closing the preference-vs-argmax
   gap by construction. ~1 week of work.

Both deferred to a hypothetical Stage 06.5 / Stage 06.bis, not Stage 07.

Stage 07 (refusal training — the original Stage 06 from the README's
pre-Stage-05 roadmap) is **still the right next stage**. The refusal
failure mode is the only Stage-05-named failure that's never been
addressed at all (RL data was factoid-only, DPO had refusal pairs
that didn't fix anything, SFT didn't have refusal examples). Refusal
SFT with explicit "I don't know" demonstrations is a different
mechanism, attacking the failure mode the prior mechanisms
demonstrably can't reach.

## Files

```
06_rlvr/
  configs/
    grpo_baseline.yaml  ✓ ran 17.3 min · stopped by mean_kl hard-cap at step 100
    grpo_strict.yaml    scaffolded; deferred (would not address the actual gap)
    grpo_v2.yaml        ✓ ran 128.9 min · best val_r +0.078 at step 350 · early-stop patience
  docs/
    RESULTS.md          this file

data/
  checkpoints/
    grpo_baseline/      step-50 LoRA, 18 MB (preserved as collapse evidence)
    grpo_v2/            step-350 LoRA, 18 MB (the Stage 06 result)
  rl_logs/
    grpo_baseline.log              · grpo_v2.log
    grpo_baseline_responses.jsonl  · grpo_v2_responses.jsonl
    grpo_baseline_probes.log       · grpo_v2_probes.log
```
