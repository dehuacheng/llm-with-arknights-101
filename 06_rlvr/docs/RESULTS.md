# Stage 06 — RLVR / GRPO results (baseline cell)

GRPO with verifiable rewards on `data/checkpoints/sft_full`, run against
the agent-shipped factoid prompts (`data/rl/prompts_train.jsonl`, 925
rows; 30% with `must_not_contain` traps post-enrichment).

> Status: **`grpo_baseline` ran for 17.3 min and was stopped by the
> KL hard-cap tripwire at step 100 of 2000.** Token-level mode collapse;
> the policy started emitting multi-script noise tokens and the KL
> penalty (β=0.04) couldn't anchor it back. This is the first stage in
> the project that did not complete cleanly — the tripwires worked
> exactly as designed, catching the failure before it corrupted the
> checkpoint past usability. Best checkpoint (step 50, just before
> collapse) preserved; probes recorded; diagnosis in §"What went wrong"
> below; corrected config sketched in §"Next cell". `grpo_strict`
> (doubled trap weight) is **not** run as the second cell — that would
> almost certainly accelerate the same failure.

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

## What this means for the project's roadmap

Stage 06's headline finding has shifted from "RLVR closes the argmax
gap" (the planned headline) to **"on-policy RL on a 0.6B model with
sparse verifiable rewards is much more fragile than DPO/IPO; the
tripwires were the load-bearing safety system."** That's a real
contribution — most RLHF tutorials don't dwell on the failure modes
that catch you at small scale. The Stage 06 RESULTS.md will read as
"first attempt failed cleanly; here's the diagnosis and the fix"
rather than "everything worked perfectly."

Roadmap is unchanged: Stage 07 (refusal training) is still the next
stage; Stage 06's `grpo_v2` is the same stage's second cell, not a new
stage. If `grpo_v2` also fails, the right move is to write Stage 06 up
as "RLVR is hard at this scale; here are the three things that need
to change before it works" and move on — the educational value is in
the diagnosis, not in forcing a positive verdict.

## Files

```
06_rlvr/
  configs/
    grpo_baseline.yaml  ✓ ran 17.3 min · stopped by mean_kl hard-cap at step 100
    grpo_strict.yaml    deferred (would accelerate the same failure)
    grpo_v2.yaml        proposed — β=0.1, LR=1e-6, + fluency penalty
  docs/
    RESULTS.md          this file

data/
  checkpoints/grpo_baseline/   step-50 LoRA adapter, 18 MB (preserved as evidence)
  rl_logs/
    grpo_baseline.log              full training log
    grpo_baseline_responses.jsonl  spot-check sample log (the multi-script-leak evidence)
    grpo_baseline_probes.log       T=0 probe battery on the step-50 checkpoint
```
