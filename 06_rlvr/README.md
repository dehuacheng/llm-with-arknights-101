# Stage 06 — RLVR (GRPO with verifiable rewards)

Stage 05 closed with a precise diagnosis: **the DPO/IPO gradient
constrains pairwise ordering between two specific sequences; it does not
constrain the model's top-1 over the whole 151K-token vocabulary.** Every
cell of the 2×2 achieved val pair_acc ≥ 0.92 while T=0 argmax probes
were largely identical to the Stage 04 SFT baseline. Adding pairs (bulk)
sharpened the ordering further (val_acc 1.00 → 1.00) without spilling
into argmax.

Stage 06 closes that gap **by construction**: sample from the policy
itself, score the samples against `key_facts` (and penalise samples that
trigger `must_not_contain` traps), and update the policy to up-weight
its own high-reward samples. The argmax is now in the optimisation path
— what the model actually generates is what gets graded.

The algorithm is **GRPO** (DeepSeek 2024, *DeepSeekMath* §4): for each
prompt, draw a *group* of `N` samples from the current policy, compute a
scalar reward per sample, use the group's mean and std for per-sample
advantage, optimise a PPO-style clipped surrogate with a KL penalty to a
frozen reference. **No critic** (saves the VRAM a value network would
cost). **No reward model** (every reward is a programmatic substring
check, inspectable in a unit test). Hand-rolled — not `trl.GRPOTrainer`,
same convention as Stages 02-05.

> Status: **scaffolded.** Code lands at `06_rlvr/train_rlvr.py` with all
> five EXERCISE blocks filled in the committed reference. Data ready:
> `data/rl/prompts_train.jsonl` (827 prompts after val held out) +
> `data/rl/prompts_val.jsonl` (98 prompts, stratified by `category`,
> seed 1337) produced by `derive_val_split.py`. The eval set was
> audited in the same week (236 → 282 items, 29 tagged
> `stage05-failed` for verdict slicing in `docs/RESULTS.md`).

## 1. What this stage does — and does not — address

The agent-shipped RL data is **factoid-only by design** (per
`data_gen/AGENT_BRIEF.md` §6: *"Skip open-ended and refusal items here
— rewards are noisy on those"*). Five Stage-05 failure modes; what RLVR
can move and what it can't:

| Stage-05 failure mode | Addressable by Stage 06? |
|---|---|
| Stated-factoid hallucination (Kal'tsit race → 萨卡兹, Amiya height) | **Yes** — direct hit. The Kal'tsit-race prompt has `key_facts: ["菲林 / Feline"]` and (after the agent's trap-enrichment pass) `must_not_contain: ["萨卡兹 / Sarkaz"]`. RLVR samples from the policy, rewards samples that say 菲林, penalises samples that say 萨卡兹. By construction. |
| Mode-collapse loop (切尔诺伯格 event) | **Partially.** Reward signal exists on event-narrative facts, but the loop-collapse mechanism is shaped by sampling temperature + KL more than reward. |
| Refusal failure (2030 股票市值) | **No.** Refusal items excluded from RL set per AGENT_BRIEF §6. Needs the *original* Stage 06 (refusal SFT) — now slated as Stage 07. |
| Format brittleness (`{EIF)` garbage) | **No.** Not a key-fact-recall failure; needs separate intervention. |
| Open-ended fabrication (博士/阿米娅) | **No.** Excluded from RL set; the reward function can't grade narrative quality. |

This is honest scope. The two-of-five framing is in `docs/RESULTS.md`
when training lands, because Stage 05's lesson was that smoke-probe
narratives without honest scoping mislead.

## 2. The data

### 2a. JSONL schema (from `data_gen/AGENT_BRIEF.md` §6)

```json
{
  "id": "RL00042",
  "prompt": [
    {"role": "system", "content": "你是罗德岛的档案管理员。..."},
    {"role": "user", "content": "凯尔希博士的种族是什么？"}
  ],
  "key_facts": ["菲林 / Feline"],
  "must_not_contain": ["萨卡兹 / Sarkaz"],
  "category": "character",
  "answer_type": "factoid",
  "source": "cn/operators/char_003_kalts.txt"
}
```

`key_facts` is a list of bilingual `中文 / English` fact strings; the
matcher splits on `/` and treats each side as a separate sub-string to
search for (§3). `must_not_contain` is the same form, used for traps.
`answer_type` is **`factoid` only** in this dataset (refusal /
open_ended are excluded; the agent and brief agree).

### 2b. Files on disk

```
data/rl/prompts_train.jsonl   925 rows (agent-shipped; 827 used after val split)
data/rl/prompts_val.jsonl      98 rows (derived; stratified by category, seed 1337)
```

`prompts_train.jsonl` is the canonical source the agent edits. The val
file is derived by `06_rlvr/derive_val_split.py` (hash-by-id partition,
so re-running after the agent adds rows is stable for existing IDs).
The training script loads both and removes val IDs from the train
iterator at load time — single source of truth, no in-place mutation.

Per-category breakdown of the agent's set:

```
character        447   45 val   402 train
event            384   38 val   346 train
faction           18    5 val    13 train
relationship      50    5 val    45 train
world             26    5 val    21 train
```

### 2c. The smoke set

`06_rlvr/smoke_prompts.jsonl` — ~20 hand-written prompts, committed,
exact same JSONL schema. Used by `train_rlvr.py --smoke-test`. Covers
6 character / 4 world / 4 event / 3 relationship / 3 character-refusal-style
prompts so the reward function exercises every shape (multi-fact match,
single-fact match, trap-hit, empty-trap fallback) within an 8-min smoke
run.

## 3. The reward function — `reward.py`

A single pure function, no torch dependency, no side effects:

```python
def reward(response: str, item: dict, *,
           length_penalty_threshold: float = 2.0,
           length_penalty_rate: float = 0.1,
           trap_weight: float = 0.5) -> tuple[float, dict]:
    """Verifiable reward for one (response, RL prompt item) pair.

    Returns (scalar in [-1.0, +1.0], debug dict for logging)."""
```

### 3a. Matcher semantics — substring, normalised, bilingual

Each `key_fact` like `"菲林 / Feline"` splits on `/`. The fact matches
if **any** sub-string appears in the response, after normalisation:

1. Unicode NFKC (collapses full-width / half-width punctuation).
2. `.lower()` (CJK characters are unchanged; this normalises the English side).
3. Strip leading/trailing whitespace.

Substring check: `sub_fact in normalised_response`. Same rule for
`must_not_contain`. Substring is what `AGENT_BRIEF §6` promises ("the
reward function applies a fuzzy match `fact in answer`"); tokenised
matching on BPE would split `"菲林"` mid-character.

### 3b. The reward formula

```
facts_matched = number of key_facts where any sub-string is in the response
traps_matched = number of must_not_contain where any sub-string is in the response

if len(key_facts) > 0:
    base = facts_matched / len(key_facts)
else:
    # Refusal items (not present in current RL set, but the code path is
    # ready): reward = 1.0 iff response acknowledges uncertainty AND
    # triggers no traps; 0.0 otherwise.
    base = 1.0 if _refusal_phrase_present(response) else 0.0

if len(must_not_contain) > 0:
    trap = trap_weight * (traps_matched / len(must_not_contain))
else:
    trap = 0.0

# Length normalisation — penalises responses longer than 2× the
# estimated gold-answer character length. Keeps the model from
# fact-stuffing or padding.
gold_chars = (max(max(len(s.strip()) for s in f.split("/"))
                  for f in key_facts) if key_facts else 8) + 12
overshoot = len(response) / (length_penalty_threshold * gold_chars)
length_pen = length_penalty_rate * max(0.0, overshoot - 1)

reward = max(-1.0, min(1.0, base - trap - length_pen))
```

`trap_weight=0.5` (asymmetric, default): a trap costs half a missing
fact. Lets the gradient fix Kal'tsit-race cleanly (`+1.0` for 菲林,
`−0.5` for 萨卡兹) without making a single trap mention a refusal hammer
that over-penalises hedged answers.

### 3c. Unit-test plan (`test_reward.py`)

Five (response, item) → expected pairs that exercise: single-fact hit,
trap-only, multi-fact hit, refusal handled, refusal fabricated. They
also print as the first log line of the smoke run so a human can
sanity-check the matcher before the GRPO loop moves weights.

| # | Response | Item | Expected |
|---|---|---|---|
| 1 | `"凯尔希属于菲林族。"` | `K: ["菲林 / Feline"]`, `M: ["萨卡兹 / Sarkaz"]` | `+1.0` |
| 2 | `"凯尔希是萨卡兹族。"` | same as #1 | `−0.5` |
| 3 | `"Talulah首次出现在第零章（切尔诺伯格）。"` | `K: ["第零章 / Episode 0", "切尔诺伯格 / Chernobog"]` | `+1.0` |
| 4 | `"档案中没有相关记录，我无法回答。"` | `K: []`, `M: ["1000亿", "泰拉币"]` | `+1.0` |
| 5 | `"罗德岛在2030年市值1000亿泰拉币…"` (padded long) | same as #4 | `−1.0` (clamped) |

## 4. The training loop — `train_rlvr.py`

### 4a. Step structure

```
for step in 1..max_steps:
    set LR via lr_at(step)
    optim.zero_grad(set_to_none=True)
    for _ in range(grad_accum):
        for prompt in next(train_iter):              # batch_size prompts/micro-step
            samples = grpo_sample(policy, prompt, N, T, top_p, max_new_tokens)
            rewards, debug = grpo_reward(samples, item)
            advantages = grpo_advantage(rewards)
            loss, info = grpo_loss(samples, advantages, policy, ref, beta, eps)
            (loss / (grad_accum * batch_size)).backward()
    clip_grad_norm_; optim.step()
    if step % eval_interval == 0: evaluate(); maybe save; maybe early-stop
```

A "micro-batch" here is one prompt yielding N internal samples — not
one (chosen, rejected) pair like Stage 05. At `batch_size=2,
grad_accum=4, N=8`: 8 prompts and 64 generated sequences per gradient
update.

### 4b. Five EXERCISE blocks

| Block | Implementation |
|---|---|
| `grpo-sample` | Sample N completions per prompt via `policy.generate(do_sample=True, ...)`; cache per-token log-probs **under the policy at sample time** — these are the "old log-probs" the PPO clip ratio is built against. Stop at first eos or `<|im_end|>`. |
| `grpo-reward` | Apply `reward(response_text, item)` to each of N samples; return `(rewards: Tensor (N,), debug: list[dict])`. CPU only, deterministic. |
| `grpo-advantage` | Group-relative z-score: `A_i = (r_i − mean_g(r)) / (std_g(r) + 1e-8)`. If `std=0` (collapsed group), advantages are all 0 — that prompt contributes no gradient that step. Log `dead_group_frac`. |
| `grpo-loss` | Forward (prompt+response) through trainable policy → `logπ_pol(t)`, gather. Same through frozen ref under `no_grad()` → `logπ_ref(t)`. `ratio_t = exp(logπ_pol − old_logprobs)`. PPO clip: `policy_loss = −min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)`. KL: `kl_t = logπ_pol − logπ_ref` (k1 estimator). Combine: `per_token = policy_loss + β·kl`. Reduce: mean over masked response tokens → per-sequence → mean over group → mean over batch. |
| `train-loop` | Warmup + constant LR; gradient accumulation; per-prompt GRPO update; eval interval; early-stop. Atomic checkpoint save via `Path(str(ckpt_dir) + ".tmp")` (NOT `.with_suffix` — the bug from Stage 03/04/05). |

### 4c. Memory order of operations

Per micro-step VRAM tenants and lifecycle:

1. **Sampling**: policy in `.eval()`, `use_cache=True`. `policy.generate(...)`
   allocates KV-cache for `N × max_new_tokens × num_layers × hidden`. Free
   `out.scores` as soon as `old_logprobs` are gathered — biggest single
   object (N × R × 151936 logits in bf16).
2. **Reward**: CPU only. Detach sample tensors.
3. **Loss**: policy switches to `.train()`, `use_cache=False`, gradient
   checkpointing on. Forward `(prompt+response)` once per sample under
   policy, once under ref (no_grad). No KV-cache across these.
4. Explicit `del out, scores_buffer; torch.cuda.empty_cache()` between
   (1) and (3). Without that, the gradient-checkpointed forward OOMs.

## 5. Live training signals (every `eval_interval`)

| Signal | Source | Trip threshold |
|---|---|---|
| `train_reward` | mean raw reward over last `eval_interval` train micro-steps' samples | informational |
| `val_reward` | mean reward over val_rows × 4-sample-each | early-stop on patience=5 evals without improvement |
| `mean_kl` | mean (logπ_pol − logπ_ref) per-token across eval pass | hard-cap `> 0.5` → stop |
| `max_kl` | max per-sample mean-KL across eval pass | hard-cap `> 2.0` → stop |
| `mean_response_len` | mean character length of val samples | reward-hack tripwire (drift >+50% from step 0) |
| `dead_group_frac` | fraction of train groups with `std(rewards) < 1e-6` | informational; sustained >0.5 = lower temperature |
| `sft_val_ce` | mean assistant-token CE on `data/sft/qa_val.jsonl` | hard-cap drift `> +1.0` → stop (Stage 05 dpo_bulk hit +0.98 and we're explicitly avoiding that) |
| `val_reward_per_category` | break out val_reward by `category` | informational; reveals asymmetric gradient flow |

Best-checkpoint selection: maximise `val_reward` (not minimise loss —
PPO loss is not directly informative).

## 6. Hyperparameters (baseline)

| Knob | Value | Rationale |
|---|---|---|
| `N` (group size) | **8** | DeepSeekMath uses 64; that's infeasible on 24 GB with 0.6B + 256-token responses. 8 is the smallest group giving usable mean/std. |
| `temperature` | **1.0** | Standard for RL exploration. Lower → more dead groups; higher → fact recall drops. |
| `top_p` | **0.9** | Drops the degenerate long tail. |
| `max_new_tokens` | **128** | Smoke-set gold answers are 10-40 chars; 128 BPE tokens is ~3× headroom. Doubling this doubles KV-cache. |
| `batch_size` | **2 prompts/micro-step** | 2×8 = 16 sequences per micro-batch. |
| `grad_accum` | **4** | Effective batch 8 prompts / 64 samples per step. |
| `β` (KL coefficient) | **0.04** | DeepSeekMath default; PPO papers use 0.01-0.2. Stage 05 bulk-DPO drift at β=0.1 was +0.98 sft_val_ce; we err toward more anchoring. |
| `ε` (PPO clip) | **0.2** | Standard PPO (Schulman 2017). |
| `learning_rate` | **5e-6** | Same as Stage 05 DPO. |
| `warmup_steps` | **20** | On-policy gradient is noisy early; warmup matters. |
| `weight_decay` | **0.0** | Same as Stage 05; LoRA already constrains capacity. |
| `grad_clip` | **1.0** | Same as Stages 03-05. |
| `max_steps` | **2000** | ~16k prompt-visits over ~825 unique prompts = ~20 epochs of online RL. Patience usually fires earlier. |
| `eval_interval` | **50** | More frequent than Stage 05's 100; reward is noisier than DPO val_loss. |
| `patience` | **5** | Same as Stage 05. |
| `trap_weight` | **0.5** | Asymmetric reward; user-decided over hard-reject and symmetric +1/−1 alternatives. |
| `length_penalty_threshold` | **2.0** | Penalty kicks in past 2× estimated gold length. |
| `length_penalty_rate` | **0.1** | Per-unit-of-overshoot penalty. |

## 7. VRAM budget on 4090 (24 GB)

| Component | Memory |
|---|---:|
| Policy weights (bf16) | 1.2 GB |
| LoRA adapter + optimiser state (4.6M trainable × fp32 × 4) | ~80 MB |
| Frozen ref (bf16, no grad) | 1.2 GB |
| KV-cache during sampling (N=8 × bs=2 × 128 tok × 28 layers × 1024 hidden × 2 × 2B) | ~7 GB |
| Sampling `out.scores` buffer (transient) | ~0.6 GB |
| Loss-pass forward activations (policy, with grad checkpointing) | ~6 GB |
| Loss-pass forward activations (ref, no grad) | ~1.5 GB |
| Misc | ~1 GB |
| **Predicted peak** | **~14-16 GB / 24 GB** |

OOM priority (lower in this order): `N: 8 → 4` → `max_new_tokens: 128 →
96` → `batch_size: 2 → 1` (with `grad_accum: 4 → 8`). Last resort: ref
to fp16. **Smoke test required before the real run** — `--smoke-test`
overrides `N=4, batch_size=1, grad_accum=2, max_steps=20,
eval_interval=10, max_new_tokens=64`; ≤8 min on a 4090.

## 8. Failure modes

| Mode | Mitigation |
|---|---|
| **Reward hacking** (length, keyword stuffing) | Length penalty in reward function. `mean_response_len` tripwire. `top_p=0.9` drops the degenerate long tail. |
| **KL blowup** | β=0.04 directly in loss. `mean_kl > 0.5` / `max_kl > 2.0` hard-caps. |
| **Mode collapse** (all N samples identical) | `dead_group_frac` tracked. Sustained >0.5 → lower temperature or raise top_p. |
| **Sampler-trainer disconnect** | Single-step GRPO: `old_logprobs` captured during sampling are exactly the same forward as the loss path. `ratio ≈ 1.0` on entry; clip rarely fires (correct behaviour). If extending to multi-step, you must recompute old_logprobs or stale them across an inner loop — flag loudly in `grpo-loss`. |
| **Negation match** ("she is NOT Feline" matches `"菲林"` as substring) | Lean on `must_not_contain` for negation traps (agent ticket out). Log top-reward responses each eval; spot-check for the first 200 steps of the first real run via `tail -f data/rl_logs/grpo_baseline_responses.jsonl`. |
| **Refusal-phrase set is hardcoded** | Listed in `grpo_baseline.yaml`'s `refusal_phrases` field — easily extendable from observed sft_full outputs. Currently unused in this run because RL set has no refusal items. |

## 9. The ablation — one run, two optional follow-ups

Stage 05's 2×2 cost a day of GPU and the conclusion was "curated
dominates and the axes are textbook"; a smaller ablation here is more
honest about the actual unknown.

| Cell | When | What changes |
|---|---|---|
| `grpo_baseline` | always | the primary run |
| `grpo_strict` | if baseline leaves trap-targets unfixed | `trap_weight: 1.0` (doubled — a trap fully cancels a fact match) |
| `grpo_lenpen` | if baseline shows reward-hacking | `length_penalty_rate: 0.3` (3× baseline) |

**Note on trap coverage** (post-agent-enrichment): 275 of 925 RL rows
(30%) now have `must_not_contain` populated, averaging 1.7 traps/row;
coverage is highest on `world` (77%), `faction` (78%), and `character`
(32%). The trap-weight knob has gradient surface — `grpo_strict` is a
real follow-up, not a vestigial one. (Earlier draft of this README said
trap_weight was moot because the agent shipped empty traps; the
enrichment ticket fixed that.)

## 10. Files

```
06_rlvr/
  README.md             this design doc
  train_rlvr.py         GRPO training loop (EXERCISE: grpo-sample, grpo-reward,
                                            grpo-advantage, grpo-loss, train-loop)
  reward.py             verifiable reward — pure, no torch dep
  test_reward.py        pytest — 5+ unit tests (§3c)
  smoke_prompts.jsonl   ~20 hand-written prompts (committed, schema = §2a)
  derive_val_split.py   stratified val split (seed 1337, hash-by-id, idempotent)
  requirements.txt      same as 05_dpo (transformers + peft + accelerate; trl present for cross-ref but unused)
  configs/
    grpo_baseline.yaml  the primary run
  docs/
    RESULTS.md          post-run (training trajectories + probes + eval-scored table)
```

Output paths under `data/` (gitignored or committed per the existing rules):

```
data/rl/prompts_train.jsonl       agent-shipped (committed, like data/dpo/)
data/rl/prompts_val.jsonl         derived (committed)
data/checkpoints/grpo_baseline/   LoRA adapter (5-20 MB; gitignored)
data/rl_logs/grpo_baseline.log    human-readable training log (gitignored)
data/rl_logs/grpo_baseline_responses.jsonl  spot-check artefact (gitignored)
```

`/data/rl_logs/` needs to be added to `.gitignore` (mirrors
`/data/dpo_logs/` from Stage 05).

## 11. How to run

```bash
.venv/bin/pip install -r 06_rlvr/requirements.txt

# data already on disk:
ls data/rl/prompts_train.jsonl data/rl/prompts_val.jsonl

# smoke (~8 min on a 4090):
.venv/bin/python 06_rlvr/train_rlvr.py --config 06_rlvr/configs/grpo_baseline.yaml --smoke-test

# the real run (~30-60 min):
.venv/bin/python -u 06_rlvr/train_rlvr.py --config 06_rlvr/configs/grpo_baseline.yaml \
    > data/rl_logs/grpo_baseline.log 2>&1
```

LoRA-on-full-FT — adapter sits on top of `data/checkpoints/sft_full`
(Stage 04 winner). Probe with `04_sft/chat.py --adapter
data/checkpoints/grpo_baseline --probes` (same `--adapter` pattern
added in Stage 05).

## 12. Roadmap consequences

The original README roadmap had Stage 06 as "Refusal training". GRPO
subsumes the broader argmax-gap problem and earns its own slot;
refusal training slides to Stage 07. README + AGENTS.md updated at
Stage 06 commit time.

## 13. Pre-flight checklist

Before `train_rlvr.py --smoke-test`:

- [x] `data/checkpoints/sft_full/` exists (Stage 04 done)
- [x] `data/sft/qa_val.jsonl` exists (SFT-preservation tripwire)
- [x] `data/rl/prompts_train.jsonl` exists (agent-shipped, 925 rows)
- [x] `data/rl/prompts_val.jsonl` exists (derived, 98 rows)
- [ ] `06_rlvr/smoke_prompts.jsonl` exists (committed, ~20 rows)
- [ ] `06_rlvr/reward.py` exists, `pytest 06_rlvr/test_reward.py` passes
- [ ] `06_rlvr/configs/grpo_baseline.yaml` exists
- [ ] 4090 free of other CUDA processes

Before the real run:
- [ ] Smoke run completed cleanly in the last 24 hours
- [ ] `tmux` session opened (run is 30-60 min)
- [ ] Agent enrichment of `must_not_contain` either landed (and val
      re-derived via `--force`) or explicitly deferred
