# Stage 05 — DPO against plausible hallucinations

Stage 04 left us with a model that answers Arknights questions in chat
format. It mostly recalls the right facts — but when it doesn't, it
**fabricates fluently**. That's the project's named failure mode (the
README has been calling out "preference pairs against plausible
hallucinations" since the first commit), and Stage 05 is where we fix
it.

The mechanism: **direct preference optimisation** (DPO; Rafailov et al.
2023). For each prompt the model sees a *correct* answer (`chosen`) and
a *plausible-but-wrong* answer (`rejected`), and the loss pushes the
policy's likelihood of `chosen` above `rejected` relative to a frozen
reference (the SFT checkpoint). No reward model, no rollouts — just
log-likelihood arithmetic.

> Status: **All four cells complete** — `dpo_curated`, `ipo_curated`,
> `dpo_bulk`, `ipo_bulk`. The headline result holds at scale: every cell
> achieves val pair_acc ≥ 0.92, but argmax-decoded probes are largely
> unchanged from the Stage-04 SFT baseline. 5× more pairs did not close
> the gap — bulk DPO **tripled SFT drift (+0.31 → +0.98) for zero val_loss
> improvement**, while bulk IPO floored at the same `1/(2β)` target as
> curated. See [`docs/RESULTS.md`](docs/RESULTS.md) for the full 2×2
> trajectory tables, the four-way probe comparison, and the
> interpretation. **Data generation is done by an Arknights-knowledge
> agent** — see `data_gen/AGENT_BRIEF.md` §5 for the full pair spec.

## 1. The data — produced by the agent

`05_dpo/` does not include a `generate_pairs.py` script. The preference
pairs are authored by the Arknights-knowledge agent following
`data_gen/AGENT_BRIEF.md` §5. Two flavours land in `data/dpo/`:

```
data/dpo/
├── bulk_train.jsonl        ~5000 lines — adversarial canon-swap at scale
├── bulk_val.jsonl          ~250
├── curated_train.jsonl     ~500     — hand-vetted, focused on top failure modes
└── curated_val.jsonl       ~50
```

Format (one JSON object per line, schema in `data_gen/AGENT_BRIEF.md` §5):

```json
{
  "prompt":   [{"role": "system", ...}, {"role": "user", "content": "..."}],
  "chosen":   [{"role": "assistant", "content": "<canon-correct answer>"}],
  "rejected": [{"role": "assistant", "content": "<plausible-but-wrong answer>"}],
  "category": "event",
  "fault_type": "wrong_episode",
  "source": "cn/stories/main_00.txt"
}
```

`fault_type` is the *kind* of wrongness — `wrong_subject`, `wrong_episode`,
`canon_swap`, `off_canon`, `over_specific`, `partial_truth`,
`should_refuse`. Used for stratified analysis, not fed to the model.

## 2. The experiment — a 2×2 ablation

|              | **DPO**          | **IPO**          |
|--------------|------------------|------------------|
| **Bulk pairs (~5000)** | `dpo_bulk`       | `ipo_bulk`       |
| **Curated pairs (~500)**  | `dpo_curated`    | `ipo_curated`    |

Two axes:

- **DPO vs IPO** — same data, different loss function. DPO uses a
  log-sigmoid link (Rafailov 2023); IPO (Azar et al. 2023) uses an
  identity link that's more robust to label noise. At 0.6B + a
  hand-authored dataset where the chosen/rejected boundary is sometimes
  fuzzy, IPO is plausibly better; the ablation answers it.
- **Bulk (5000) vs curated (500)** — data quantity vs quality. Bulk is
  the agent at scale; curated is the agent in deliberation. The
  classical preference-tuning finding (Lambert et al. 2024) is that a
  few hundred high-quality pairs often beat thousands of bulk ones — we
  test whether that holds at 0.6B on Arknights.

All four runs start from the **Stage 04 SFT winner** (full-FT or LoRA,
whichever closed-book-Q&A'd best) and use that same SFT checkpoint as
the frozen DPO reference.

## 3. Catastrophic preference collapse — what to watch for

DPO has two infamous failure modes at small scale:

1. **The model learns to refuse everything** — pushing `chosen` over
   `rejected` can be cheaply done by making the policy assign low
   probability to *both* and slightly lower to `rejected`. Output
   quality drops across the board. Watch: assistant-token loss on the
   held-out **SFT val set** (`data/sft/qa_val.jsonl`); it should stay
   close to the Stage 04 final value.

2. **Verbatim memorisation of `chosen`** — the policy collapses to
   reproducing the chosen string token-for-token instead of learning
   the underlying preference. Watch: at eval time, sample at T=1.0 and
   check distinct-N — a sudden drop in lexical diversity is the
   tripwire.

The Stage 03 README warned about "catastrophic forgetting" between CPT
and the base; Stage 05's analogue is preference collapse. Same shape of
risk: aggressive alignment to a small dataset erodes the broader
capability.

## 4. The training script (`train_dpo.py`)

Hand-rolled, **not** `trl.DPOTrainer` — `AGENTS.md` Conventions: every
operation is visible. The DPO/IPO loss derivation is the educational
core of this stage and lives in the `dpo-loss` and `ipo-loss` EXERCISE
blocks.

The loop holds **two copies** of the model in memory:

- **Policy** (`model`) — trainable, gets gradient updates.
- **Reference** (`ref_model`) — frozen Stage-04 SFT checkpoint, no grad.

Each batch:

1. Tokenize `prompt`, `chosen`, `rejected` for the whole batch.
2. Forward both policy and reference on (prompt + chosen) → log-prob of
   chosen tokens under each model.
3. Same for (prompt + rejected).
4. Compute log-ratios:
   - `log_ratio_chosen   = logπ_policy(chosen | prompt) − logπ_ref(chosen | prompt)`
   - `log_ratio_rejected = logπ_policy(rejected | prompt) − logπ_ref(rejected | prompt)`
5. Apply the DPO or IPO loss to those two log-ratios.

VRAM check: 0.6B model in bf16 ≈ 1.2 GB; two copies = 2.4 GB; activations
+ optimiser state pushes the LoRA path to ~6 GB peak, well within a
4090's 24 GB. Full-FT DPO peaks around 14-16 GB — tight but feasible.

Pedagogically central regions wear `EXERCISE` markers:

| Block | What you implement |
|-------|--------------------|
| `dpo-format`       | Tokenize prompt/chosen/rejected; mask prompt tokens with -100 the same way Stage 04 does |
| `dpo-logprobs`     | Sum log-probs of the response tokens under one model (helper used by both policy and ref) |
| `dpo-loss`         | The DPO loss: `−E[log σ(β · (log_ratio_chosen − log_ratio_rejected))]` |
| `ipo-loss`         | The IPO loss: `E[((log_ratio_chosen − log_ratio_rejected) − 1/(2β))²]` |
| `train-loop`       | Standard warmup + constant LR + eval + early-stop |

## 5. Evaluation during training

Three signals, run every `eval_interval` steps:

- **DPO val loss** — same loss, computed on held-out preference pairs.
- **Pairwise accuracy** — fraction of val pairs where
  `logπ_policy(chosen | prompt) > logπ_policy(rejected | prompt)`. The
  natural primary metric: it asks the model the question DPO is trying
  to answer, in raw form.
- **SFT preservation** — mean assistant-token CE on
  `data/sft/qa_val.jsonl`. The tripwire for failure mode #1 above. If
  this rises significantly, drop the LR or stop.

End-of-run scoring against `eval/questions.yaml` is again outside the
train loop (`05_dpo/score_eval.py`, planned).

## 6. How to run

```bash
.venv/bin/pip install -r 05_dpo/requirements.txt

# pre-requisite: agent has produced data/dpo/{bulk,curated}_{train,val}.jsonl
ls data/dpo/*.jsonl

# smoke test on the cheap config
.venv/bin/python 05_dpo/train_dpo.py --config 05_dpo/configs/dpo_curated.yaml --smoke-test

# the four-run ablation
for cfg in dpo_bulk dpo_curated ipo_bulk ipo_curated; do
    .venv/bin/python 05_dpo/train_dpo.py --config 05_dpo/configs/$cfg.yaml
done
```

Checkpoints land at `data/checkpoints/<run>/`. LoRA-on-LoRA (DPO LoRA
adapter on top of the Stage 04 SFT LoRA adapter) keeps disk cost
trivial; full-FT DPO writes the full ~1.2 GB.

## 7. Files

```
05_dpo/
  README.md            this design doc
  train_dpo.py         DPO + IPO training loop  (EXERCISE: dpo-format,
                                                 dpo-logprobs, dpo-loss,
                                                 ipo-loss, train-loop)
  requirements.txt     transformers + peft + trl
  configs/             one YAML per ablation cell — clone, never edit in place
    dpo_bulk.yaml
    dpo_curated.yaml
    ipo_bulk.yaml
    ipo_curated.yaml
```

Eventual `docs/RESULTS.md` and an Arknights-style write-up follow once
training lands. The Stage 05 RESULTS table should report, per cell:
DPO/IPO val loss, pairwise accuracy on val, SFT-val CE delta vs Stage
04, and the eval/-scored gain/loss vs the Stage 04 baseline on
`eval/questions.yaml`.
