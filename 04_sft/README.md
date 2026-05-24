# Stage 04 — SFT distillation into closed-book Q&A

Stage 03 left us with a CPT-ed Qwen3-0.6B that has Arknights lore in its
weights but doesn't *act* like an assistant — it's still a base model. It
continues text, it doesn't answer questions, and it has no notion of the
chat template. Stage 04 fixes that by **supervised fine-tuning on
distilled Q&A pairs**:

1. A teacher LLM (Claude / GPT-4) sees the source text from `data/raw/cn/`
   and writes high-quality Q&A pairs about it. (This is the
   "distillation" step.)
2. The student (post-CPT Qwen3-0.6B) is fine-tuned on those pairs **with
   the source removed** from the prompt — closed-book recall.
3. The result is a model that answers Arknights questions in Qwen's chat
   format, drawing on what CPT taught it.

> Status: **`sft_full` implemented and run** (1.2 GB ckpt at
> `data/checkpoints/sft_full/`, best val ppl 19.29). `sft_lora` skipped —
> the Stage-03 conclusion (full-FT + replay strictly dominates) extended
> here. See [`docs/RESULTS.md`](docs/RESULTS.md) for the training run, the
> probe battery, and the failure-mode analysis that motivates Stage 05.
> **The data generation itself is done by an Arknights-knowledge agent**
> — see `data_gen/AGENT_BRIEF.md` for the full spec.

## 1. The data — produced by the agent, not this stage's code

Stage 04 **does not include a `distill_qa.py` script**. The Q&A pairs
are authored by a separate Arknights-knowledge agent following
`data_gen/AGENT_BRIEF.md`. The agent's output lands at:

```
data/sft/
├── qa_train.jsonl    ~5000 lines
└── qa_val.jsonl      ~250 lines
```

Format (one JSON object per line):

```json
{
  "messages": [
    {"role": "system", "content": "你是罗德岛的档案管理员。..."},
    {"role": "user", "content": "凯尔希博士的种族是什么？"},
    {"role": "assistant", "content": "凯尔希博士是菲林 (Feline) 种族..."}
  ],
  "category": "character",
  "answer_type": "factoid",
  "source": "cn/operators/char_003_kalts.txt",
  "key_facts": ["菲林 / Feline"]
}
```

The `category` / `answer_type` / `source` / `key_facts` metadata is not
fed to the model; the train script uses it for stratified eval logging.

## 2. The training objective

Standard SFT: teacher-forced cross-entropy on the assistant's response
tokens only. Prompt tokens (system + user turn) get `label = -100` so
they don't contribute to the loss — the model learns to *generate* the
answer, not to *predict* the question.

The "label mask" is the pedagogically central trick of SFT and lives in
the `sft-format` EXERCISE block in `train_sft.py`.

## 3. Ablation — small, two cells

Stage 03 already did the full-FT vs LoRA ablation on CPT. Stage 04 picks
the winner from Stage 03 and runs **two SFT configs**:

|              | What |
|--------------|------|
| `sft_full`   | Full-FT of the Stage-03 best checkpoint on the distilled Q&A. Higher capacity, slower, expected to fit the Q&A distribution best. |
| `sft_lora`   | LoRA on the Stage-03 best checkpoint. Faster, cheaper, expected to underfit on the long-tail of Q&A items but preserve more of the CPT distribution. |

Both ablation cells of Stage 03 are valid starting points; pick the one
that beat the eval on Arknights without trashing general-Chinese ppl.

> Open: this is the place to consider **ORPO** (fused SFT + DPO) as a
> third config — it would land in one stage what Stage 04 + Stage 05
> currently do in two. Pedagogically we prefer the staged split, so ORPO
> isn't in scope here, but the configs are clone-and-rename if you
> change your mind.

## 4. The training script (`train_sft.py`)

Hand-rolled loop, **not** `trl.SFTTrainer` — same reason Stage 02 and 03
are hand-rolled: every operation is visible (`AGENTS.md` Conventions).
PEFT and Transformers handle the model and tokenizer; the loop is ours.

The Qwen3 tokenizer's `apply_chat_template` builds the prompt-formatted
input from `messages`. We then derive the label mask by re-running the
template **without** the assistant turn and recording where the assistant
content starts (everything before that index is set to -100).

Pedagogically central regions wear `EXERCISE` markers:

| Block | What you implement |
|-------|--------------------|
| `sft-format` | Apply Qwen's chat template; compute the prompt/response split; build the `-100` label mask on prompt tokens |
| `sft-loss` | Cross-entropy on the response tokens only (the masking is already in the labels — pass them through HF's internal shift) |
| `train-loop` | Standard warmup + constant LR + eval + early-stop (mirrors 02_pretrain and 03_cpt) |

## 5. Evaluation during training

Two eval signals, both run every `eval_interval` steps:

- **Held-out SFT loss** — mean cross-entropy on `qa_val.jsonl`. The
  primary early-stop signal.
- **Format-followage spot check** — sample 10 prompts from `qa_val`,
  generate at T=0, and grade with a tiny in-script check: did the model
  emit `<|im_end|>` at the right place? Did the output start with the
  expected role tag? (Format failures should fall to ~0% within a few
  hundred steps; the metric is mostly a tripwire.)

End-of-run scoring against `eval/questions.yaml` happens *outside* the
train loop, in a separate `04_sft/score_eval.py` script (planned, not
scaffolded yet — depends on the judge-mode agent invocation pattern).

## 6. How to run

```bash
# one-time: stage 04 deps (overlap with stage 03 — already installed if
# you ran CPT)
.venv/bin/pip install -r 04_sft/requirements.txt

# pre-requisite: agent has produced data/sft/qa_{train,val}.jsonl
ls data/sft/qa_train.jsonl data/sft/qa_val.jsonl

# smoke test (~5 min on the SMOKE preset)
.venv/bin/python 04_sft/train_sft.py --config 04_sft/configs/sft_lora.yaml --smoke-test

# real runs (use tmux)
.venv/bin/python 04_sft/train_sft.py --config 04_sft/configs/sft_lora.yaml
.venv/bin/python 04_sft/train_sft.py --config 04_sft/configs/sft_full.yaml
```

Checkpoints land at `data/checkpoints/<run>/`. The train loop calls
`model.save_pretrained(ckpt_dir)`: LoRA writes ~5-20 MB adapter weights,
full-FT writes the full ~1.2 GB. Best-val only, atomic via `.tmp` rename.

## 7. Files

```
04_sft/
  README.md            this design doc
  train_sft.py         SFT training loop  (EXERCISE: sft-format, train-loop — both implemented)
  chat.py              inference smoke-tester (--probes / --repl)
  derive_val_split.py  one-time category-stratified val split from agent JSONL
  requirements.txt     transformers + peft + accelerate + datasets
  configs/             one YAML per ablation cell — clone, never edit in place
    sft_lora.yaml      scaffolded; not run (Stage-03 lesson extended)
    sft_full.yaml      the run reported in docs/RESULTS.md
  docs/
    RESULTS.md         training-run numbers, probe battery, failure-mode reading
```

The agent-brief link: **data generation lives at `data_gen/AGENT_BRIEF.md`**.
Stage 04 is a consumer; the agent is the producer.

Eventual `docs/RESULTS.md` and an Arknights-style write-up will follow
once training lands.
