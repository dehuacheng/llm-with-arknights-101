# Stage 03 — Continued pretraining: Qwen3-0.6B base on Arknights lore

Stage 02 (from-scratch GPT) is the *learning* arm of this project — its job
was to expose every mechanic. Stage 03 begins the **real model** arm: take
**Qwen3-0.6B base**, a 600M-parameter open-weight checkpoint already trained
on trillions of tokens, and **continue pretraining** it on the Arknights cn/
corpus. The bet is that a model that already knows how to read Chinese —
syntax, idiom, the rest of the world's facts — will pick up the Arknights
domain from 11.66M characters in a way that the from-scratch `tiny`/`small`/
`large` models couldn't.

That bet is also the experiment: **does CPT actually teach the lore, or does
it just overwrite the model's general capability?** The 2×2 ablation below is
the answer.

> Status: **planned / scaffolded.** Code and configs land here as the stage is
> implemented; this README is the design doc.

## 1. What "continued pretraining" actually is

Same training objective as the original pretraining — next-token prediction on
packed sequences, no instruction format, no labels. The only difference is the
data: instead of the open web, the model now sees mostly Arknights cn/. The
weights start from Qwen3-0.6B-Base (not Instruct — we want the raw
distribution, not the chat-tuned one), and the optimizer keeps going from
there.

This is **not SFT.** No `(question, answer)` pairs, no instruction templates.
That comes in Stage 04. Here the model is simply consuming more text, the way
its original pretraining consumed text — just from a specific domain.

## 2. The experiment — a 2×2 ablation

Two binary choices, four runs, one shared evaluation set:

|                | **No replay** | **With replay** |
|----------------|---------------|-----------------|
| **Full-FT**    | `full_ft_no_replay` — every weight updates on Arknights only | `full_ft_replay` — every weight updates on a mix of Arknights + general Chinese |
| **LoRA**       | `lora_no_replay` — rank-r adapters only, Arknights only | `lora_replay` — rank-r adapters only, mixed corpus |

- **Full-FT vs LoRA.** Full-FT updates all 600M parameters; LoRA learns a
  low-rank delta and freezes the base. Full-FT has more capacity to encode the
  domain but more capacity to *also* damage what the base already knew. LoRA
  constrains the update, which is often a better match for a small,
  narrow-domain corpus like ours.
- **Replay vs no-replay.** "Replay" is mixing a small fraction (typically
  10–30%) of general-domain Chinese text into every batch. It is the standard
  CPT countermeasure to **catastrophic forgetting** — the failure mode where
  the model gets fluent in Arknights but loses the rest of Chinese. Whether
  this matters at 0.6B with our corpus is exactly what the ablation answers.

The shared evaluation set (`/eval/questions.yaml`) is scored across all four
runs *and* the base Qwen3-0.6B-Base, against a held-out general-Chinese
perplexity check. The picture worth drawing: **gain on Arknights vs loss on
general Chinese**, four points in that plane.

## 3. Catastrophic forgetting — what to watch for

The 0.6B base has seen far more general text than our 11.66M-character corpus.
Two failure modes are possible:

1. **The model never picks up Arknights** because the LR / steps / mix is too
   timid. Eval-set scores barely move above the base.
2. **The model picks up Arknights and forgets everything else** because the LR
   / steps / mix is too aggressive. Eval-set scores rise; general Chinese
   perplexity rises with them.

The replay-mix runs are the explicit defence against (2). Full-FT runs are
*expected* to forget more than LoRA runs — that prediction is what makes the
2×2 worth running.

## 4. Data pipeline — re-tokenize cn/ with Qwen's tokenizer

Stage 02 trained from-scratch ByteBPE tokenizers (`vocab_{8k,16k,32k}`); those
are useless to Qwen3, which has its own 151,936-token tokenizer baked into the
checkpoint. `prepare_data.py` re-tokenizes the same Stage-00 splits with
Qwen's tokenizer and packs them into a single token stream under
`data/tokenized/qwen3-0.6b-base/`, mirroring the Stage-02 layout.

Two streams are prepared:

- `arknights/{train,val,test_T0,test_T1,test_T2}.bin` — Stage-00 splits,
  tokenized with `Qwen/Qwen3-0.6B-Base`'s tokenizer.
- `replay/{train,val}.bin` — a slice of **Chinese Wikipedia** for the replay
  mix. `prepare_replay.py` streams the `wikimedia/wikipedia` HF dataset
  (`20231101.zh` config — the maintained parquet mirror; the legacy
  `wikipedia` builder only ships pre-built `20220301.*` configs), shuffles
  with a fixed seed, takes the first `N_train` + `N_val` articles
  (default ~10K / 500, ≈100 MB of text → tens of millions of Qwen tokens),
  and packs them with the same `<|im_start|>` / `<|im_end|>` convention.
  Pinned so all four runs see the same replay distribution.

Packing convention is unchanged from Stage 02: documents bracketed with
`<|im_start|>` / `<|im_end|>` (Qwen's structural tokens), concatenated, sampled
at random offsets — the GPT-2 / nanoGPT convention also used in Stage 02 §2.

## 5. The training script (`train_cpt.py`)

Custom loop, **not** `transformers.Trainer` — the same reason Stage 02 is
hand-rolled: every operation is visible. PEFT and Transformers handle the
model and tokenizer; the loop is ours. The script reads one YAML config,
optionally injects LoRA, optionally interleaves the replay stream, and writes
checkpoints to `data/checkpoints/<run>/`.

Pedagogically central regions carry `EXERCISE` markers (same convention as
Stage 02):

| Block | What you implement |
|-------|--------------------|
| `lora-injection` | Build the `LoraConfig`, wrap the base model with `get_peft_model`, freeze base weights |
| `replay-mix`     | Sample one micro-batch from Arknights vs replay with mix probability `p` |
| `pack-sequences` | Concatenate tokens + sample windows (mirrors Stage 02 `get-batch`) |
| `eval-loss`      | Forward pass on val + held-out general perplexity check |

## 6. Why Qwen3-0.6B base

`AGENTS.md`: best sub-1B CJK tokenizer (151,936-vocab covering both English
and Chinese; *Arknights* is overwhelmingly Chinese), Apache-2.0, sibling
Instruct and Thinking variants for later stages (Stage 04 SFT, Stage 06
thinking distillation). The architecture (28 layers, hidden 1024, GQA 16/8,
RoPE θ=1e6, BF16 native, tied embeddings) is `Qwen3ForCausalLM` —
`AutoModelForCausalLM.from_pretrained` handles all of it.

The weights are pre-fetched to `data/models/qwen3-0.6b-base/` (git-ignored;
1.2 GB on disk).

## 7. How to run

```bash
# one-time: extend the venv with stage-03 deps
.venv/bin/pip install -r 03_cpt/requirements.txt

# one-time: re-tokenize the Arknights corpus with Qwen's tokenizer
.venv/bin/python 03_cpt/prepare_data.py

# one-time: download + tokenize the zh-wiki replay slice (needed for *_replay)
.venv/bin/python 03_cpt/prepare_replay.py

# smoke test — one of the configs, --smoke-test runs ~5 min
.venv/bin/python 03_cpt/train_cpt.py --config 03_cpt/configs/lora_no_replay.yaml --smoke-test

# a real run (use tmux — full CPT is many hours on a single 4090)
.venv/bin/python 03_cpt/train_cpt.py --config 03_cpt/configs/lora_replay.yaml

# the four-run sweep (full ablation)
for cfg in full_ft_no_replay full_ft_replay lora_no_replay lora_replay; do
    .venv/bin/python 03_cpt/train_cpt.py --config 03_cpt/configs/$cfg.yaml
done
```

Checkpoints land at `data/checkpoints/<run>/`. The `train-loop` EXERCISE
expects you to call `model.save_pretrained(ckpt_dir)`: for a `PeftModel`
that writes only the adapter (~5–20 MB) plus an `adapter_config.json`; for
a full-FT model it writes the full ~1.2 GB checkpoint. Keep best-val only —
that single overwrite is the entire on-disk footprint per run.

## 8. Files

```
03_cpt/
  README.md            this design doc
  prepare_data.py      re-tokenize Stage-00 splits with Qwen's tokenizer
  prepare_replay.py    fetch + tokenize a fixed slice of zh-wiki (replay)
  train_cpt.py         CPT training loop   (EXERCISE: lora-injection, replay-mix,
                                            pack-sequences, eval-loss, train-loop)
  requirements.txt     transformers + peft + datasets + accelerate
  configs/             one YAML per ablation — clone, never edit in place
    full_ft_no_replay.yaml
    full_ft_replay.yaml
    lora_no_replay.yaml
    lora_replay.yaml
```

Eventual `docs/RESULTS.md` and an Arknights-style write-up will follow once
training lands.
