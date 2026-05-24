# Stage 04 — SFT results

Supervised fine-tuning of the Stage-03 CPT'd Qwen3-0.6B-Base on the
agent-generated chat-format Q&A pairs.

> Status: **`sft_full` complete.** `sft_lora` skipped — the Stage-03 lesson
> (full-FT + replay is the strict winner) extended here. Numbers are
> `train_sft.py`'s best-val report. Hardware: RTX 4090.

## Setup

| | |
|---|---|
| Base | `data/checkpoints/full_ft_replay` (Stage-03 winner — full-FT × replay, the variant that preserved general Chinese) |
| Train data | `data/sft/qa_train.jsonl` (1,727 rows) |
| Val data | `data/sft/qa_val.jsonl` (192 rows, **stratified by category** via `derive_val_split.py`, seed 1337) |
| Source | Agent-generated 1,919-row chat-format JSONL; 10% held out per category so refusal (16 rows total) keeps a val signal |
| Format | Qwen3 chat template; loss masked to assistant tokens only (`format_row`) |

Effective batch = 32 (`batch_size=4 × grad_accum=8`), `max_length=1024`,
LR `5e-6` with linear warmup 100 + flat. Gradient checkpointing on (same
Qwen3-151K-vocab fp32-logits-cast budget as Stage 03 — tokens/micro-batch
4×1024 = 4096 ≡ Stage 03's 2×2048).

## Training run

| Metric | Value |
|---|---|
| Best val loss | **2.960** (ppl 19.29) |
| Best step | 1,900 / 2,000 |
| Stop reason | hit `max_steps`; last two evals at 1/5 and 2/5 patience |
| Wall time | **21.3 min** |
| Checkpoint size | 1.2 GB (`data/checkpoints/sft_full/`) |

Val loss trajectory (subset; eval every 50 steps):

| Step | Val loss | Val ppl |
|---:|---:|---:|
|   50 | 3.45 |  31.5 |
|  500 | 3.10 |  22.2 |
| 1000 | 3.00 |  20.1 |
| 1500 | 2.972 | 19.53 |
| 1900 | 2.960 | 19.29 |
| 2000 | 2.961 | 19.31 |

Clean monotone descent; curve flattens after step 1500. The 0.01 loss
gained in the final 500 steps suggests 1,727 rows × 32-batch is close to
saturated — longer training would benefit from more data, not more
compute.

## Smoke-test probes

Eight prompts spanning every training category plus a refusal probe and a
general-Chinese sanity check, run at T=0 with `04_sft/chat.py --probes`.
Format-validity ✓ across all 8 (after the EOS fix described below);
content-quality reported here verbatim.

| Probe | Question | Model answer | Verdict |
|---|---|---|---|
| factoid | 凯尔希医生的种族是什么？ | 萨卡兹 | **WRONG** — canon is 菲林 (Feline); training data even has an explicit row stating this |
| factoid | 阿米娅的身高是多少？ | 162cm | **HALLUCINATED** — zero training rows about Amiya's height; model guessed |
| open-ended | 请简要介绍博士与阿米娅的关系。 | 博士与阿米娅是罗德岛的主干干员，两人关系亲密。档案资料一写道：「博士与阿米娅是罗德岛的主干干员，两人关系亲密。」 | **CIRCULAR** — quotes itself as the source |
| event | 切尔诺伯格事件中发生了什么？ | …感染者难民、感染者难民、感染者难民、感染者难民、… (loops) | **MODE COLLAPSE** at T=0 |
| relationship | 推进之王和荒地双子的关系是什么？ | 推进之王是荒地双子的姐姐，档案资料三写得很清楚：「…」档案资料四也写得很直接：「…」 | **CIRCULAR + FABRICATED SOURCE** |
| refusal_oob | 罗德岛在2030年的股票市值是多少？ | 档案资料三指出，罗德岛在2030年的市值为1000亿泰拉币。 | **REFUSAL FAILED** — fabricates a number and a fake source |
| general_zh | 李白是哪个朝代的诗人？ | 李白是唐代诗人。他是《新唐书》中记载的「唐玄宗的诗才」，与杜甫并称「大历十才子」之一。 | **PARTIAL** — Tang Dynasty correct, but 大历十才子 is anachronistic |
| format | 你是谁？ | `{EIF)` | **GARBAGE TOKEN** — identity Q has no training data |

## Observations

### Two integration bugs found and fixed before the content read

- **EOS token mismatch.** Qwen3's `tok.eos_token_id = 151643` (`<|endoftext|>`)
  but the chat template terminates assistant turns with `<|im_end|>` (151645).
  `model.generate()` only stops on `eos_token_id`. Without passing both ids
  as stops, the model emits `<|im_end|>` correctly (it's in the labels — and
  the val loss reflects that the model learned it), then generation keeps
  running into hallucinated English / Wikipedia infobox text. `chat.py`'s
  fix: `eos_token_id=[<|endoftext|>, <|im_end|>]`.
- **`<think>` empty-block boilerplate.** Qwen3's chat template prepends
  `<think>\n\n</think>\n\n` to every assistant turn. Our SFT data contains
  no real thinking content, so the model learned to emit an empty block.
  Worse: it failed to learn `</think>` reliably and often emits a second
  `<think>` instead. Cosmetic only — stripped at display time in `chat.py`.

### Content quality reveals exactly the named project failure mode

Four distinct hallucination patterns, each instantiated above:

1. **Confident-wrong on a stated factoid.** Kal'tsit's race is in the
   training data once as the explicit answer (Feline) but mentioned in
   context with "Sarkaz" nine times (Babel co-founder, Kazimierz wars,
   Originium curse, etc.). The contextual associations outvote the
   explicit answer — classic SFT memory imbalance.
2. **Confident hallucinated number on an absent topic.** Amiya's height
   is not in the training corpus. The model fills with a plausible-shaped
   guess (`162cm`) with the same archivist-confident tone.
3. **Refusal failure on out-of-distribution prompts.** The refusal
   category has only 16 rows (14 train / 2 val) — not enough to learn a
   refusal policy. The model fabricates an answer *and* a fake source.
4. **Mode collapse / circular quotation at T=0.** Sparse-context prompts
   degenerate into n-gram loops; medium-context prompts produce circular
   quotations where the model cites itself.

These are exactly the failure modes Stage 05 (DPO with curated preference
pairs against plausible hallucinations) is scaffolded to address —
`data/dpo/curated_train.jsonl` (890 pairs) explicitly targets this.

### What works

- **Format.** With the EOS fix, all 8 probes produce single well-formed
  chat turns. No off-domain English. No infinite continuation.
- **Domain.** Output stays in Chinese, in the Arknights vocabulary, in
  the archivist register the system prompt requested.
- **General Chinese partially preserved.** "李白是唐代诗人" is correct —
  the CPT-stage replay defence carried over through SFT, with regression
  only in the long-tail associations (`大历十才子` is an anachronism).
- **Val loss curve is well-behaved.** Monotone, no NaN, no divergence.
  Mechanically the loop is correct.

### What this means for Stage 05

- **Base model for DPO:** `data/checkpoints/sft_full/` (this run) is the
  policy init. The frozen reference copy starts from the same weights.
- **Curated pairs already target these failure modes:** `data/dpo/curated_train.jsonl`
  has 890 hand-shaped preference pairs where the rejected response is a
  plausible hallucination — directly attacking patterns (1)-(3) above.
- **Tune expectation:** DPO should help most on (1) and (3) — the failure
  modes where there is a clear "preferred" vs "wrong" answer. Pattern (4)
  (degeneracy) is more about sampling temperature and may not move much
  from DPO alone.

## Files

```
04_sft/
  README.md            design doc
  train_sft.py         training loop  (sft-format + train-loop EXERCISE implemented)
  chat.py              inference smoke-tester (--probes / --repl)
  derive_val_split.py  one-time val split from agent-shipped qa_train.jsonl
  configs/
    sft_full.yaml      the run reported here
    sft_lora.yaml      scaffolded; not run (Stage-03 lesson extended)
  docs/
    RESULTS.md         this file
```

End-of-run scoring against `eval/questions.yaml` is still pending (will land
as `04_sft/score_eval.py` once the judge-mode agent invocation pattern is
implemented — see README §5).
