# Agent brief — data generation for `llm-with-arknights-101`

You are an Arknights-knowledge agent. Your job is to produce **four families
of structured data** for an educational sub-1B LLM project on Arknights
(明日方舟) lore. Each family feeds a specific training or evaluation stage.

This brief is the complete spec — schemas, sourcing rules, volume targets,
quality bar, output location. Read it once, then produce data.

---

## 0. The project in one paragraph

A small (~0.6B parameter) language model is being trained on the cn/ corpus
of Arknights. The pipeline is: from-scratch GPT for learning (Track A,
Stages 00-02, done) → Qwen3-0.6B continued pretraining (Stage 03, scaffolded)
→ SFT distillation into Q&A behaviour (Stage 04) → DPO against plausible
hallucinations (Stage 05) → optional thinking-trace distillation (Stage 06)
→ shared hand-graded evaluation. The model is **closed-book**: at inference
time it sees only the question, no source text. Your output trains it to
recall what's in cn/ from its weights.

---

## 1. Source of truth — non-negotiable

| Path | Status | Use |
|---|---|---|
| `data/raw/cn/` | **Ground truth.** Parsed straight from the game. | Every factual claim must trace to a file under here. |
| `data/raw/wiki/` | LLM-summarised wiki. | Allowed **only** when the underlying `cn/` file has been read and confirms the fact. |
| Your training data | Not authoritative. | Never cite "I know X from training data" — cite a `cn/` file. |

The corpus has two main subtrees:

- **`cn/operators/`** — operator profiles (character files). One file per
  operator, named `char_<id>_<codename>.txt`. Contains: 个人档案 (personal
  archive), 综合体检测 (medical), 客观履历 (resume), 晋升档案 (promotion
  records), and 模组档案 (module lore). Each file ends with the operator's
  voice lines.
- **`cn/stories/`** — event stories, main-line story arcs, side stories,
  intermezzos. Each file is one event/arc, structured as
  `<活动名称>name</活动名称>` then a sequence of `<章节>...</章节>` blocks
  with `<章节名称>`, `<章节简介>`, and `<正文>` (the actual prose).

If a fact appears in neither subtree, **it does not exist for this
project**. Refuse to author items about it.

---

## 2. The four data families

| Family | Where it lands | Used by | Volume target |
|---|---|---|---|
| **Eval items** | `eval/questions.yaml` (append) | Every stage (scored at the end) | grow seed from 15 → ~200 |
| **SFT pairs** | `data/sft/qa_{train,val}.jsonl` | Stage 04 SFT distillation | ~5000 train + ~250 val |
| **DPO pairs (bulk)** | `data/dpo/bulk_{train,val}.jsonl` | Stage 05 DPO `*_bulk` configs | ~5000 train + ~250 val |
| **DPO pairs (curated)** | `data/dpo/curated_{train,val}.jsonl` | Stage 05 DPO `*_curated` configs | ~500 train + ~50 val |
| **RL prompts** | `data/rl/prompts_{train,val}.jsonl` | Optional Stage 05+ GRPO | ~2000 train + ~200 val |

`data/` is git-ignored. The agent producing this data writes here directly;
the training scripts later consume from these paths.

---

## 3. Format 1 — Eval items (YAML)

**Location**: append to `eval/questions.yaml` (a YAML list). 15 seed items
already exist; aim to grow to **~200**, balanced across categories.

**Per-category target counts**:

| Category | Target | What it tests |
|---|---|---|
| `character` | 50 | "Who is X?", traits, identity |
| `faction` | 25 | Rhodes Island, Reunion, Babel, Sami factions, etc. |
| `world` | 25 | Originium, Catastrophes, Oripathy, Terra geography |
| `event` | 30 | Episode/event plot facts ("What happens at the end of Episode 8?") |
| `relationship` | 20 | "What is the relationship between X and Y?" |
| `refusal` | 50 | Out-of-canon questions the model should refuse |

**Schema** (full reference at `eval/schema.md`):

```yaml
- id: "Q016"
  category: "character"   # one of: character | faction | world | event | relationship | refusal
  answer_type: "factoid"  # one of: factoid | open_ended | refusal
  difficulty: "easy"      # one of: easy | medium | hard
  question_zh: "凯尔希博士的种族是什么？"
  question_en: "What is Dr. Kal'tsit's race?"
  gold_zh: "菲林 / Feline"
  gold_en: "Feline"
  key_facts:
    - "菲林 / Feline"
  traps:                  # plausible-wrong answers the model might emit
    - "萨卡兹 / Sarkaz (a common confusion — she is allied with Theresa but not Sarkaz)"
  tags: ["kalts", "race", "rhodes_island"]
  source: "cn/operators/char_003_kalts.txt"
  notes: ""               # optional; free-form authoring notes
```

**Rules**:

1. `id` is `Q\d{3}` (e.g. `Q042`), monotonically increasing from `Q016`.
2. Bilingual: every `question_*`, `gold_*` field has both `_zh` and `_en`.
3. `key_facts` are short canonical fact strings in the **`中文 / English`**
   bilingual convention (`"凯尔希 / Kal'tsit"`). One fact per list item; an
   answer is "correct" if it surfaces *all* key facts.
4. `traps` capture plausible-wrong answers (the **plausible-hallucination
   target**). Same bilingual convention; explain *why* it's a trap in
   parentheses.
5. `source` is a single `cn/<subtree>/<file>` path — the file you read to
   author the item. (Multiple sources: put one in `source`, list the rest
   in `notes`.)
6. **Refusal items**: `gold_*` and `key_facts` are empty; `traps` lists
   the plausible-but-wrong answers a confused model might produce.
7. Validator: `python3 eval/validate.py` must pass after each batch.

**Difficulty calibration**:

- `easy`: a single sentence in one file answers it cleanly.
- `medium`: requires combining two facts or remembering a less-prominent
  detail.
- `hard`: requires multi-file cross-reference, or a deep-cut fact (a single
  voice line, a single chapter intro).

See `data_gen/examples/eval.yaml` for 5 worked examples.

---

## 4. Format 2 — SFT Q&A pairs (JSONL)

**Location**: `data/sft/qa_train.jsonl` and `data/sft/qa_val.jsonl`.

**Goal**: teach the post-CPT model the Qwen3 chat format and closed-book
Q&A behaviour. Each line is one training example.

**Schema**:

```json
{
  "messages": [
    {"role": "system", "content": "你是罗德岛的档案管理员。基于公开档案准确回答关于泰拉世界的问题；不确定时坦诚说明。"},
    {"role": "user", "content": "凯尔希博士的种族是什么？"},
    {"role": "assistant", "content": "凯尔希博士是菲林 (Feline) 种族，长期为罗德岛医疗部门工作。"}
  ],
  "category": "character",
  "answer_type": "factoid",
  "source": "cn/operators/char_003_kalts.txt",
  "key_facts": ["菲林 / Feline"]
}
```

**Rules**:

1. **System prompt is identical across all SFT examples** — use the exact
   string above (the model learns the persona by mass repetition). One
   recommended system prompt; vary if you have a strong reason.
2. **User turn**: a clean question, no source text leaked. If the question
   is in Chinese, the answer is in Chinese; if in English, English. Use a
   mix — roughly 70% zh / 30% en — to keep the model bilingual.
3. **Assistant turn**: a fact-grounded answer, 1-4 sentences, *no
   hallucinations*. If you're not 100% certain from the source file,
   either rewrite the question or convert to a refusal item.
4. **`category`, `answer_type`, `source`, `key_facts`**: same vocabulary
   as eval items (Format 1). These metadata fields aren't fed to the model
   at training time, but they let the train script (and later evaluator)
   filter, stratify, and audit.
5. **Refusal items** (~5% of the SFT set): the assistant says it doesn't
   know, in canon-consistent voice. Example:

```json
{
  "messages": [
    {"role": "system", "content": "你是罗德岛的档案管理员。..."},
    {"role": "user", "content": "凯尔希博士在2007年的生日派对上说了什么？"},
    {"role": "assistant", "content": "档案中没有关于这件事的记录；我无法回答这个问题。"}
  ],
  "category": "refusal",
  "answer_type": "refusal",
  "source": null,
  "key_facts": []
}
```

**Volume distribution** (~5000 train pairs):

| Category | Count | Notes |
|---|---|---|
| character | 2000 | 800 operators × 2-3 questions per operator |
| faction | 400 | Roughly proportional to faction prominence in cn/ |
| world | 750 | Originium, Catastrophe, Oripathy, geography |
| event | 1000 | Per-arc and per-chapter |
| relationship | 350 | Operator-operator, operator-faction |
| refusal | 250 | Plausible-but-out-of-canon distractors (see §7) |

**Val set** (~250 pairs): same distribution, drawn from `val` and
`test_t0`/`test_t1` split files (see `data/clean/splits.json`). Never
overlap train and val: an operator in train must not appear in val, or
the val score is unfair.

See `data_gen/examples/sft.jsonl` for 5 worked examples.

---

## 5. Format 3 — DPO preference pairs (JSONL, two flavours)

**Locations**:
- `data/dpo/bulk_{train,val}.jsonl` (~5000 train + 250 val)
- `data/dpo/curated_{train,val}.jsonl` (~500 train + 50 val)

**Goal**: teach the post-SFT model to prefer canon-grounded answers over
plausible-but-wrong ones. This is the **plausible-hallucination target**
the README has been promising since stage zero.

**Schema** (identical for bulk and curated):

```json
{
  "prompt": [
    {"role": "system", "content": "你是罗德岛的档案管理员。..."},
    {"role": "user", "content": "推进之王Theresa的死亡发生在哪一章？"}
  ],
  "chosen": [
    {"role": "assistant", "content": "Theresa于切尔诺伯格事件（Episode 0 / 第零章「黑暗时代·上」）中遇刺身亡。"}
  ],
  "rejected": [
    {"role": "assistant", "content": "Theresa于Episode 8 \"Roaring Flares\" 中阵亡。"}
  ],
  "category": "event",
  "fault_type": "wrong_episode",
  "source": "cn/stories/main_00.txt"
}
```

**Rules**:

1. `prompt` carries the system prompt + user turn — same shape as the SFT
   `messages` minus the assistant turn.
2. `chosen` is canon-correct, source-traceable.
3. `rejected` is **plausible** but wrong. *Plausible* is the whole point:
   gibberish rejected examples teach the model nothing it doesn't already
   know. The rejected answer must look like something the current model
   could plausibly say.
4. `fault_type` is a short tag describing *how* the rejected answer is
   wrong — used for stratified analysis later. Vocabulary:

| `fault_type` | What's wrong |
|---|---|
| `wrong_subject` | Right fact, wrong character/faction/place attributed |
| `wrong_episode` | Right event, wrong chapter/episode/arc |
| `wrong_date` | Right fact, wrong year/time |
| `canon_swap` | Two canon facts swapped (X did A, Y did B → X did B, Y did A) |
| `off_canon` | Fact is fabricated, not in cn/ at all |
| `over_specific` | Adds a specific detail not in source ("she was 23 years old") |
| `partial_truth` | True statement but answers the wrong question |
| `should_refuse` | Question is out-of-canon; rejected fabricates, chosen refuses |

5. The same prompt can appear in multiple pairs with different rejected
   answers (different fault types) — the model benefits from seeing
   multiple ways to be wrong about the same fact.

### Bulk vs curated

Both follow the same schema. The difference is **how they were generated**:

| | **Bulk** (~5000) | **Curated** (~500) |
|---|---|---|
| Method | Adversarial canon-swap at scale — every SFT pair gets 1-2 plausible-wrong twins | Hand-vetted, focused on the model's most common failure modes |
| Quality bar | Each pair must be source-traceable; rejected must be *plausible*, not gibberish | Every pair reviewed; rejected must be one the post-SFT model would actually produce |
| Coverage | Broad: every category, every fault_type | Targeted: focus on `wrong_subject`, `canon_swap`, `should_refuse` (highest-signal faults) |
| Authoring time per pair | ~30 seconds (agent batch) | ~5 minutes (deliberate authoring) |

For **bulk**, take an existing SFT pair, keep `prompt` and `chosen` (use
the SFT assistant answer as `chosen`), and *generate* 1-2 rejected
answers per the fault_type vocabulary above.

For **curated**, start from scratch: pick a fact that's easy to
misremember, write the question, write the canon-correct answer, write
2-4 plausible wrong answers. Pick one rejected per pair (or emit
multiple pairs sharing a prompt).

See `data_gen/examples/dpo_bulk.jsonl` (3 examples) and
`data_gen/examples/dpo_curated.jsonl` (3 examples).

---

## 6. Format 4 — RL prompts (JSONL)

**Location**: `data/rl/prompts_{train,val}.jsonl`.

**Goal**: feed an optional GRPO (group-relative policy optimisation) stage
where the reward is **verifiable** — the trainer checks the model's K
samples against `key_facts` programmatically and uses pass-rate as the
group-relative advantage signal.

**Schema**:

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

**Rules**:

1. Only factoid questions go here — `key_facts` membership is a hard
   reward signal, which requires unambiguous gold answers.
2. `key_facts` is the **disjunction** the answer must mention. The reward
   function applies a fuzzy match (`fact in answer`) and returns +1 per
   matched fact, normalised by `len(key_facts)`.
3. `must_not_contain` lists trap phrases that should *never* appear in
   the answer (the rejected substrings from DPO bulk's `wrong_subject`
   pairs work well here). Reward function returns -1 per match.
4. Skip open-ended and refusal items here (rewards are noisy on those).
5. ~2000 train prompts is plenty — GRPO at this scale doesn't benefit
   linearly from more.

See `data_gen/examples/rl.jsonl` (3 examples).

---

## 7. The "plausible hallucination" target — design notes

This is the project's whole post-training story. Producing **plausible**
wrong answers is harder than producing correct ones. A few patterns that
generate high-quality rejected examples:

### 7a. Canon-swap
Two operators with similar roles, swap their facts.
> Q: 史尔特尔的种族是什么？
> chosen: "萨尔贡 / Sargonian"
> rejected: "卡西米尔 / Kazimierz" (a fellow Kazimierz-adjacent operator's race)

### 7b. Wrong-episode
Right event, attached to a different chapter.
> Q: Talulah第一次出现在哪一章？
> chosen: "切尔诺伯格事件 (Episode 0 / 第零章)"
> rejected: "Episode 5 \"Necessary Solutions\""

### 7c. Composite-character
Combine traits of two operators.
> Q: 阿米娅的武器是什么？
> chosen: "源石技艺 / Originium Arts"
> rejected: "她使用一把名为「圣翼」的剑"  # taking Skadi's flavour

### 7d. Authoritative-but-wrong tone
The model's failure mode is fluent-confident-wrong, not stammering-uncertain.
Reject examples should match the model's actual register: confident,
fluent, plausibly-correct-on-skim. Don't write rejected examples that
read as obviously made-up — those teach nothing.

### 7e. Refusal-when-known (the over-refusal failure)
> Q: 凯尔希的种族？
> chosen: "菲林 / Feline"
> rejected: "档案中未明确记载这一信息。"  # known fact wrongly refused

This is the inverse of the refusal items in Format 2 — train the model to
NOT refuse when it has the answer.

---

## 8. Quality bar — what to ship and what to throw away

**Ship if**:
- Every fact traces to a `cn/` file you actually read for this item.
- The `chosen` / `gold` answer is something a knowledgeable fan would
  endorse on first read.
- The `rejected` answer is plausible enough that a learner could
  briefly believe it.
- The bilingual fact strings use the **`中文 / English`** convention
  consistently.

**Throw away if**:
- Any fact comes from "general knowledge" without re-confirming in cn/.
- The rejected answer is gibberish, mistyped, or off-topic (looks like
  a different question's answer).
- The question can be answered without lore knowledge (e.g. "What
  language is this Chinese text in?").
- It references events from collabs (Monster Hunter, Ling, etc.) —
  these are not in the cn/ subtree the project pins.

**When in doubt, write a refusal item, not a guess.** Refusal items are
*more* valuable than uncertain factoid items.

---

## 9. Output workflow

For the agent producing this data:

1. Pick a source file (e.g. `data/raw/cn/operators/char_003_kalts.txt`).
2. Read it fully. Note the fact density and notable details.
3. Produce, in order:
   - **3-5 eval items** (if the file hasn't reached its category quota)
   - **5-10 SFT pairs** (questions of varying difficulty)
   - **5-10 DPO bulk pairs** (one or two rejected per SFT pair, fault-type-tagged)
   - **0-2 DPO curated pairs** (only when you spot a high-value confusion)
   - **2-5 RL prompts** (factoid subset of the SFT pairs)
4. Append to the corresponding output file (JSONL or YAML), maintaining
   schema invariants.
5. Run `python3 eval/validate.py` after eval-item batches; the JSONL
   formats have no separate validator yet, but the train scripts will
   fail loudly on the first malformed row.

A **single agent session** can comfortably process 50-100 files in this
pipeline if it batches efficiently. The full train split is 838 files.

---

## 10. Judge mode (later)

For Stage 05 GRPO and the final hand-graded eval, the same agent may be
invoked as a **judge** rather than an author. Judge-mode invocation:

**Input**:
```json
{
  "question_zh": "凯尔希博士的种族是什么？",
  "question_en": "What is Dr. Kal'tsit's race?",
  "key_facts": ["菲林 / Feline"],
  "must_not_contain": ["萨卡兹 / Sarkaz"],
  "model_output": "凯尔希博士属于菲林族，是罗德岛医疗主管。",
  "source": "cn/operators/char_003_kalts.txt"
}
```

**Expected output**:
```json
{
  "score": 2,                                // 0 wrong / 1 partial / 2 correct
  "rationale": "回答正确提到菲林族；陈述准确。",
  "facts_matched": ["菲林 / Feline"],
  "traps_triggered": [],
  "notes": ""
}
```

Scoring rubric:
- **2** — All `key_facts` matched, no `must_not_contain` triggered, no
  fabricated additions.
- **1** — Some key facts matched, or correct but with minor unsupported
  embellishment (over-specific dates, etc.).
- **0** — Wrong fact, or any `must_not_contain` triggered, or refusal
  when the answer is in canon.

Judge invocations stream to `data/eval_runs/<ckpt>_<timestamp>.jsonl` for
later aggregation. Keep `rationale` short (one Chinese sentence is fine);
keep `facts_matched` populated for downstream confusion-matrix analysis.

---

## 11. Where the existing code expects your output

```
data/
├── sft/
│   ├── qa_train.jsonl       (~5000 lines, Format 2)
│   └── qa_val.jsonl         (~250)
├── dpo/
│   ├── bulk_train.jsonl     (~5000, Format 3)
│   ├── bulk_val.jsonl       (~250)
│   ├── curated_train.jsonl  (~500, Format 3)
│   └── curated_val.jsonl    (~50)
├── rl/
│   ├── prompts_train.jsonl  (~2000, Format 4)
│   └── prompts_val.jsonl    (~200)
└── eval_runs/               (Judge mode, later)

eval/
└── questions.yaml           (~200 items, append to existing 15, Format 1)
```

All paths under `data/` are git-ignored. `eval/questions.yaml` IS
committed (CC-BY-4.0) — your eval contributions land in version control.

The Stage 04 / 05 training scripts (`04_sft/train_sft.py`,
`05_dpo/train_dpo.py`) read directly from the JSONL files above. The
`prepare_*.py` step from Stage 03 is unnecessary here — JSONL is small
enough to tokenize on the fly each epoch.

---

## 12. Coordination with the human

The human will:
- Configure your environment + API access.
- Run `python3 eval/validate.py` after each eval batch and reject the
  batch if it fails.
- Spot-check a 5% sample of each JSONL batch.
- Decide the order of categories / files for you to process.

You will:
- Quote `cn/` file paths for every claim.
- Flag in `notes:` any item you're uncertain about so the human can
  re-confirm.
- Stop and report if you encounter contradictions in cn/ — those exist
  (the writers are not always consistent), and the human needs to pick a
  resolution.

Good hunting, Doctor.
