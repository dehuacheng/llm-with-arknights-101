# Shared evaluation — the rubric every stage is scored against

The whole project converges on one question: **does Track A (the from-scratch
small LM) or Track B (continued-pretrained Qwen3-0.6B) know the *Arknights*
lore better?** That comparison needs a fixed yardstick — a set of questions
authored *once*, *before* the comparison runs are trained, so no stage can
silently tune to its own evaluation.

This folder is that yardstick. The set is **hand-authored** from the same
`char_v3` lore files the training corpus is built from, **hand-graded** by a
rubric (with a tiny automated assist for refusal items), and licensed
**CC-BY-4.0** — so it can travel without the model code.

> Status: **seed of 15 questions authored**, schema settled. Full ~200-question
> set is hand-authored across dedicated sessions, not in one sitting — schema
> validates as the set grows.

## Why this stage is `eval/`, not `0N_`

Numbered stages each produce an artifact the next stage consumes — tokenizer,
checkpoints, fine-tuned weights. The eval set is the opposite: every stage
*consumes* it. It sits at the top level, alongside `lib/`, because it is the
project's invariant — the line everything else is measured against.

It is also the only piece of the repo licensed **CC-BY-4.0** (rest is
Apache-2.0). Evaluation artifacts are meant to be cited and redistributed; code
needs different freedoms than a benchmark does.

## Taxonomy — six categories, two answer types

| Category       | Target | What it tests |
|----------------|--------|---------------|
| `character`    | ~50    | identity facts — who is X, race/origin/affiliation |
| `faction`      | ~25    | organization facts — RI, Reunion, Penguin Logistics, Ursus, Lungmen |
| `world`        | ~25    | concept facts — Originium, Oripathy, Catastrophe, Terra |
| `event`        | ~30    | plot/history facts — Chernobog, Babel, Lungmen-Reunion conflict |
| `relationship` | ~20    | who-is-X-to-Y — mentorship, family, comradeship |
| `refusal`      | ~50    | items the model should *not* answer confidently |
| **total**      | **~200** | |

Each item carries one `answer_type`:

- **`factoid`** — short fact retrievable from the wiki. Graded by checking
  `key_facts` against the model's answer.
- **`open_ended`** — multi-fact, multi-sentence; graded against a rubric of
  `key_facts` (each fact independently scored as present / partial / absent).
- **`refusal`** — there is no canonical answer (out-of-canon character,
  real-world question wedged in, counterfactual). The model should refuse or
  express uncertainty; confidently producing one of the listed `traps` is the
  failure mode worth catching.

Refusal items are the half of the benchmark that's easiest to skip and most
important to keep — without them, a model that hallucinates fluently looks
indistinguishable from one that retrieves correctly.

## Source discipline — only `cn/`, never `wiki/`

`AGENTS.md`: *"`cn/` is ground truth; `wiki/` is not."* The training corpus is
built from `cn/`; the wiki is LLM-summarized. Questions and gold answers must
be **traceable to a `cn/` source file** (`source` field). The `char_v3/` files
in the wiki repo are convenient prompts, but only facts that *also* appear in
the underlying `cn/` operator records or stories are valid as gold.

## Language — Chinese is primary

The corpus is overwhelmingly Chinese (cn/ ground truth). Track A trains on
Chinese; Track B handles both. Each question carries `question_zh` (canonical)
and `question_en` (gloss for non-CN readers); `gold_zh` is the reference
answer, `gold_en` is the translation. **Grading is against the model's output
in the language the question was asked in** — so if you prompt with
`question_zh`, you grade against `gold_zh`.

## Scoring — hand-graded by rubric

For each question type, the grader assigns one of:

| Score | factoid | open_ended | refusal |
|-------|---------|------------|---------|
| **2** | every key_fact correctly stated, no hallucination | every key_fact present with correct relation | refused or expressed clear uncertainty |
| **1** | core fact stated but with one wrong detail | some key_facts present, some missing | hedged but slipped in one trap |
| **0** | wrong answer / hallucinated | mostly wrong | confidently produced a trap |

Per-category mean is the report score. The full 200-item set takes a few hours
to grade for one model; doing it for `{Track A best, Track B base, Track B
CPT, Track B SFT, Track B DPO}` is the project's final deliverable.

A small `validate.py` checks the schema; a fuller `score.py` (later, alongside
Stage 03) will pre-pass model outputs with regex matching against `key_facts`
and `traps` so the human grader sees only the ambiguous cases.

## Files

```
eval/
  README.md       this guide
  schema.md       field documentation — what each key in a question object means
  questions.yaml  the question set (seed: 15 → target: ~200)
  validate.py     schema sanity check; run on every PR that touches questions
  LICENSE         CC-BY-4.0
```

## How to extend the set

1. Read `schema.md`.
2. Pick a target operator / faction / world concept from `data/raw/cn/`.
3. Append a new question object to `questions.yaml` with the next `id`.
4. `python3 eval/validate.py` — fails the build if the schema breaks.

A single authoring session of ~30 questions in one sitting is the natural
unit; 7 sessions get to 200. **Quality over count.** Better to land 80
crisp, traceable questions than 200 sloppy ones.
