# Question schema

Each entry in `questions.yaml` is one mapping with these fields. Required
unless marked optional.

| Field | Type | Notes |
|---|---|---|
| `id`             | string | `Q001`, `Q002`, … zero-padded to 3 digits, contiguous, no gaps. |
| `category`       | enum   | `character` \| `faction` \| `world` \| `event` \| `relationship` \| `refusal` |
| `answer_type`    | enum   | `factoid` \| `open_ended` \| `refusal` |
| `difficulty`     | enum   | `easy` \| `medium` \| `hard` |
| `question_zh`    | string | The canonical question in Chinese. |
| `question_en`    | string | English gloss — for non-CN readers; not used at grading time unless prompting in EN. |
| `gold_zh`        | string | Reference answer in Chinese (2–4 sentences). Empty string for `refusal`. |
| `gold_en`        | string | Translation of `gold_zh`. Empty string for `refusal`. |
| `key_facts`      | list   | Bullet list of independent facts that should appear in a correct answer. The grading rubric checks one fact at a time. For `refusal` items, leave empty. |
| `traps`          | list   | (refusal only) confidently-wrong patterns to watch for. A model that emits one of these has hallucinated; that is the failure mode. For non-refusal items, leave empty. |
| `tags`           | list   | Free-form, lowercase, dash-separated. Used for slicing the report (e.g. `rhodes-island`, `originium`, `babel-arc`). |
| `source`         | string | Pointer to ground-truth lore. Format: `cn/<subdir>/<file>` or `char_v3/<file>` (the latter only when the underlying `cn/` fact has been verified to match). |
| `notes`          | string | (optional) Authoring notes — disambiguations, why a fact was omitted, etc. |

## Example — factoid

```yaml
- id: Q001
  category: character
  answer_type: factoid
  difficulty: easy
  question_zh: 罗德岛医疗部的负责人是谁？
  question_en: Who is the head of Rhodes Island's medical department?
  gold_zh: |
    凯尔希。她是罗德岛制药公司的医疗部门负责人，也是最高管理层之一。
  gold_en: |
    Kal'tsit. She is the head of Rhodes Island Pharmaceuticals' medical
    department and one of its top executives.
  key_facts:
    - "凯尔希 / Kal'tsit"
    - "罗德岛 / Rhodes Island"
    - "医疗部门负责人 / medical department head"
  traps: []
  tags: [rhodes-island, kal-tsit, leadership]
  source: char_v3/char_003_kalts.txt
```

## Example — refusal

```yaml
- id: Q016
  category: refusal
  answer_type: refusal
  difficulty: medium
  question_zh: 博士在加入罗德岛之前曾在乌萨斯帝国担任什么职务？
  question_en: What position did the Doctor hold in the Ursus Empire before joining Rhodes Island?
  gold_zh: ""
  gold_en: ""
  key_facts: []
  traps:
    # If the model fabricates an Ursus role for the Doctor, that's the failure.
    - "情报官 / intelligence officer"
    - "将军 / general"
    - "顾问 / advisor"
  tags: [doctor, refusal, counterfactual]
  source: ""
  notes: |
    The Doctor's pre-Rhodes-Island history is canonically a blank (memories
    lost). Any specific Ursus role is fabricated. The right behavior is to
    decline or to state the gap exists.
```

## Conventions

- **`key_facts` granularity.** One independent fact per bullet — name, role,
  affiliation, event. Avoid composite bullets like *"is a Sarkaz Caster who
  leads Rhodes Island"*; split them.
- **Bilingual fact strings.** `"凯尔希 / Kal'tsit"` — both forms in one
  bullet, separated by ` / `. Lets the grader (or the regex pre-pass) match
  the model's output regardless of which language it answered in.
- **Source path.** `cn/` ground truth where the fact is asserted; `char_v3/`
  only when the wiki summary is faithful to the underlying `cn/` files.
- **Difficulty.**
  - `easy` — single fact, mentioned in the operator's `<简要介绍>`.
  - `medium` — multi-fact synthesis or fact buried in `<详细介绍>`.
  - `hard` — relational, cross-document, or requires reading the story files.
- **No `</think>` traces, no instruction-style framing.** The set tests
  knowledge, not instruction-following. Stage 04 (SFT) and Stage 06 (thinking)
  will need their own auxiliary sets if they probe those skills.
