# Stage 00 — Data preparation

Turns the raw lore corpus into a **cleaned, split** dataset that the rest of the
project consumes. This stage produces no model — it produces cleaned text and a
deterministic train/val/test split manifest.

> Status: **implemented.** Decisions below are settled. `clean_corpus.py`,
> `derive_split.py` and `apply_split.py` produce the cleaned corpus, the
> committed `split_manifest.json`, and `data/clean/splits.json` — see *Scripts*.

## Inputs

- `data/raw/cn/` — CN ground truth (461 event/story scripts, 444 operator
  profiles). The training and splitting target.
- `data/raw/wiki/` — LLM-summarized; **eval/test only**, never a training input.

## Why this stage is `00`

The from-scratch track (Track A) fits a tokenizer and a model on this corpus.
A tokenizer must be fit on the **train split only**, so the split has to be
decided before the tokenizer — hence data prep is stage `00`, tokenizer is `01`.
The final shipped model trains on *all* data; the splits exist to *run the
learning experiment*.

---

## Sub-step 1 — Corpus cleaning

The parsed `cn/` files still carry engine markup. What to do with each kind:

| Category | Found in `cn/` | Decision |
|---|---|---|
| Structural tags — `<干员名称>`, `<活动名称>`, `<章节名称>`, … | every file | **Keep**, and register as explicit special tokens in the tokenizer (see stage 01). They give the model cheap document structure. |
| Engine directives — `[HEADER(...)]`, `[name=""]`, `[animtext(...)]`, `[multiline(...)]`, `[spellsticker(...)]`, `[character]`, `[delay(...)]`, `[charslot]`, `[Background]`, `[stopmusic]`, `[CameraShake]`, `[Blocker]` | story files | **Strip the directive syntax, keep any human text.** Not stripped by the parser, contrary to earlier expectation. `[HEADER(...)]`/`[animtext(...)]`/`[spellsticker(...)]` carry text (stage titles, on-screen captions, spell text) — kept. `[multiline(name="X")]` carries the **speaker** — the line is rewritten to the canonical `X:` form. Content-less directives (`[character]`, `[delay]`, …) leave nothing — the line is dropped. Bracketed in-fiction strings like `[DWDB-221E]` are lore, not directives — kept. |
| Rich-text markup — `<i>`, `<b>`, `<color=…>`, `<p=N>`, `</>` | story files | **Strip the tags, keep the inner text.** `<p=N>`/`</>` separate on-screen panels → replaced with a space; the rest is inline emphasis → removed. |
| Placeholders — `{@nickname}`, `{@Nickname}` | stories + operators | **Replace with the in-world reader stand-in:** CN → `博士`, EN → `Doctor`, JP → `ドクター`. Our corpus is CN, so → `博士`. |
| Placeholder — `{@nbs}` | stories | Formatting artifact (non-breaking-space-like). **Strip**. |
| Speaker labels — `可萝尔:`, `赏金猎人:`, `？？？:` | story dialogue | **Keep as-is.** Real lore (who speaks). |

> BPE does **not** create special tokens on its own — it only learns frequent
> merges. To make a structural tag atomic and reserved, it is added to the
> tokenizer's special/added-token list explicitly. See `01_tokenizer/`.

## Sub-step 2 — Train/val/test split

> The final model trains on **all** data. The splits exist to run the learning
> experiment — comparing how train-set composition changes goodness of fit.
> There is no single "correct" split; each is a data point.

### The corpus has two populations

`cn/stories/` (461 files) splits cleanly in two, and they are dated differently:

| Population | Count | Dated in the game data? |
|---|---|---|
| Calendar story events — side / mini / main-story chapters | 89 | **Partly** — `story_review_table.json` gives a real `startTime` for the 71 ACTIVITY/MINI events; the 18 MAINLINE chapters carry `startTime -1` (undated, all long-released) |
| Operator records (`story_<op>_set_N`, 干员密录) | 372 | **No** — they unlock by operator trust, not by a date |

### Design: a family of nested test sets

The split is not one test set but **three growing ones**, `T0 ⊂ T1 ⊂ T2`.
Nesting is a hard requirement: because the smaller test set is always contained
in the larger one, a model trained for any variant can be scored on the common
yardstick `T0`, making goodness-of-fit comparable across variants.

Every operator's **profile stays in train** in every variant — the model can
always *fit* each operator. The test sets only ever hold out *story text*, never
an entity's core description, so there is no "entity seen only in test" gap.

| Set | What is held out | What it measures |
|---|---|---|
| **T0** (smallest) | A subset of **`<章节>` blocks** inside T1's events. | Perplexity on unseen chapters of a partly-seen event. |
| **T1** | All post-cutoff events, **whole** (4 at the pinned snapshot). | Cold generalization to unseen recent events. |
| **T2** (largest) | T1 **+ all post-cutoff operator records** (19 at the pinned snapshot). | Adds: recall of unseen lore about a **known** operator. |

### The cutoff

**Cutoff = 2026-01-01, Beijing time (UTC+8)** — the held-out pool is the most
recent slice of the game. Move it earlier to enlarge the test set.

### Construction (top-down — guarantees nesting)

1. **Date events** from `story_review_table.json` (`<entry>.startTime`) in the
   ArknightsGameData snapshot — authoritative in-game release timestamps, no git
   history needed (the `--depth 1` clone already has the table). A story event
   is *post-cutoff* iff `startTime ≥ cutoff`. Reruns are handled for free:
   `startTime` is the *original* release date (a rerun's date lives in a
   separate `remakeStartTime`). MAINLINE chapters carry `startTime -1` (undated)
   — placed pre-cutoff and reported by `derive_split.py`; at `de85be49` all 18
   are long-released, but a re-derivation must re-verify that. At snapshot
   `de85be49` the post-cutoff set is exactly **4 events**: `act48side` 雅赛努斯
   复仇记, `act49side` 辞岁行, `act20mini` 十字路口, `act51side` 人们，我们.
2. **Date operator records** using the wiki repo as a **proxy oracle.** Records
   carry no in-game date, but `arknights_lore_wiki` has full git history.
   Diffing it between the last commit before the cutoff (`2025-12-06`,
   `3cacb8f8`) and HEAD lists what was added since — **19 records** (slugs:
   botany, branch, closur, crosly, flamtl, folnic, gyuki, hadiya, headb2, ju,
   vendla, wang, wintim, wulfen, xingzh as `set_1`; hmau, orchid, popka, ray as
   `set_2`). The same diff re-derives the 4 events independently — a cross-check
   that the proxy tracks game releases for recent content.
3. The post-cutoff side is graded: `T1` = **all** post-cutoff events, whole;
   `T2` = `T1` **+ all** post-cutoff operator records (operators' *profiles*
   stay in train — only the record story is held out); `T0` = a seeded subset
   of `<章节>` blocks inside T1's events. `T0 ⊂ T1 ⊂ T2`. ✓
4. Every **pre-cutoff** file is written to the committed **pre-cutoff manifest**
   (see *Outputs*); the build consumes that manifest, not this derivation.

> Two dating sources, by necessity: events from the game data (exact), records
> from the wiki git delta (a proxy). The wiki delta also surfaces `main_17` —
> an undated MAINLINE chapter the wiki backfilled late, *not* post-cutoff game
> content; `derive_split.py` flags it and keeps to the game date for events.
> This is exactly why events are dated from the game data, not the proxy. The
> boundary commit is `2025-12-06`; the next wiki commits (`2026-01-10`) add the
> first post-cutoff content, so the boundary is clean.

### Validation set & stratification

`test` is purely temporal (the post-cutoff slice — never stratified). `val` is
a separate, seeded **10 % holdout of the pre-cutoff stories**, stratified by
**doc type only** — main / side / mini / substory / record, all derivable from
filenames — so every type is represented. Every operator *profile* stays in
train: the same "entity always fit-able" guarantee the test sets give. No
operator rarity / class / faction stratification — it would need an extra
metadata pass for marginal benefit on a learning experiment.

## Outputs

- Cleaned corpus (under `data/`, git-ignored).
- A **committed pre-cutoff manifest** (`split_manifest.json`) — the list of
  every story file dated *before* the cutoff (the immutable past), all operator
  profiles (always train), the T0 `<章节>`-block selection, and provenance
  (cutoff date, ArknightsGameData snapshot commit, wiki boundary commit
  `2025-12-06` / `3cacb8f8`). Filenames + provenance only (no lore text) —
  IP-safe, committed; both tracks read this identical file.
- `data/clean/splits.json` (git-ignored) — the concrete `train` / `val` /
  `test_t0` / `test_t1` / `test_t2` file lists `apply_split.py` derives from
  the manifest for *this* snapshot.

### The split is a date rule, not a frozen instance

The split is **train = all pre-cutoff content, test = all post-cutoff content**
— a *rule*, applied to whatever ArknightsGameData snapshot a user builds from.

It works because **the past is immutable.** The set of files released before the
cutoff never changes; new game data only ever adds *future* content. So the
thing we freeze is the past, and everything else is test:

- The pre-cutoff manifest is derived **once** — events dated from
  `story_review_table.json` `startTime`, records from the wiki git delta (the
  wiki's only role; a derivation script is committed for provenance) — and
  committed. Users never run the wiki check.
- **Rule: file ∈ pre-cutoff manifest ⇒ `train`; file ∉ it ⇒ `test`.**
- The manifest stays correct permanently. A newer snapshot's added content is
  *future* — absent from the manifest — so it falls to `test`. The test set
  grows with the game; that is correct, because new content genuinely is the
  "future" the experiment tests on.

Consequences:

- **Using the pinned snapshot is optional.** Any snapshot yields a valid split:
  `train` is identical (the past does not change), `test` is however much
  future that snapshot carries. Pin `de85be49` only for a *bit-identical* test
  set — e.g. a rigorous cross-model comparison.
- This is the deliberate inverse of an "unlisted ⇒ train" default, which would
  break the hypothesis by dumping post-cutoff (future) content into `train`.
  "Unlisted ⇒ test" is sound *only* because the frozen set is the past.
- One edge case: an upstream **rename** of an old file would send it to `test`
  by mistake; re-deriving the manifest corrects it.

## Scripts

Run from the repo root, after `tools/build_raw_data.py`:

| Script | Role | Output |
|---|---|---|
| `clean_corpus.py` | Sub-step 1 — cleans `data/raw/cn/` per the table above. Deterministic; needs only the raw corpus. | `data/clean/cn/` (git-ignored) |
| `derive_split.py` | Sub-step 2, **one-time** — dates events + records, writes the pre-cutoff manifest. Needs the sibling `ArknightsGameData` + `arknights_lore_wiki` (with git history). Repo users do **not** run this; it stays committed so the derivation is auditable. | `split_manifest.json` (committed) |
| `apply_split.py` | Sub-step 2 — applies the manifest's date rule to the cleaned corpus. Needs only the corpus + the manifest (no game data, no git). | `data/clean/splits.json` (git-ignored) |

Shared constants and the split loader live in `lib/corpus.py`; stage 01+ read
the splits through it. Stage 00 needs no third-party packages (Python stdlib).

**T0 segmentation unit** (the one design item left open): resolved to the
**`<章节>` block** — it is explicit in the cleaned text, addressable by ordinal,
and stable across snapshots, whereas HEADER stage markers live inside `<正文>`
and would need extra sub-parsing. The seeded selection (seed `20260101`, ⅓ of
each post-cutoff event's blocks) is frozen in `split_manifest.json`.
