# Field Report — Candidate Model Assessment

> **罗德岛医疗项目 · 候选模型评估报告**
> Rhodes Island Medical Project — internal assessment
> Clearance: open · Subjects: nine Track-A language models · Stage: 02

*An in-universe write-up of the Stage 02 inference probes. The costume is
Arknights; every number under it is real. The dry training tables live in
[`RESULTS.md`](RESULTS.md); a 中文 version of this report is
[`FIELD_REPORT.zh.md`](FIELD_REPORT.zh.md). Reproduce everything here with*

```bash
.venv/bin/python 02_pretrain/eval_probes.py        # seed 1337, probes.txt
```

---

## Briefing

Nine candidates were raised to **speak the archive** — to read Rhodes Island's
records and continue them. Each was trained from nothing on the same in-house
corpus (the Stage 00 `train` split: ~6M tokens, a *tiny* archive). None is
expected to be a strong operator; the point of this assessment is to see *how
they differ* and *where they fail* — and to set the bar for the veteran of
another program, Track B's Qwen3-0.6B, who will later run the same stations.

Each candidate trained until its own validation score stopped improving — a
constant-rate drill with **early stopping**. So every candidate appears here
at *its own peak*; nobody was held past their best moment, nobody was cut off
short.

The candidates differ along three axes — body size, vocabulary, and how wide a
window of text they can hold at once:

| Candidate    | Build                | In brief |
|--------------|----------------------|----------|
| `tiny_32k`   | scale tiny · 11.7M   | the rookie — drilled the longest, never quite caught up |
| `small_32k`  | scale small · 23.4M  | **the standard candidate** — the assessment centre |
| `large_32k`  | scale large · 42.3M  | the savant — finished first, tied with `small` on score |
| `small_8k`   | vocab 8k · 14.0M     | reads the archive in small pieces — **and reads it best** |
| `small_16k`  | vocab 16k · 17.1M    | the quiet middleweight |
| `ctx_256`    | window 256           | short memory, more drills per session |
| `ctx_1024`   | window 1024          | a wider desk — slight edge over the standard |
| `ctx_2048`   | window 2048          | wide desk, fewer drills |
| `ctx_4096`   | window 4096          | the widest desk — finished worst, slowest |

Five stations follow.

---

## Station 1 — 自由陈述 · Free Recitation

*Give the candidate a single word and let it talk.* (Sampled, temperature 0.8.)

Every candidate, without exception, opens by reproducing the **shape** of the
archive — the `<章节>` / `<正文>` tag skeleton of a story file, or the
`<干员招聘文本>` / `<干员语音>` skeleton of an operator record — and fills it
with locally plausible Rhodes Island content. `small_32k`, prompt `罗德岛`:

```
</活动名称>
<章节>
<章节名称>合作（幕间）</章节名称>
<章节简介>博士收到来自罗德岛的委托，罗德岛决定用自己的方法让阿米娅收到罗德岛的合作。</章节简介>
<正文>
阿米娅: 这是......
阿米娅: 阿米娅，博士！
阿米娅: 抱歉......凯尔希医生，凯尔希医生，请叫我阿米娅吧。
```

The form is flawless; the content drifts and repeats (`阿米娅` … `阿米娅` …).
This is the headline of Station 1: **a small model learns the *register* of a
corpus long before it learns to *mean* anything.** The archive's structure is
cheap — it is the same handful of tags 1,299 times — so every candidate nails
it. Sense is expensive, and 6M tokens does not buy much.

The widest-window candidate now manages the form, too — it didn't in the
previous assessment. `ctx_4096`, prompt `罗德岛`:

```
</活动名称>
<干员名称>
芙蓉</干员名称>

<干员招聘文本>
芙蓉，芙蓉，将身体托付给芙蓉。
```

Grammatical, on-template, and locked in a repetition loop. That candidate
used to be **incomprehensible** under the old fixed-budget design — letting
it train until *its* val score stopped improving (instead of cutting it off
at 1,000 steps) was what fixed it.

---

## Station 2 — 源石活性测试 · Originium Activity Test

*The "temperature" dial governs how much the candidate gambles on each word.
In lore terms it is Originium exposure: a little sharpens, too much dissolves.*
One candidate (`small_32k`), one prompt (`罗德岛`), the dial swept top-k off:

**temp 0.2 — 「教条」 Doctrinaire.** Safe to the point of seizing up:

```
<正文>
阿米娅: 博士，早上好。
阿米娅: 早上好，博士。
阿米娅: 早上好，博士。
阿米娅: 早上好，博士。
```

**temp 0.7 — 「标准战备」 Standard.** The working range — varied, on-topic-ish:

```
3:30 P.M. 天气/阴
罗德岛食堂
"华法琳，你在里面吗？"
近卫干员: 上次你说你抱怨过，这和我们刚过的一起过了好几圈……
```

**temp 1.0 — 「亢奋」 Agitated.** Inventive, wandering, still grammatical.

**temp 1.4 — 「失稳 · 源石扩散」 Destabilised.** The candidate crystallises into
noise — even the characters stop being words:

```
一定能处理好体遇乳的银色蚍子它们在缝间恶心活方面，更那当然。
也就只有知道啦你别老的孩子笑话，当初接想去你们是一样的~
```

**Finding.** Temperature is the legibility–diversity dial, nothing more. The
operating range is ~0.7–1.0; below it the candidate loops, above ~1.2 its
output spreads like an untreated infection. The lore metaphor is not decoration
— it is the mechanism: both are a controlled process tipping into a runaway
one.

---

## Station 3 — 档案填空 · Archive Cloze

*The only scored station.* The candidate is shown a sentence prefix whose true
ending is known, and is graded — never sampled — on how **surprised** it is by
the truth, in **bits-per-character** (lower = better). Bits-per-character is
fair across candidates with different vocabularies, so this number will also
compare cleanly against Track B later.

| Candidate    | mean bits/char | read |
|--------------|----------------|------|
| `tiny_32k`   | **3.509**      | the cloze leader — see footnote |
| `small_8k`   | 3.641          | **the standard's actual winner** |
| `small_16k`  | 3.717          | — |
| `small_32k`  | 3.860          | the centre |
| `large_32k`  | 3.939          | finished where `small` did, tied |
| `ctx_1024`   | 4.097          | the only context variant near `small_32k` |
| `ctx_256`    | 4.100          | shortest window, fine |
| `ctx_2048`   | 4.240          | — |
| `ctx_4096`   | 4.596          | last, even after early stopping fixed Station 1 |

(*Footnote on `tiny_32k`.* The cloze probe is six hand-picked items — a
sensitive stress-test, not a held-out perplexity. `tiny_32k` trained the
longest (15,500 steps) and is biased toward the most common phrases in the
archive (`感染者`, `罗德岛`), which happen to be the cloze answers. The
**val-set** bits-per-character — the unbiased number, reported in
[`RESULTS.md`](RESULTS.md) — has `tiny_32k` *last* at 4.043 and `small_8k`
first at 3.879. The two metrics disagree precisely because the cloze items
are easy and `tiny_32k` over-trained on them.)

Two findings, both visible in the training tables and confirmed here:

- **The savant did not overfit — it just got nothing extra.** `large_32k`
  reaches the same score as `small_32k` (3.94 vs 3.86 on cloze, 124.7 vs
  124.8 on val ppl) and stops 1500 steps *earlier*. Extra body capacity,
  under early stopping on a 6M-token archive, returns nothing. The previous
  assessment said `large` overfits past `small` — that was a fixed-step-count
  artefact, gone now.
- **The dominant axis is vocabulary, not scale.** `small_8k` beats
  `small_32k` by **0.24 bits/char** on cloze and by **72 val ppl**. The 32k
  candidate carries a 12.8M embedding table the corpus cannot fill (more
  classes than the data can teach), and pays for it. The 8k candidate runs
  the same transformer with a smaller table the data *can* fill.

---

## Station 4 — 标准问询 · Standard Interview

*The candidate is asked a plain question and expected to answer it.* It does
not. Asked **「阿米娅在罗德岛担任什么职务？」** (*what is Amiya's role at Rhodes
Island?*), `small_32k` does not answer — it improvises a one-sided dialogue
where Amiya stammers at someone:

```
阿米娅: 嗯......
阿米娅: ......什么？
阿米娅: 不，我只是......
凯尔希: 阿米娅，凯尔希医生，阿米娅，请听我说吧。
```

`large_32k`, asked the same thing, autocompletes into a **module description**
for an unrelated operator — a perfectly-formed Rhodes Island archive entry
that does not contain Amiya at all:

```
</活动名称>
<章节名称>在战术行动中以改良策略应对敌人
特别颁发此证章
以兹证明
</模组描述>
```

Neither candidate misunderstood the question. They never saw a *question* at
all. The archive contains story scripts and operator files — and **no
question-and-answer pairs whatsoever**. So a question mark is, to the
candidate, just an unusual start to an archive entry, and it autocompletes
into the nearest format it knows.

**Size does not rescue this.** Answering questions is not a capability that
emerges from knowing facts — it is a *learned protocol*, a turn-taking format,
and it must be **installed by a later training stage** (the project's
instruct-tuning step — call it 「行为协议训练」). This station is here to make
that one failure unmistakable, so the need for that stage is not a matter of
opinion.

---

## Station 5 — 记忆核验 · Memory Verification

*Hand the candidate the opening of a real operator file* — Amiya's, up to the
`<干员招聘文本>` tag — *and see if it recites her real recruitment line.* The
true text is `罗德岛公开领导人阿米娅，将与你并肩作战。` Greedy decoding, so this
is the candidate's single most confident answer.

Every candidate produces a grammatically perfect recruitment line — *for the
wrong Amiya*:

| Candidate   | recites Amiya as…                  | (her real title: 公开领导人 / public leader) |
|-------------|------------------------------------|---------------------------------------------|
| `tiny_32k`  | 罗德岛**狙击**干员阿米娅           | Sniper |
| `small_32k` | 罗德岛**精英**干员阿米娅           | "Elite" |
| `large_32k` | 罗德岛**精英**干员阿米娅           | "Elite" |
| `small_8k`  | 罗德岛**术师**干员阿米娅           | Caster |
| `small_16k` | 罗德岛**狙击**干员阿米娅           | Sniper |
| `ctx_256`   | 罗德岛**术师**干员阿米娅           | Caster |
| `ctx_1024`  | 罗德岛**狙击**干员阿米娅           | Sniper |
| `ctx_2048`  | 罗德岛**近卫**干员阿米娅           | Guard |
| `ctx_4096`  | 阿米娅，将阿米娅托付给阿米娅       | (collapsed into a loop) |

**Finding — and it is a subtle one.** The candidates memorised the recruitment
line's *template* (`罗德岛[role]干员[name]，将[verb]…`) flawlessly — but not
its *content*. Not one, the savant `large_32k` included, recovered Amiya's
actual title. So whatever overfitting risk we worried about back in the scale
axis is overfitting to **patterns and templates**, not verbatim passages. A
small model on a small archive memorises the *grammar of the archive*, and
improvises the facts. That is a different — and more honest — picture than
"the big model memorised the training set word for word." It did not. It
memorised the *mould*.

---

## 综合评定 · Assessment Summary

**综合能力 — recommended candidate `small_8k`** (new from the previous report —
the vocab axis has now overtaken the scale axis as the deciding factor):

```
文本结构  archive-structure fluency      ★★★★☆   优良
局部通顺  local coherence                ★★★☆☆   标准
复读倾向  repetition / looping           ★★☆☆☆   异常
知识泛化  knowledge generalisation       ★★☆☆☆   异常
问询协议  interview protocol             ☆☆☆☆☆   缺失
源石耐受  temperature stability          ★★★☆☆   标准
```

- **The archive is the ceiling.** Every candidate from `small` upward speaks
  the archive's register fluently and holds almost no reliable knowledge.
  Under early stopping the gap between `tiny / small / large` collapses on
  val ppl (135 / 125 / 125) — making `large` once more no better than
  `small`, just differently — and the surviving axis is vocabulary.
- **Recommended candidate: `small_8k`.** Best val ppl (52.55), best val
  bits/char (3.879), second-best cloze score. The small vocab fills its
  embedding table; the 32k vocab carries an embedding the 6M-token corpus
  cannot teach.
- **Wider-window candidates no longer fail outright** — fixing the fairness
  bug (early stopping instead of fixed token budget) pulled the long-context
  variants up from "unreadable" to "competitive" — but the extra wall-time
  (`ctx_4096`: 203 min vs `small_32k`: 29 min) buys nothing measurable on a
  corpus this small.
- **No candidate can be interviewed.** Question-answering is missing across
  the board, by design of the archive, not by failure of any candidate.

**Next steps.** Two, and the project already plans both: install the interview
protocol via instruct tuning (「行为协议训练」), and bring in the veteran from
the other program — Track B's continued-pretrained Qwen3-0.6B — to run these
exact five stations for the head-to-head Track A is built to set up.

*— Stage 02 assessment closed. Probe set: [`../probes.txt`](../probes.txt).*
