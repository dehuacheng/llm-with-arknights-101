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

The candidates differ along three axes — body size, vocabulary, and how wide a
window of text they can hold at once:

| Candidate    | Build                | In brief |
|--------------|----------------------|----------|
| `tiny_32k`   | scale tiny · 11.7M   | the rookie — fluent in fragments, short on sense |
| `small_32k`  | scale small · 23.4M  | **the standard candidate** — the assessment centre |
| `large_32k`  | scale large · 42.3M  | the savant — drilled the archive hardest, generalises least |
| `small_8k`   | vocab 8k · 14.0M     | reads the archive in small pieces |
| `small_16k`  | vocab 16k · 17.1M    | the quiet top scorer |
| `ctx_256`    | window 256           | short memory, but the most drill time |
| `ctx_1024`   | window 1024          | a wider desk, fewer drills |
| `ctx_2048`   | window 2048          | wide desk, a quarter of the drills |
| `ctx_4096`   | window 4096          | the widest desk — and almost no time at it |

Five stations follow.

---

## Station 1 — 自由陈述 · Free Recitation

*Give the candidate a single word and let it talk.* (Sampled, temperature 0.8.)

Every candidate, without exception, opens by reproducing the **shape** of the
archive — the `<章节>` / `<正文>` tag skeleton of a story file — and fills it
with locally plausible Rhodes Island dialogue. `small_32k`, prompt `罗德岛`:

```
</活动名称>
<章节>
<章节名称>罗德岛舰船（幕间）</章节名称>
<章节简介>罗德岛一行在工程方面，凯尔希正需要博士提供罗德岛的技术支持。</章节简介>
<正文>
10:10 P.M. 天气/阴
罗德岛本舰，工程部
可露希尔: 可露希尔，你在后面等着？
```

The form is flawless; the content drifts and repeats (`罗德岛` … `罗德岛` …).
This is the headline of Station 1: **a small model learns the *register* of a
corpus long before it learns to *mean* anything.** The archive's structure is
cheap — it is the same handful of tags 1,299 times — so every candidate nails
it. Sense is expensive, and 6M tokens does not buy much.

The two widest-window candidates do not even manage the form. `ctx_4096`,
prompt `凯尔希`:

```
:                                            （      她。
佐菲娅: ......
索娜: （: ——
玛莉娅: 啧
```

That is not a stylistic choice — it is a candidate who never finished training
(see Station 3, and `RESULTS.md`). Hold that thought.

---

## Station 2 — 源石活性测试 · Originium Activity Test

*The "temperature" dial governs how much the candidate gambles on each word.
In lore terms it is Originium exposure: a little sharpens, too much dissolves.*
One candidate (`small_32k`), one prompt (`罗德岛`), the dial swept top-k off:

**temp 0.2 — 「教条」 Doctrinaire.** Safe to the point of seizing up:

```
<章节简介>罗德岛一行小队，罗德岛一行小队小队，带领小队小队小队小队小队，
带领小队小队小队小队小队小队小队。
```

**temp 0.7 — 「标准战备」 Standard.** The working range — varied, on-topic-ish.

**temp 1.0 — 「亢奋」 Agitated.** Inventive, wandering, still grammatical.

**temp 1.4 — 「失稳 · 源石扩散」 Destabilised.** The candidate crystallises into
noise — even the characters stop being words:

```
<章节名称>这玩意暂时不敢号称的军事沉重在于罗德岛不会遇到什么这么小恹那当然
聒噪也就只有知道啦的铁的城市的孩子正在痛快一杯趴在那里你们是一样的~
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
| `small_16k`  | **3.820**      | the quiet winner |
| `small_32k`  | 3.880          | the standard candidate, close behind |
| `small_8k`   | 3.915          | — |
| `tiny_32k`   | 3.963          | the rookie, predictably last of the standards |
| `large_32k`  | 4.054          | **the savant scores *worse* than `small`** |
| `ctx_256`    | 4.057          | shortest window, fine |
| `ctx_1024`   | 4.153          | — |
| `ctx_2048`   | 4.910          | — |
| `ctx_4096`   | 5.272          | the widest desk, the worst score |

Two results, both already visible in the training tables, confirmed on held-out
sentences:

- **The savant overfit.** `large_32k` has the most capacity and the best
  *training* score, yet on these probes it lands *behind* `small_32k`. Extra
  capacity, on a 6M-token archive, was spent memorising rather than
  understanding.
- **The wide-window candidates were never finished.** `ctx_2048` and `ctx_4096`
  trail badly — not because a wide window is bad, but because at a fixed token
  budget a wider window means *far fewer* training drills (1,000 for `ctx_4096`
  vs 16,000 for `ctx_256`). They are undertrained, not misdesigned.

---

## Station 4 — 标准问询 · Standard Interview

*The candidate is asked a plain question and expected to answer it.* It does
not. Asked **「阿米娅在罗德岛担任什么职务？」** (*what is Amiya's role at Rhodes
Island?*), `small_32k` replies with — a **clinical diagnosis report**:

```
临床诊断分析:造影检测结果显示，该干员体内脏器轮廓清晰，未见异常阴影……
【体细胞与源石融合率】0%
干员阿米娅没有被源石感染的迹象。
```

It did not misunderstand the question. It never saw a *question* at all. The
archive contains story scripts and operator files — and **no question-and-answer
pairs whatsoever**. So a question mark is, to the candidate, just an unusual
start to an operator file, and it autocompletes into the nearest archive format
it knows.

`large_32k`, asked the same thing, rambles about a different operator's posting.
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

| Candidate   | recites Amiya as… | (her real title: 公开领导人 / public leader) |
|-------------|-------------------|---------------------------------------------|
| `tiny_32k`  | 罗德岛**狙击**干员阿米娅 | Sniper |
| `small_32k` | 罗德岛**狙击**干员阿米娅 | Sniper |
| `large_32k` | 罗德岛**先锋**干员阿米娅 | Vanguard |
| `small_8k`  | 罗德岛**医疗**干员阿米娅 | Medic |
| `ctx_256`   | 罗德岛**精英**干员阿米娅 | "Elite" |

**Finding — and it is a subtle one.** The candidates memorised the recruitment
line's *template* (`罗德岛[role]干员[name]，将[verb]…`) flawlessly — but not its
*content*. Not one, the savant `large_32k` included, recovered Amiya's actual
title. So the "overfitting" measured back in the scale axis is overfitting to
**patterns and templates**, not verbatim passages. A small model on a small
archive memorises the *grammar of the archive*, and improvises the facts. That
is a different — and more honest — picture than "the big model memorised the
training set word for word." It did not. It memorised the *mould*.

---

## 综合评定 · Assessment Summary

**综合能力 — recommended candidate `small_32k`:**

```
文本结构  archive-structure fluency      ★★★★☆   优良
局部通顺  local coherence                ★★★☆☆   标准
复读倾向  repetition / looping           ★★☆☆☆   异常
知识泛化  knowledge generalisation       ★★☆☆☆   异常
问询协议  interview protocol             ☆☆☆☆☆   缺失
源石耐受  temperature stability          ★★★☆☆   标准
```

- **The archive is the ceiling.** Every candidate from `small` upward speaks
  the archive's register fluently and holds almost no reliable knowledge. More
  body (`large`) overfits; a wider window (`ctx_2048+`), at a fixed budget,
  starves itself of drills. The 6M-token corpus — not the model — is the limit.
- **Recommended candidate: `small_32k`** (or `small_16k`, marginally the best
  scorer). The standard build is the right one; bigger and wider both regress.
- **No candidate can be interviewed.** Question-answering is missing across the
  board, by design of the archive, not by failure of any candidate.

**Next steps.** Two, and the project already plans both: install the interview
protocol via instruct tuning (「行为协议训练」), and bring in the veteran from
the other program — Track B's continued-pretrained Qwen3-0.6B — to run these
exact five stations for the head-to-head Track A is built to set up.

*— Stage 02 assessment closed. Probe set: [`../probes.txt`](../probes.txt).*
