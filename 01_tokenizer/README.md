# Stage 01 — Tokenizer: experiment report

**English | [简体中文](README.zh-CN.md)**

Before a language model can learn anything, it has to *read* — and for a
computer, reading means chopping text into a fixed menu of pieces called
**tokens**. The thing that does the chopping is the **tokenizer**, and it has
to be built before the model is. This stage builds ours.

Track A of this project builds a language model *from scratch*, so we don't
reach for an off-the-shelf tokenizer either. This stage implements a
**byte-level BPE in pure Python — no HuggingFace, no `tokenizers` library**
(`lib/bpe.py`), trains it on the *Arknights* lore corpus, and sweeps the
vocabulary size.

There is one idea behind **BPE** (byte-pair encoding): it starts out knowing
nothing. It scans the corpus, finds the two neighbouring pieces that sit side
by side most often, and glues them into one new piece — a **merge**. Then it
repeats, thousands of times. Early merges glue raw bytes into characters; later
ones glue characters into words.

The headline result of the sweep is a single number — *fertility* falls with
diminishing returns as the vocabulary grows. But the result worth staying for
is the **merge log**: the complete, ordered list of every merge the algorithm
made — effectively its diary. Read it back and you can watch the algorithm
teach itself, one glue-step at a time, that 罗德岛 (Rhodes Island) is a word —
and that 罗德 on its own is not. This report walks through both: the number,
and the diary.

---

## 1. Setup

**Corpus.** The stage-00 `train` split: 838 cleaned lore files, 11.66M
characters, overwhelmingly Chinese. Held-out `val` / `test` splits are never
seen during training.

**The tokenizer (`lib/bpe.py`).**

- **Byte-level.** Text is UTF-8 bytes, so the base alphabet is the 256 byte
  values — there is *never* an out-of-vocabulary character, even for rare CJK.
  A Chinese character costs 3 bytes; the entire job of BPE is to win that back.
- **From scratch.** The merge-training loop, the encoder, the decoder and the
  on-disk format are all hand-written. Training uses incremental pair-count
  bookkeeping and a lazy max-heap — the optimisation that makes BPE practical.
- **Special tokens** (reserved ids, never split, never merged): the 28
  structural lore tags from stage 00 (`<正文>`, `<干员名称>`, …) plus
  `<|bos|>` / `<|eos|>` / `<|pad|>`.
- **Pre-tokenization** splits text into runs of one character class — Han /
  Latin / digit / whitespace / punctuation — so a merge never crosses a class
  boundary. (This is the GPT-2 idea, with a CN-appropriate rule.)

**The sweep.** Four vocabulary sizes — 4 096 / 8 192 / 16 384 / 32 768 — each
its own `configs/vocab_*.yaml`.

---

## 2. Result — fertility vs. vocabulary size

**Fertility** is the standard score for a tokenizer: tokens per character.
Lower is better — the same text is packed into a shorter sequence, which the
model is then faster and cheaper to train on. `tok/Han` divides by Chinese
characters only; `tok/char` by all characters.

| vocab  | merges | train tok/Han | val tok/Han | test_t1 | test_t2 | train time |
|--------|--------|---------------|-------------|---------|---------|------------|
| 4 096  | 3 809  | 0.962         | 1.034       | 1.023   | 1.018   | 255 s      |
| 8 192  | 7 905  | 0.837         | 0.918       | 0.904   | 0.898   | 270 s      |
| 16 384 | 16 097 | 0.754         | 0.831       | 0.819   | 0.815   | 281 s      |
| 32 768 | 32 481 | 0.695         | 0.772       | 0.753   | 0.751   | 290 s      |

A no-merge byte tokenizer would spend ~3 tokens on every Chinese character.
Even the 4k vocabulary already pulls that down to ~1.0 — a 3× win — because
whole common words collapse into single tokens.

Four findings:

- **Diminishing returns.** Each doubling of the vocabulary buys less: train
  `tok/Han` falls by 0.125, then 0.083, then 0.059. An **8× vocabulary
  (4k → 32k) cuts fertility only ~28%**, while the model's embedding table
  (vocab × d_model) grows 8×. That trade-off is stage 02's problem.
- **The train→held-out gap is mild and flat.** `val − train` is
  0.072 / 0.081 / 0.077 / 0.077 — the vocabulary is somewhat fit to the train
  set, but a bigger vocabulary does *not* blow the gap open.
- **Temporal test is not harder than random val** (for fertility). The
  post-cutoff `test` splits score *slightly better* than the random `val`
  holdout. Fertility is just compressibility, not modelling difficulty — the
  post-cutoff side-stories are stylistically close to the training bulk.
- **Training time is flat (~255–290 s)** regardless of vocabulary size. Cost
  is dominated by the fixed work — reading 838 files, pre-tokenizing 11.66M
  characters, building the initial pair counts — not the merge loop. The
  incremental-update design makes 32k merges almost as cheap as 4k.

---

## 3. Reading the diary — tracing Arknights proper nouns (专有名词)

Every run writes the full merge log to `data/tokenizers/<name>/merge_log.txt`.
`ByteBPE.trace()` (and `trace.py`) reads it the other way round: give it a
word, and it pulls out every merge that went into building that word and lists
them in the order BPE learned them — the word's whole life story, from raw
bytes up to a single token. Because BPE is greedy (it always merges the
*currently* most frequent pair), a trace is a strict bottom-up build: small,
common pieces first, the rare whole thing last. All traces below come from the
32k tokenizer.

### 3.1 罗德岛 (Rhodes Island) — bytes → character → word

```
罗德岛  ->  1 token
  #37    count 96,613   b'\xe5\xbe' = b'\xe5' + b'\xbe'
  #142   count 27,850   '德'        = b'\xe5\xbe' + b'\xb7'
  #169   count 23,247   b'\xe7\xbd' = b'\xe7' + b'\xbd'
  #251   count 16,294   '罗'        = b'\xe7\xbd' + b'\x97'
  #307   count 13,154   b'\xe5\xb2' = b'\xe5' + b'\xb2'
  #383   count  9,741   '岛'        = b'\xe5\xb2' + b'\x9b'
  #391   count  9,567   '罗德'      = '罗' + '德'
  #392   count  9,554   '罗德岛'    = '罗德' + '岛'
```

Three layers, bottom to top. **Byte pairs** first (`#37`, `#169`, `#307`) —
each is the shared 2-byte lead of a family of UTF-8 characters, merged before
any single character is complete. Then **characters** finish as their third
byte arrives (`#142` 德, `#251` 罗, `#383` 岛). Then **words**: `罗`+`德`→`罗德`,
and immediately `罗德`+`岛`→`罗德岛`.

Look at the last two counts: `罗德` 9,567 and `罗德岛` 9,554 — just 13 apart.
In this corpus the fragment `罗德` is `罗德岛` **99.9 %** of the time; `罗德` is
barely a word, it exists almost only as the start of Rhodes Island.

### 3.2 凯尔希 (Kal'tsit) — the suffix becomes a token first

```
凯尔希  ->  1 token   (byte-pair rows omitted)
  #79    count 45,770   '尔'     = b'\xe5\xb0' + b'\x94'
  #192   count 21,059   '希'     = b'\xe5\xb8' + b'\x8c'
  #368   count 10,057   '凯'     = b'\xe5\x87' + b'\xaf'
  #451   count  8,261   '尔希'   = '尔' + '希'
  #454   count  8,139   '凯尔希' = '凯' + '尔希'
```

`尔希` becomes its own token at `#451`, three merges *before* the full name at
`#454`. `尔` is one of the most frequent characters in the whole corpus
(45,770) — it is the glue of transliterated names (凯**尔**希, 切**尔**诺伯格,
…). BPE merges the more-frequent pair `尔`+`希` first, then attaches `凯`.

### 3.3 Is the prefix a real word? Read the counts.

The count gap between a compound and its prefix reveals whether the prefix
stands on its own:

```
#391   count 9,567   '罗德'                       prefix of 罗德岛 (9,554) — barely a word
#532   count 6,759   '源石'                       Originium — a heavily used word
#1336  count 1,939   '源石技艺' = '源石' + '技艺'   the compound, 3.5× rarer than its prefix
```

`源石` ("Originium", the substance) merges at `#532` and is used 6,759 times;
the compound `源石技艺` ("Originium Arts") only 1,939. The prefix dwarfs the
compound — `源石` is a word in its own right. `罗德`, by contrast, scarcely
outlives its compound. Same shape of trace, opposite linguistics — and the
count column is what tells them apart.

### 3.4 The long tail — 罗德岛制药

```
罗德岛制药  ->  1 token   (early rows omitted)
  #392    count 9,554   '罗德岛'     = '罗德' + '岛'
  #11255  count    92   '制药'       = '制' + '药'
  #26287  count    27   '罗德岛制药' = '罗德岛' + '制药'
```

The full company name "Rhodes Island Pharmaceutical" reuses the `罗德岛` token
from merge #392, but does not itself merge until **#26,287**, at a frequency of
just 27. This is the diminishing-returns curve of §2 seen at the level of one
term: late merges each save only a handful of tokens.

### 3.5 PRTS — an English term, no byte stage

```
PRTS  ->  1 token
  #4666  count 354   'RT'   = 'R' + 'T'
  #5473  count 279   'PRT'  = 'P' + 'RT'
  #5488  count 279   'PRTS' = 'PRT' + 'S'
```

ASCII letters are single bytes, so there is no byte-assembly layer — the trace
starts straight from letters. `PRT` and `PRTS` have the *identical* count, 279:
every `PRT` in the corpus is part of `PRTS` (the in-world operating system).
Like `罗德`, `PRT` is not a word — it is a fragment that only ever appears
inside one name. (Operator codenames such as `Logos` and `Lancet` trace the
same way — run `trace.py` on them.)

### 3.6 BPE also learns structure, not only words

Some of the most-merged tokens are recurring *shapes* of the text rather than
words:

```
#10   count 237,751   '....'   = '..' + '..'
#27   count 118,362   '......' = '....' + '..'    the dialogue ellipsis, one token
#551  count   6,494   '\n\n'   = '\n' + '\n'       a blank line / paragraph break
#791  count   3,885   '？？？:' = '？？' + '？:'      the "unknown speaker" dialogue label
```

Stage 00 keeps speaker labels as ordinary text (only the *structural* tags are
reserved special tokens), so BPE is free to discover that the mystery-speaker
label `？？？:` is a single recurring unit — and it does, by merge #791.

---

## 4. Tokenizing a real line

A trace follows one term down to its bytes; here is a whole line of real lore,
tokenized. The text is Amiya's in-game **recruitment line** — a short,
public-facing line — shown under the smallest and largest vocabularies:

> 罗德岛公开领导人阿米娅，将与你并肩作战。加油，博士。
>
> *"Rhodes Island's public leader Amiya will fight at your side. Stay strong, Doctor."*

```
vocab  4 096  — 20 tokens / 26 chars
  罗德岛 | 公 | 开 | 领 | 导 | 人 | 阿米娅 | ， | 将 | 与 | 你 | 并 | 肩 | 作战 | 。 | 加 | 油 | ， | 博士 | 。

vocab 32 768  — 13 tokens / 26 chars
  罗德岛 | 公开 | 领导人 | 阿米娅 | ， | 将 | 与你 | 并肩作战 | 。 | 加油 | ， | 博士 | 。
```

Both vocabularies already keep the proper nouns `罗德岛`, `阿米娅`, `博士`
whole — these are the single tokens traced in §3. The difference is the
*connective tissue*: at 4k, `公开领导人` ("public leader") is five loose
characters and `并肩` is split; at 32k they fuse into `公开` | `领导人`, and
`并肩作战` ("fight shoulder to shoulder") becomes one token. Same line,
20 → 13 tokens — the fertility curve of §2 made concrete on a single sentence.

In the file this line is wrapped by the structural tag `<干员招聘文本>`, which
is a reserved special token — emitted as one id, never entering the BPE merges
(§1).

---

## 5. Sanity checks

`ByteBPE.sanity_check()` runs after every training run (and is reusable after
`ByteBPE.load`). All five pass on all four tokenizers:

- **vocab layout** — sizes add up; ids are contiguous.
- **merges reference prior ids** — a merge only combines tokens that already exist.
- **special tokens atomic** — each structural tag / control token encodes to
  exactly its one reserved id, never split.
- **merge counts non-increasing** — greedy BPE always takes the *current* most
  frequent pair, so each merge's count can only be ≤ the previous one. This is
  a genuine invariant (visible as the falling `count` column in §3) — and it
  doubles as a check that the incremental pair-count bookkeeping in `lib/bpe.py`
  is correct.
- **round-trip lossless** — `decode(encode(t)) == t`; byte-level BPE never
  drops information.

---

## 6. Reproduce

| File | Role |
|------|------|
| `lib/bpe.py` | `ByteBPE` — train / encode / decode / save / load, plus `merge_log()`, `trace()`, `sanity_check()`. The whole tokenizer. |
| `train_tokenizer.py` | Train one config; writes the tokenizer + `merge_log.txt`; runs the sanity checks. |
| `fertility.py` | Fertility table + example segmentations for a trained tokenizer. |
| `trace.py` | Trace one term's merge path; defaults to a sample of Arknights 专有名词. |
| `configs/vocab_*.yaml` | One per vocab size; clone, never edit in place. |
| `docs/RESULTS.md` | The raw sweep table. |

```sh
pip install -r 01_tokenizer/requirements.txt          # PyYAML only

# train (--smoke-test = tiny vocab on a few files, for a quick end-to-end check)
python3 01_tokenizer/train_tokenizer.py --config 01_tokenizer/configs/vocab_16k.yaml
python3 01_tokenizer/train_tokenizer.py --config 01_tokenizer/configs/vocab_4k.yaml --smoke-test

# score a trained tokenizer
python3 01_tokenizer/fertility.py --name vocab_16k

# trace the merge path of any term
python3 01_tokenizer/trace.py --name vocab_32k 罗德岛 整合运动 PRTS
```

Trained tokenizers and the full `merge_log.txt` land under
`data/tokenizers/<name>/` — git-ignored, because the merge table holds many
short byte-fragments of the corpus. The merge examples quoted in this report
are byte pairs and public faction/operator names, which are IP-safe to share.
