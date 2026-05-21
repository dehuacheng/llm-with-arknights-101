# Stage 01 — results

Tokenizer experiments. Each row is one `configs/vocab_*.yaml` run, scored with
`fertility.py`. Per the project convention, record surprises and failures here,
not only the clean numbers.

## Vocab-size sweep

Trained on the stage-00 `train` split (838 files, 11.66M chars). Fertility =
tokens per CN (Han) character; lower = shorter sequences. The train → val gap
shows how much the vocabulary is fit to the train set.

| vocab  | merges | train tok/Han | val tok/Han | test_t1 | test_t2 | train time |
|--------|--------|---------------|-------------|---------|---------|------------|
| 4 096  | 3 809  | 0.962         | 1.034       | 1.023   | 1.018   | 255 s      |
| 8 192  | 7 905  | 0.837         | 0.918       | 0.904   | 0.898   | 270 s      |
| 16 384 | 16 097 | 0.754         | 0.831       | 0.819   | 0.815   | 281 s      |
| 32 768 | 32 481 | 0.695         | 0.772       | 0.753   | 0.751   | 290 s      |

All four pass every `sanity_check` (vocab layout, merges reference prior ids,
special tokens atomic, merge counts non-increasing, round-trip lossless).

## Merge sequence

The first 15 merges of the run (identical across vocab sizes — same corpus, so
the same most-frequent pairs; vocab size only sets where the sequence stops).
Full logs in `data/tokenizers/<name>/merge_log.txt`.

```
#0   count=633,811  b'\xef\xbc' = b'\xef' + b'\xbc'
#1   count=594,399  '..'        = '.' + '.'
#2   count=493,945  b'\xe4\xb8' = b'\xe4' + b'\xb8'
#3   count=442,352  '，'        = b'\xef\xbc' + b'\x8c'
#4   count=353,199  b'\xe7\x9a' = b'\xe7' + b'\x9a'
#5   count=350,049  '的'        = b'\xe7\x9a' + b'\x84'
#6   count=315,902  b'\xe3\x80' = b'\xe3' + b'\x80'
#7   count=315,412  b'\xe4\xba' = b'\xe4' + b'\xba'
#8   count=287,205  '。'        = b'\xe3\x80' + b'\x82'
#9   count=249,580  b'\xe4\xbb' = b'\xe4' + b'\xbb'
#10  count=237,751  '....'      = '..' + '..'
#11  count=235,418  b'\xe6\x88' = b'\xe6' + b'\x88'
#12  count=224,658  b'\xe8\xbf' = b'\xe8' + b'\xbf'
#13  count=203,220  b'\xe4\xbd' = b'\xe4' + b'\xbd'
#14  count=188,579  '我'        = b'\xe6\x88' + b'\x91'
```

The progression is the lesson: raw byte pairs first (`#0`), then a pair
completing a character the moment its UTF-8 bytes are whole (`#3` →`，`,
`#5` →`的`, `#8` →`。`, `#14` →`我`), then larger units (`#10` `..`+`..`→`....`,
the dialogue ellipsis). `#2` `b'\xe4\xb8'` is the shared lead bytes of a whole
family of common Han characters (一, 上, 不, …) — BPE merges that prefix before
any single one of those characters completes.

## Observations

- **Diminishing returns.** Each vocab doubling buys less: train tok/Han falls
  0.962 → 0.837 → 0.754 → 0.695, i.e. −0.125, −0.083, −0.059 per doubling. An
  8× vocab (4k → 32k) cuts fertility only ~28%, while the embedding table
  (vocab × d_model) grows 8× — the params-vs-fertility trade-off stage 02 has
  to weigh for a *small* model.
- **train vs held-out gap is mild and roughly flat.** val − train is
  0.072 / 0.081 / 0.077 / 0.077 across the sweep — the vocabulary is somewhat
  fit to train, but a bigger vocab does *not* blow the gap open.
- **Temporal test is not harder than random val (for fertility).** test_t1 /
  test_t2 (post-cutoff, the temporal future) score *slightly better* than val
  (random pre-cutoff holdout) at every vocab size. Fertility is just
  compressibility, not modelling difficulty — the post-cutoff side-story
  events are stylistically close to the bulk of train; val also contains
  operator records, which are more varied. (test_t1 is only 4 files — small
  sample.)
- **Training time is flat (~255–290 s) regardless of vocab.** Cost is
  dominated by the fixed work — reading 838 files, pre-tokenizing 11.66M
  chars, building the initial pair counts — not the merge loop. The
  incremental pair-count + lazy-heap design in `lib/bpe.py` makes 32k merges
  almost as cheap as 4k.
- **Common names are atomic even at 4k.** `罗德岛`, `凯尔希`, `整合运动`,
  `博士` each tokenize to a single token at every vocab size — they are
  frequent enough to be merged early.
- _failures / surprises:_ none — all sanity checks passed on every run.
