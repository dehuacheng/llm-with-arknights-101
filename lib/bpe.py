"""Byte-level BPE tokenizer — implemented from scratch (no HuggingFace).

Track A builds its own tokenizer to learn the mechanics. This module is the
whole thing: training, encoding, decoding, and a custom on-disk format. Stage
01 trains it; stage 02 loads it via `ByteBPE.load`.

Design (see 01_tokenizer/README.md):

  * Byte-level — text is UTF-8 bytes, so the base alphabet is the 256 byte
    values and there is never an out-of-vocabulary character, even for rare CJK.
  * Special tokens (structural lore tags + BOS/EOS/PAD) are reserved ids,
    matched atomically *before* byte encoding — BPE never learns merges across
    them.
  * Pre-tokenization splits text into runs of one character class (Han / Latin
    / digit / whitespace / punctuation) so merges stay within a run. This is
    the GPT-2 idea, with a CN-appropriate rule instead of English contractions.

ID layout:  0..255  byte tokens
            256..255+S        the S special tokens, in given order
            256+S..vocab-1    learned merge tokens, in creation order
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BOS, EOS, PAD = "<|bos|>", "<|eos|>", "<|pad|>"
CONTROL_TOKENS = (BOS, EOS, PAD)

# Pre-tokenization: each match is a run of one character class. The classes
# partition every character, so findall() tiles the whole string with no gaps.
_CJK = r"一-鿿㐀-䶿"  # CJK ideographs (base + ext-A)
PRETOKENIZE_RE = re.compile(
    rf"[{_CJK}]+"                # CJK ideograph runs
    r"|[A-Za-z]+"               # Latin runs
    r"|[0-9]+"                  # digit runs
    r"|\s+"                     # whitespace runs
    rf"|[^{_CJK}A-Za-z0-9\s]+"  # punctuation / other runs
)


def _special_splitter(special_tokens):
    """Regex that splits text on special-token strings, keeping them."""
    if not special_tokens:
        return None
    # Longest-first so a token that is a prefix of another can't shadow it.
    alts = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    return re.compile(f"({alts})")


def _split_specials(text, special_re):
    """Yield (is_special, piece) — re.split with one group alternates the two."""
    if special_re is None:
        if text:
            yield (False, text)
        return
    for idx, piece in enumerate(special_re.split(text)):
        if piece:
            yield (bool(idx % 2), piece)


class ByteBPE:
    """A trained byte-level BPE tokenizer."""

    def __init__(self, special_tokens, merges, merge_counts=None):
        self.special_tokens = list(special_tokens)
        self.merges = [tuple(m) for m in merges]  # (left_id, right_id), creation order
        # Frequency of each pair when it was merged — provenance for the merge
        # log and trace(). Persisted by save(), so it survives a load(); None
        # only for a tokenizer built without it.
        self.merge_counts = list(merge_counts) if merge_counts else None
        self._special_re = _special_splitter(self.special_tokens)

        self.special_to_id = {t: 256 + k for k, t in enumerate(self.special_tokens)}
        self.id_to_special = {i: t for t, i in self.special_to_id.items()}
        merge_base = 256 + len(self.special_tokens)

        # encode side: pair -> merged id, and pair -> rank (creation order)
        self.merge_id = {pair: merge_base + r for r, pair in enumerate(self.merges)}
        self.merge_rank = {pair: r for r, pair in enumerate(self.merges)}
        # decode side: every non-special id -> its raw bytes
        self.id_bytes = {i: bytes([i]) for i in range(256)}
        for r, (a, b) in enumerate(self.merges):
            self.id_bytes[merge_base + r] = self.id_bytes[a] + self.id_bytes[b]

        self.vocab_size = merge_base + len(self.merges)
        self.bos_id = self.special_to_id.get(BOS)
        self.eos_id = self.special_to_id.get(EOS)
        self.pad_id = self.special_to_id.get(PAD)
        self._encode_cache = {}

    # -- training -----------------------------------------------------------

    @classmethod
    def train(cls, texts, vocab_size, special_tokens, verbose=False):
        """Learn `vocab_size` tokens from `texts` (an iterable of strings)."""
        n_special = len(special_tokens)
        merge_base = 256 + n_special
        if vocab_size < merge_base:
            raise ValueError(
                f"vocab_size {vocab_size} below the minimum {merge_base} "
                f"(256 bytes + {n_special} special tokens)")
        n_merges = vocab_size - merge_base

        # 1. Pre-tokenize into byte chunks and count them. Special-token spans
        #    are skipped: those ids are reserved, never learned.
        special_re = _special_splitter(special_tokens)
        word_freqs = Counter()
        for text in texts:
            for is_special, piece in _split_specials(text, special_re):
                if is_special:
                    continue
                for chunk in PRETOKENIZE_RE.findall(piece):
                    word_freqs[chunk.encode("utf-8")] += 1

        # === EXERCISE START: train-merge-loop ==============================
        # Concept: BPE *learns* its vocabulary by greedily merging the most
        #   frequent adjacent pair of tokens, again and again, until the
        #   vocabulary is full. Each merge becomes one new token.
        # Given:   word_freqs -- Counter of {pre-token bytes: occurrences}.
        # Produce: merges       -- ordered list of (left_id, right_id) pairs,
        #          merge_counts -- each pair's frequency the moment it won.
        # Steps:   1) start every word as its list of raw byte ids
        #          2) count every adjacent pair, weighted by word frequency
        #          3) take the most frequent pair; record it as a new token
        #          4) rewrite all words, replacing that pair with the new id
        #          5) repeat from (2) until n_merges tokens are learned
        # The reference below keeps pair stats in an incremental max-heap so a
        # merge only re-scans the words it changed -- an optimisation of steps
        # (2)-(4), not a different algorithm. A plain full rescan each loop is
        # also correct, just slower.
        # Learning mode: delete the body below and rewrite it from the spec;
        #   the committed code is the reference (`git diff` shows your delta).
        # --------------------------------------------------------------------
        # 2. BPE merge loop. Each word is a mutable list of token ids; `freqs`
        #    runs parallel to `words`. Pair statistics are kept incrementally
        #    so a merge only re-scans the words it actually changes.
        words = [list(w) for w in word_freqs]
        freqs = list(word_freqs.values())
        pair_freq = Counter()
        where = defaultdict(set)  # pair -> indices of words containing it
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_freq[p] += freqs[i]
                where[p].add(i)

        # A lazy max-heap: stale entries are filtered on pop, and the heap is
        # rebuilt if it accumulates too much staleness.
        heap = [(-c, p) for p, c in pair_freq.items()]
        heapq.heapify(heap)

        merges, merge_counts = [], []
        while len(merges) < n_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            if -neg_count != pair_freq.get(pair, 0):
                continue  # stale: the count moved since this entry was pushed
            a, b = pair
            new_id = merge_base + len(merges)
            merges.append(pair)
            merge_counts.append(-neg_count)  # frequency at merge time

            touched = set()
            for i in where.pop(pair, ()):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):           # retract this word's old pairs
                    pair_freq[p] -= f
                    where[p].discard(i)
                    touched.add(p)
                merged = []                       # rewrite the word, a,b -> new_id
                j = 0
                while j < len(w):
                    if j + 1 < len(w) and w[j] == a and w[j + 1] == b:
                        merged.append(new_id)
                        j += 2
                    else:
                        merged.append(w[j])
                        j += 1
                words[i] = merged
                for p in zip(merged, merged[1:]):  # post this word's new pairs
                    pair_freq[p] += f
                    where[p].add(i)
                    touched.add(p)

            for p in touched:
                count = pair_freq.get(p, 0)
                if count <= 0:
                    pair_freq.pop(p, None)
                    where.pop(p, None)
                else:
                    heapq.heappush(heap, (-count, p))
            if len(heap) > 4 * len(pair_freq) + 1024:
                heap = [(-c, p) for p, c in pair_freq.items() if c > 0]
                heapq.heapify(heap)
            if verbose and len(merges) % 2000 == 0:
                print(f"  merge {len(merges):>6,}/{n_merges:,}")

        if verbose:
            print(f"  learned {len(merges):,} merges "
                  f"(requested {n_merges:,})")
        # === EXERCISE END: train-merge-loop ================================
        return cls(special_tokens, merges, merge_counts)

    # -- encoding / decoding ------------------------------------------------

    def _encode_chunk(self, raw):
        """Apply learned merges to one pre-token's bytes (lowest rank first)."""
        cached = self._encode_cache.get(raw)
        if cached is not None:
            return cached
        # === EXERCISE START: apply-merges ==================================
        # Concept: encoding re-applies the learned merges to fresh text. For
        #   one pre-token's bytes, repeatedly apply the *lowest-rank* (i.e.
        #   earliest-learned, most-frequent) merge available, until none fits.
        #   Rank order matters: it reproduces the order training merged in.
        # Given:   raw -- the pre-token's bytes; self.merge_rank {pair: rank};
        #          self.merge_id {pair: merged token id}.
        # Produce: word -- the list of token ids for this pre-token.
        # Steps:   1) start `word` as the list of raw byte values
        #          2) among adjacent pairs in `word` that have a merge, find
        #             the one with the smallest rank
        #          3) replace every occurrence of that pair with its merged id
        #          4) repeat until no adjacent pair has a merge
        # Learning mode: delete the body below and rewrite it from the spec;
        #   the committed code is the reference (`git diff` shows your delta).
        # --------------------------------------------------------------------
        word = list(raw)
        while len(word) >= 2:
            best_rank, best_at = None, None
            for k in range(len(word) - 1):
                rank = self.merge_rank.get((word[k], word[k + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_at = rank, k
            if best_at is None:
                break
            a, b = word[best_at], word[best_at + 1]
            new_id = self.merge_id[(a, b)]
            merged, k = [], 0
            while k < len(word):
                if k + 1 < len(word) and word[k] == a and word[k + 1] == b:
                    merged.append(new_id)
                    k += 2
                else:
                    merged.append(word[k])
                    k += 1
            word = merged
        # === EXERCISE END: apply-merges ====================================
        self._encode_cache[raw] = word
        return word

    def encode(self, text, add_bos=False, add_eos=False):
        """Text -> list of token ids."""
        # === EXERCISE START: encode ========================================
        # Concept: turn a string into token ids. Special tokens are matched
        #   atomically first (they must never be split into bytes or merged);
        #   ordinary text is pre-tokenized into character-class runs, and each
        #   run is byte-encoded + merged by _encode_chunk.
        # Given:   text; add_bos/add_eos flags; self.bos_id / self.eos_id;
        #          self.special_to_id; _split_specials(); PRETOKENIZE_RE.
        # Produce: ids -- list[int], optionally bracketed by BOS / EOS.
        # Steps:   1) if add_bos, emit self.bos_id
        #          2) split text into special / non-special pieces
        #          3) a special piece -> its reserved id, as one token
        #          4) a non-special piece -> for each PRETOKENIZE_RE run,
        #             extend ids with _encode_chunk(run.encode("utf-8"))
        #          5) if add_eos, emit self.eos_id
        # Learning mode: delete the body below and rewrite it from the spec;
        #   the committed code is the reference (`git diff` shows your delta).
        # --------------------------------------------------------------------
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for is_special, piece in _split_specials(text, self._special_re):
            if is_special:
                ids.append(self.special_to_id[piece])
            else:
                for chunk in PRETOKENIZE_RE.findall(piece):
                    ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        if add_eos:
            ids.append(self.eos_id)
        # === EXERCISE END: encode ==========================================
        return ids

    def decode(self, ids):
        """List of token ids -> text. Lossless for ids produced by encode()."""
        # === EXERCISE START: decode ========================================
        # Concept: invert encode(). Every non-special token maps to a fixed
        #   byte string (self.id_bytes); concatenate those bytes and UTF-8
        #   decode. A token's bytes may end mid-character, so bytes are
        #   buffered and only decoded at a boundary -- flush the buffer right
        #   before each special token, and once more at the end.
        # Given:   ids; self.id_to_special {id: str}; self.id_bytes {id: bytes}.
        # Produce: the decoded string.
        # Steps:   1) keep a bytearray buffer; walk ids in order
        #          2) special id -> flush+decode the buffer, then emit the
        #             special string verbatim
        #          3) ordinary id -> append self.id_bytes[id] to the buffer
        #          4) at the end, flush+decode whatever is left
        # Learning mode: delete the body below and rewrite it from the spec;
        #   the committed code is the reference (`git diff` shows your delta).
        # --------------------------------------------------------------------
        out, buf = [], bytearray()
        for i in ids:
            special = self.id_to_special.get(i)
            if special is not None:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                out.append(special)
            else:
                buf += self.id_bytes[i]
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        # === EXERCISE END: decode ==========================================
        return "".join(out)

    # -- inspection ---------------------------------------------------------

    def _token_value(self, token_id):
        """A token's value: its text (str) when its bytes form valid UTF-8,
        else the raw bytes. repr() renders both readably — a partial token
        shows as e.g. b'\\xe5\\x8d' until its bytes complete a character."""
        special = self.id_to_special.get(token_id)
        if special is not None:
            return special
        raw = self.id_bytes[token_id]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw

    def merge_log(self):
        """Yield one record per learned merge, in the order BPE learned them.

        This is the teaching artifact: the sequence shows the vocabulary built
        bottom-up — byte pairs first, then whole characters, then words — and
        `count` (the pair's frequency when merged) falling along the way.
        `count` is None only for a tokenizer built without it.
        """
        merge_base = 256 + len(self.special_tokens)
        for rank, (a, b) in enumerate(self.merges):
            yield {
                "rank": rank,
                "count": self.merge_counts[rank] if self.merge_counts else None,
                "left": self._token_value(a),
                "right": self._token_value(b),
                "token": self._token_value(merge_base + rank),
            }

    def trace(self, text):
        """Trace the merge path of `text` — every merge that builds its
        encoding, in the order BPE learned them (raw bytes up to the final
        token). Returns (token_ids, merge_records); each record is a
        merge_log()-style dict. Handy for following one proper noun down."""
        merge_base = 256 + len(self.special_tokens)
        seen = {}

        def visit(token_id):
            if token_id < merge_base:
                return  # byte or special token — a leaf
            rank = token_id - merge_base
            if rank in seen:
                return
            a, b = self.merges[rank]
            visit(a)
            visit(b)
            seen[rank] = {
                "rank": rank,
                "count": self.merge_counts[rank] if self.merge_counts else None,
                "left": self._token_value(a),
                "right": self._token_value(b),
                "token": self._token_value(token_id),
            }

        ids = self.encode(text)
        for token_id in ids:
            visit(token_id)
        return ids, [seen[r] for r in sorted(seen)]

    def sanity_check(self, sample_texts=()):
        """Self-checks; returns a list of (name, passed, detail).

        Used by train_tokenizer.py and handy after load() in later stages.
        """
        results = []
        merge_base = 256 + len(self.special_tokens)

        expected = merge_base + len(self.merges)
        results.append(("vocab layout", self.vocab_size == expected,
                         f"{self.vocab_size} tokens, expected {expected}"))

        # a merge may only reference ids that already exist (child < parent)
        forward = [r for r, (a, b) in enumerate(self.merges)
                   if a >= merge_base + r or b >= merge_base + r]
        results.append(("merges reference prior ids", not forward,
                         "ok" if not forward else f"{len(forward)} forward refs"))

        # special tokens stay atomic — each encodes to exactly its one id
        split = [t for t in self.special_tokens
                 if self.encode(t) != [self.special_to_id[t]]]
        results.append(("special tokens atomic", not split,
                         "ok" if not split else f"{len(split)} split: {split[:2]}"))

        # greedy BPE always takes the current most-frequent pair, so each
        # merge's count can only be <= the previous one
        if self.merge_counts:
            rises = sum(1 for i in range(1, len(self.merge_counts))
                        if self.merge_counts[i] > self.merge_counts[i - 1])
            results.append(("merge counts non-increasing", rises == 0,
                             "ok" if not rises else f"{rises} increases"))

        # byte-level BPE is lossless: decode(encode(t)) == t
        if sample_texts:
            bad = sum(1 for t in sample_texts if self.decode(self.encode(t)) != t)
            results.append(("round-trip lossless", bad == 0,
                             f"{len(sample_texts) - bad}/{len(sample_texts)} ok"))
        return results

    # -- persistence --------------------------------------------------------

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format": "arknights-bpe",
            "version": 1,
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for a, b in self.merges],
            "merge_counts": self.merge_counts,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "arknights-bpe":
            raise ValueError(f"{path}: not an arknights-bpe tokenizer")
        return cls(data["special_tokens"], data["merges"],
                   data.get("merge_counts"))
