#!/usr/bin/env python3
"""Stage 02 — probe trained Track A models with a fixed inference set.

Loads each run's checkpoint and pushes the probe set in 02_pretrain/probes.txt
through it. Four probes, four things to watch:

  continuation  free generation across a temperature dial — what the model
                sounds like, and how raising temperature dissolves it.
  cloze         bits-per-character on a known gold span — an automatic,
                tokenizer-fair knowledge metric (lower = less surprised).
  qa            naked questions — a base model has never seen the
                question->answer form; the rambling non-answer is the point.
  memorize      greedy decoding vs. the real training text — the longest
                shared verbatim run measures how much the model memorised.

Track A's corpus is tiny; this is for *seeing what the models learned and how
they differ*, not a leaderboard. The same probes re-run on Track B's Qwen
model later, for the head-to-head.

    python3 02_pretrain/eval_probes.py
    python3 02_pretrain/eval_probes.py --runs tiny_32k,small_32k,large_32k
    python3 02_pretrain/eval_probes.py --section cloze

Carries one EXERCISE block: cloze-nll (see 01_tokenizer/README.md §7).
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import bpe, corpus  # noqa: E402
from lib import model as model_lib  # noqa: E402
from sample import generate  # noqa: E402 — reuse the sample-loop EXERCISE

# The nine sweep runs, in axis order (scale, then vocab, then context). The
# eval skips any whose checkpoint is missing, so a partial sweep still works.
DEFAULT_RUNS = ["tiny_32k", "small_32k", "large_32k",
                "small_8k", "small_16k",
                "ctx_256", "ctx_1024", "ctx_2048", "ctx_4096"]
# Q&A is a "watch it fail once" probe — run it on just the centre and the
# largest model: enough to show that size does not buy the missing format.
QA_RUNS = ["small_32k", "large_32k"]
# The temperature dial — four states of one model, from rigid to dissolved.
DIAL_TEMPS = [0.2, 0.7, 1.0, 1.4]
DIAL_RUN = "small_32k"


def _unescape(s):
    r"""Turn a literal \n in the probe file into a real newline."""
    return s.replace("\\n", "\n")


def _oneline(s):
    """Collapse newlines so a generated sample fits on one log line."""
    return s.replace("\n", "↵")


def parse_probes(path):
    """Parse the sectioned probe file into {section: [content lines]}.

    '#' starts a comment, blank lines are skipped, a section header is
    [name]. Content lines are returned verbatim (tabs intact) for the
    per-section parser to split.
    """
    if not path.exists():
        sys.exit(f"probe file not found: {path}")
    sections, current = {}, None
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("[") and s.endswith("]"):
            current = s[1:-1].strip()
            sections.setdefault(current, [])
        elif current is None:
            sys.exit(f"probe file: content before any [section]:\n  {raw!r}")
        else:
            sections[current].append(s)
    return sections


def _two_fields(line, section):
    """Split a 'left<TAB>right' probe line, unescaping \\n in both fields."""
    parts = line.split("\t")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        sys.exit(f"[{section}] line must be 'left<TAB>right':\n  {line!r}")
    return _unescape(parts[0]), _unescape(parts[1])


def load_run(name, device):
    """Load one run's checkpoint into a dict, or None if it has not trained."""
    ckpt_path = corpus.DATA_DIR / "checkpoints" / name / "ckpt.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    gcfg = model_lib.GPTConfig(**ckpt["config"])
    model = model_lib.GPT(gcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok_path = (corpus.DATA_DIR / "tokenizers" / ckpt["tokenizer"]
                / "tokenizer.json")
    tok = bpe.ByteBPE.load(tok_path)
    return dict(name=name, model=model, tok=tok, cfg=gcfg)


@torch.no_grad()
def cloze_bits_per_char(model, tok, prefix, gold):
    """Bits-per-character the model assigns to `gold` continuing `prefix`.

    A cloze probe never samples: the answer is fixed, so we just ask how
    surprised the model is by the truth. Dividing the gold's total surprise
    by its character count makes the score comparable across tokenizers — a
    token covers a different amount of text under each vocabulary — so it
    also stays comparable later against Track B's Qwen tokenizer.
    """
    device = next(model.parameters()).device
    block_size = model.cfg.block_size
    # === EXERCISE START: cloze-nll ========================================
    # Concept: cross-entropy IS surprise, measured in nats — the model's
    #   -log p(actual next token). Run prefix+gold through the model, but
    #   score only the positions that predict the *gold* tokens; the prefix
    #   is just the context that conditions them.
    # Given:   model (eval mode); tok; prefix, gold (strings); block_size.
    # Produce: bits-per-character over the gold span (float, lower better).
    # Steps:   1) ids = encode(prefix, add_bos=True) + encode(gold); the
    #             last len(encode(gold)) ids are the gold tokens.
    #          2) keep the last block_size ids (the gold stays at the end).
    #          3) forward ids[:-1]; cross-entropy vs ids[1:] with
    #             reduction='none' -> one surprise value per position.
    #          4) sum the surprise over the gold positions only -> nats.
    #          5) bits = nats / ln(2); return bits / len(gold) characters.
    # Learning mode: delete the body below and rewrite it from the spec;
    #   the committed code is the reference (`git diff` shows your delta).
    # ----------------------------------------------------------------------
    gold_ids = tok.encode(gold)
    ids = (tok.encode(prefix, add_bos=True) + gold_ids)[-block_size:]
    n_gold = min(len(gold_ids), len(ids) - 1)
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    surprise = F.cross_entropy(rearrange(logits, "b t v -> (b t) v"),
                               rearrange(y, "b t -> (b t)"),
                               reduction="none")
    nats = surprise[-n_gold:].sum().item()
    # === EXERCISE END: cloze-nll ==========================================
    return nats / math.log(2) / len(gold)


def longest_common_substring(a, b):
    """The longest run of characters that appears in both strings."""
    prev = [0] * (len(b) + 1)
    best_len, best_end = 0, 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, best_end = cur[j], i
        prev = cur
    return a[best_end - best_len:best_end]


def sample_text(run, prompt, max_new_tokens, temperature, top_k, seed):
    """Generate a continuation of `prompt` and return only the new text.
    Greedy decoding is just top_k=1 — the single likeliest token each step."""
    model, tok, gcfg = run["model"], run["tok"], run["cfg"]
    device = next(model.parameters()).device
    ids = tok.encode(prompt, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = generate(model, idx, max_new_tokens, temperature, top_k,
                   gcfg.block_size, gen)
    return tok.decode(out[0, len(ids):].tolist())


def _banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def probe_continuation(runs, prompts, temp, max_new, seed):
    """Free generation: every run continues every prompt at one temperature."""
    _banner(f"CONTINUATION  —  sampled, temperature {temp}, top-k 40")
    for prompt in prompts:
        print(f"\nprompt: {prompt!r}")
        for run in runs:
            text = sample_text(run, prompt, max_new, temp, 40, seed)
            print(f"  {run['name']:<11}| {_oneline(text)}")


def probe_temperature(run, prompt, temps, max_new, seed):
    """One model, one prompt, the temperature dial — rigid to dissolved."""
    _banner(f"TEMPERATURE DIAL  —  run {run['name']}, prompt {prompt!r}, "
            f"top-k off")
    for t in temps:
        text = sample_text(run, prompt, max_new, t, 0, seed)
        print(f"  temp {t:<4} | {_oneline(text)}")


def probe_cloze(runs, items):
    """Bits-per-character on each gold span, as a run x item matrix."""
    _banner("CLOZE  —  bits-per-character on the gold span (lower = better)")
    for k, (prefix, gold) in enumerate(items, 1):
        print(f"  c{k}  {prefix} … {gold}")
    print()
    header = "  " + f"{'run':<12}" + "".join(
        f"{'c' + str(k):>8}" for k in range(1, len(items) + 1)) + f"{'MEAN':>9}"
    print(header)
    for run in runs:
        scores = [cloze_bits_per_char(run["model"], run["tok"], p, g)
                  for p, g in items]
        mean = sum(scores) / len(scores)
        row = "".join(f"{s:>8.3f}" for s in scores)
        print("  " + f"{run['name']:<12}" + row + f"{mean:>9.3f}")


def probe_qa(runs, questions, max_new, seed):
    """Naked questions — the base model has no question->answer format."""
    _banner("Q&A INTERVIEW  —  naked questions to a base model (expect "
            "non-answers)")
    for q in questions:
        print(f"\nQ: {q}")
        for run in runs:
            text = sample_text(run, q, max_new, 0.7, 40, seed)
            print(f"  {run['name']:<11}| {_oneline(text)}")


def probe_memorize(runs, items, max_new, seed):
    """Greedy decoding vs. the real training text — measures regurgitation."""
    _banner("MEMORISATION  —  greedy decoding vs. the real training text")
    for prefix, reference in items:
        print(f"\nprompt    : {_oneline(prefix)!r}")
        print(f"reference : {_oneline(reference)}")
        for run in runs:
            text = sample_text(run, prefix, max_new, 1.0, 1, seed)
            shared = longest_common_substring(text, reference)
            print(f"  {run['name']:<11}| verbatim {len(shared):>3} chars: "
                  f"{shared!r}")
            print(f"  {'':<11}|   gen: {_oneline(text)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes",
                    default=str(Path(__file__).resolve().parent / "probes.txt"),
                    help="probe-set file (default 02_pretrain/probes.txt)")
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS),
                    help="comma-separated run names to load")
    ap.add_argument("--section",
                    choices=["continuation", "cloze", "qa", "memorize"],
                    help="run only one probe section (default: all)")
    ap.add_argument("--gallery-temp", type=float, default=0.8,
                    help="temperature for the cross-run continuation gallery")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probes = parse_probes(Path(args.probes))

    wanted = [r.strip() for r in args.runs.split(",") if r.strip()]
    runs, missing = [], []
    for name in wanted:
        run = load_run(name, device)
        runs.append(run) if run else missing.append(name)
    if not runs:
        sys.exit("no checkpoints found — train a run first "
                 f"(looked under data/checkpoints/ for: {', '.join(wanted)})")

    print(f"probe set : {args.probes}")
    print(f"device    : {device}")
    print(f"runs      : {', '.join(r['name'] for r in runs)}")
    if missing:
        print(f"skipped   : {', '.join(missing)}  (no checkpoint yet)")

    want = args.section
    if probes.get("continuation") and want in (None, "continuation"):
        prompts = probes["continuation"]
        probe_continuation(runs, prompts, args.gallery_temp,
                           args.max_new_tokens, args.seed)
        dial = next((r for r in runs if r["name"] == DIAL_RUN), runs[0])
        probe_temperature(dial, prompts[0], DIAL_TEMPS,
                          args.max_new_tokens, args.seed)
    if probes.get("cloze") and want in (None, "cloze"):
        items = [_two_fields(ln, "cloze") for ln in probes["cloze"]]
        probe_cloze(runs, items)
    if probes.get("qa") and want in (None, "qa"):
        qa_runs = [r for r in runs if r["name"] in QA_RUNS] or runs[:1]
        probe_qa(qa_runs, probes["qa"], 60, args.seed)
    if probes.get("memorize") and want in (None, "memorize"):
        items = [_two_fields(ln, "memorize") for ln in probes["memorize"]]
        probe_memorize(runs, items, 120, args.seed)


if __name__ == "__main__":
    main()
