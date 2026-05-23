#!/usr/bin/env python3
"""Stage 02 — zero-shot probe of Qwen3-0.6B-Base (Track B's base model).

Sibling to ``eval_probes.py`` (which probes the Track A from-scratch runs).
Same probe file (``probes.txt``), same four probe types, same log shape — so
the output diffs cleanly against the Track A log.

This runs the *base* model before any continued-pretraining, i.e. zero-shot
on the Arknights corpus. It is the anchor for Stage 03: every number CPT
moves is measured against this baseline.

  continuation  free generation across a temperature dial.
  cloze         bits-per-character on each gold span — the tokenizer-fair
                metric, directly comparable with Track A's val_bits/char.
  qa            naked questions to a base LM (Qwen3-Base, not Qwen3-Instruct
                — so the rambling non-answer framing still holds).
  memorize      greedy decoding vs. the real training text — Track A's
                corpus is held-out for Qwen, so a long shared run is the
                interesting case (it would mean Qwen saw operator files on
                the open web).

    python3 02_pretrain/eval_probes_qwen.py
    python3 02_pretrain/eval_probes_qwen.py --section cloze
    python3 02_pretrain/eval_probes_qwen.py --model-path data/models/qwen3-0.6b-base

The probe file is shared with ``eval_probes.py``; only the model and
tokenizer differ.
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse the probe-file parser and the longest-common-substring helper so the
# two scripts cannot drift apart on probe semantics.
from eval_probes import (  # noqa: E402
    DIAL_TEMPS,
    _oneline,
    _two_fields,
    longest_common_substring,
    parse_probes,
)
from lib import corpus  # noqa: E402

DEFAULT_MODEL_PATH = corpus.DATA_DIR / "models" / "qwen3-0.6b-base"
# Single-run label that shows up in the log columns. Kept fixed-width so the
# columns line up with the Track A log (which is widest at 11 chars).
RUN_LABEL = "qwen3-0.6b-base"


def load_qwen(model_path, device, dtype):
    """Load the Qwen3 base model + tokenizer from a local checkpoint dir."""
    if not Path(model_path).exists():
        sys.exit(f"model path not found: {model_path}\n"
                 f"Pre-fetch the weights to that path (the default is "
                 f"data/models/qwen3-0.6b-base).")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def cloze_bits_per_char(model, tok, prefix, gold):
    """Bits-per-character the model assigns to ``gold`` continuing ``prefix``.

    Mirror of ``eval_probes.cloze_bits_per_char`` but routed through Qwen's
    tokenizer + forward. Scores only the gold positions; the prefix is just
    the context that conditions them. Dividing by characters (not tokens)
    keeps the score comparable with the Track A column in RESULTS.md.

    The prefix is encoded with ``add_special_tokens=False`` because Qwen3
    has no BOS in its current tokenizer config (``bos_token: null``); Track A
    explicitly prepends its own BOS via ``add_bos=True``, so the cross-track
    BPC has a one-token conditioning asymmetry today. If a future Qwen3
    release introduces a BOS token, drop ``add_special_tokens=False`` to
    keep parity.
    """
    device = next(model.parameters()).device
    # Cap context at 2048 even though Qwen3 supports 32K — a long-prefix
    # probe at the full max_position_embeddings would force a 32K bf16
    # forward, spiking VRAM on the 4090 for no scoring benefit (the gold
    # span is short; only the immediate context conditions it). Current
    # probes are <100 tokens; 2048 leaves comfortable headroom.
    block_size = min(2048, model.config.max_position_embeddings)
    gold_ids = tok.encode(gold, add_special_tokens=False)
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    ids = (prefix_ids + gold_ids)[-block_size:]
    n_gold = min(len(gold_ids), len(ids) - 1)
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits = model(x).logits.float()
    surprise = F.cross_entropy(rearrange(logits, "b t v -> (b t) v"),
                               rearrange(y, "b t -> (b t)"),
                               reduction="none")
    nats = surprise[-n_gold:].sum().item()
    return nats / math.log(2) / len(gold)


@torch.no_grad()
def sample_text(model, tok, prompt, max_new_tokens, temperature, top_k, seed):
    """Generate a continuation of ``prompt`` and return only the new text.

    Mirrors the Track A sampler: top-k truncation, multinomial sampling, or
    greedy (top_k=1 + temperature ignored). Seeded for reproducibility.

    Two cross-track alignment knobs:
      - ``eos_token_id=None`` disables HF's EOS-stop so generation runs to
        ``max_new_tokens`` regardless, matching Track A's manual loop which
        has no EOS handling. Without this, Qwen could halt early and skew
        the memorize probe's longest-shared-substring count.
      - ``skip_special_tokens=False`` on decode keeps Qwen's specials
        (``<|im_end|>`` etc.) visible to the LCS comparison, matching
        Track A's decode contract.
    """
    set_seed(seed)
    device = next(model.parameters()).device
    ids = tok.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    # Pass an explicit attention_mask so HF doesn't infer one from
    # pad_token_id — if a probe prompt ever contains the EOS token id,
    # the inferred mask would treat that position as padding and skew
    # generation. The mask is all-ones (no padding in single-sample input).
    attention_mask = torch.ones_like(input_ids)
    do_sample = top_k != 1
    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_k=top_k if top_k > 0 else 0,
        pad_token_id=tok.eos_token_id,
        eos_token_id=None,  # disable EOS-stop; match Track A's no-EOS loop
    )
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=False)


def _banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def probe_continuation(model, tok, prompts, temp, max_new, seed):
    _banner(f"CONTINUATION  —  sampled, temperature {temp}, top-k 40  "
            f"({RUN_LABEL}, zero-shot)")
    for prompt in prompts:
        print(f"\nprompt: {prompt!r}")
        text = sample_text(model, tok, prompt, max_new, temp, 40, seed)
        print(f"  {RUN_LABEL:<16}| {_oneline(text)}")


def probe_temperature(model, tok, prompt, temps, max_new, seed):
    _banner(f"TEMPERATURE DIAL  —  {RUN_LABEL}, prompt {prompt!r}, top-k off")
    for t in temps:
        text = sample_text(model, tok, prompt, max_new, t, 0, seed)
        print(f"  temp {t:<4} | {_oneline(text)}")


def probe_cloze(model, tok, items):
    _banner(f"CLOZE  —  bits-per-character on the gold span "
            f"(lower = better)  ({RUN_LABEL}, zero-shot)")
    for k, (prefix, gold) in enumerate(items, 1):
        print(f"  c{k}  {prefix} … {gold}")
    print()
    header = "  " + f"{'run':<16}" + "".join(
        f"{'c' + str(k):>8}" for k in range(1, len(items) + 1)) + f"{'MEAN':>9}"
    print(header)
    scores = [cloze_bits_per_char(model, tok, p, g) for p, g in items]
    mean = sum(scores) / len(scores)
    row = "".join(f"{s:>8.3f}" for s in scores)
    print("  " + f"{RUN_LABEL:<16}" + row + f"{mean:>9.3f}")


def probe_qa(model, tok, questions, max_new, seed):
    _banner(f"Q&A INTERVIEW  —  naked questions to a base model  "
            f"({RUN_LABEL}, zero-shot)")
    for q in questions:
        print(f"\nQ: {q}")
        text = sample_text(model, tok, q, max_new, 0.7, 40, seed)
        print(f"  {RUN_LABEL:<16}| {_oneline(text)}")


def probe_memorize(model, tok, items, max_new, seed):
    _banner(f"MEMORISATION  —  greedy decoding vs. the real training text  "
            f"({RUN_LABEL}, zero-shot)")
    for prefix, reference in items:
        print(f"\nprompt    : {_oneline(prefix)!r}")
        print(f"reference : {_oneline(reference)}")
        text = sample_text(model, tok, prefix, max_new, 1.0, 1, seed)
        shared = longest_common_substring(text, reference)
        print(f"  {RUN_LABEL:<16}| verbatim {len(shared):>3} chars: {shared!r}")
        print(f"  {'':<16}|   gen: {_oneline(text)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes",
                    default=str(Path(__file__).resolve().parent / "probes.txt"),
                    help="probe-set file (default 02_pretrain/probes.txt)")
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH),
                    help="HuggingFace model directory (local snapshot)")
    ap.add_argument("--section",
                    choices=["continuation", "cloze", "qa", "memorize"],
                    help="run only one probe section (default: all)")
    ap.add_argument("--gallery-temp", type=float, default=0.8,
                    help="temperature for the cross-run continuation gallery "
                         "(must be > 0; for greedy, run probe_memorize)")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    if args.gallery_temp <= 0:
        sys.exit("--gallery-temp must be > 0 (multinomial sampling with "
                 "temperature 0 divides logits by zero -> NaN); the memorize "
                 "probe is the canonical greedy path.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16,
             "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    probes = parse_probes(Path(args.probes))

    print(f"probe set : {args.probes}")
    print(f"device    : {device}, dtype {args.dtype}")
    print(f"model     : {args.model_path}")
    model, tok = load_qwen(args.model_path, device, dtype)
    print(f"params    : {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    print(f"vocab     : {tok.vocab_size}")

    want = args.section
    if probes.get("continuation") and want in (None, "continuation"):
        prompts = probes["continuation"]
        probe_continuation(model, tok, prompts, args.gallery_temp,
                           args.max_new_tokens, args.seed)
        probe_temperature(model, tok, prompts[0], DIAL_TEMPS,
                          args.max_new_tokens, args.seed)
    if probes.get("cloze") and want in (None, "cloze"):
        items = [_two_fields(ln, "cloze") for ln in probes["cloze"]]
        probe_cloze(model, tok, items)
    if probes.get("qa") and want in (None, "qa"):
        probe_qa(model, tok, probes["qa"], 60, args.seed)
    if probes.get("memorize") and want in (None, "memorize"):
        items = [_two_fields(ln, "memorize") for ln in probes["memorize"]]
        probe_memorize(model, tok, items, 120, args.seed)


if __name__ == "__main__":
    main()
