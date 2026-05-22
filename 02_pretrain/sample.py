#!/usr/bin/env python3
"""Stage 02 — sample text from a trained Track A model.

Loads a checkpoint written by train.py and autoregressively generates a
continuation of a prompt. Track A's corpus is small, so do not expect
coherence — this is for *watching what the model learned*, qualitatively.

    python3 02_pretrain/sample.py --name small_32k --prompt '罗德岛'
    python3 02_pretrain/sample.py --name tiny_32k  --temperature 1.0 --top-k 0
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import bpe, corpus  # noqa: E402
from lib import model as model_lib  # noqa: E402


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, top_k, block_size, gen):
    """Autoregressively extend `idx` (1, t) by `max_new_tokens` tokens."""
    # === EXERCISE START: sample-loop ======================================
    # Concept: autoregressive decoding. The model maps a context to a score
    #   for every possible next token; turn the scores into probabilities,
    #   sample one token, append it, and feed the longer context back in.
    #   Only the *last* position's logits matter at each step.
    # Given:   model (eval mode); idx (1, t) of context ids; max_new_tokens;
    #          temperature > 0 (higher = flatter, more random); top_k (0/None
    #          = off); block_size; gen -- a seeded torch.Generator.
    # Produce: idx grown by max_new_tokens columns along dim 1.
    # Steps:   1) crop idx to its last block_size tokens (the context window)
    #          2) logits, _ = model(cropped); keep logits[:, -1, :]
    #          3) divide logits by temperature
    #          4) if top_k: keep only the top_k logits, set the rest to -inf
    #          5) softmax -> probabilities; sample one id with multinomial
    #          6) append the id to idx; repeat max_new_tokens times
    # Learning mode: delete the body below and rewrite it from the spec;
    #   the committed code is the reference (`git diff` shows your delta).
    # ----------------------------------------------------------------------
    for _ in range(max_new_tokens):
        context = idx[:, -block_size:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=gen)
        idx = torch.cat([idx, next_id], dim=1)
    # === EXERCISE END: sample-loop ========================================
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True,
                    help="run name — checkpoint at data/checkpoints/<name>/")
    ap.add_argument("--prompt", default="", help="text to continue")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.temperature <= 0:
        sys.exit("--temperature must be > 0")
    if args.top_k < 0:
        sys.exit("--top-k must be >= 0 (0 disables top-k)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = corpus.DATA_DIR / "checkpoints" / args.name / "ckpt.pt"
    if not ckpt_path.exists():
        sys.exit(f"no checkpoint at {ckpt_path} — train run '{args.name}' first.")
    ckpt = torch.load(ckpt_path, map_location=device)

    gcfg = model_lib.GPTConfig(**ckpt["config"])
    model = model_lib.GPT(gcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok_path = corpus.DATA_DIR / "tokenizers" / ckpt["tokenizer"] / "tokenizer.json"
    tok = bpe.ByteBPE.load(tok_path)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    ids = tok.encode(args.prompt, add_bos=True)
    if len(ids) > gcfg.block_size:
        print(f"warning: prompt is {len(ids)} tokens, longer than block_size "
              f"{gcfg.block_size} — only the last {gcfg.block_size} are used "
              f"as context.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, args.max_new_tokens, args.temperature,
                   args.top_k, gcfg.block_size, gen)

    val_ppl = math.exp(min(20.0, ckpt["val_loss"]))
    print(f"[run {args.name} | step {ckpt['step']} | val ppl {val_ppl:.1f}]")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
