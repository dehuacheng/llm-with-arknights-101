#!/usr/bin/env python3
"""Stage 02 — pretrain Track A's from-scratch language model.

Trains lib.model.GPT on the stage-00 train split, tokenized by a stage-01
tokenizer, and reports held-out perplexity on the val split. Configs are YAML,
one per experiment; clone, never edit in place.

    python3 02_pretrain/train.py --config 02_pretrain/configs/small_32k.yaml
    python3 02_pretrain/train.py --config <any> --smoke-test

The token stream is encoded once and cached under data/tokenized/<tokenizer>/
(git-ignored); checkpoints land in data/checkpoints/<run>/.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import bpe, corpus  # noqa: E402
from lib import model as model_lib  # noqa: E402

# --smoke-test overrides: a tiny model on a handful of files, <2-minute run.
SMOKE = dict(scale="tiny", block_size=128, batch_size=4, grad_accum=2,
             max_steps=20, warmup_steps=5, eval_interval=10, eval_steps=5)
SMOKE_TRAIN_FILES, SMOKE_VAL_FILES = 15, 5


def _cache_fingerprint(tok_path, files):
    """A digest of the tokenizer file plus every input file's path and size.
    It changes whenever the tokenizer is retrained or the split's file set is
    edited — so a stale token cache is detected instead of silently reused."""
    h = hashlib.sha1()
    h.update(tok_path.read_bytes())
    for f in sorted(files):
        h.update(f"{f}:{f.stat().st_size}".encode("utf-8"))
    return h.hexdigest()


def load_tokens(tok, tokenizer_name, split, files, tok_path, use_cache=True):
    """Token stream for `files` as an int64 array, plus the total char count.

    Each document is bracketed with BOS/EOS so the model sees boundaries.
    Cached under data/tokenized/<tokenizer>/<split>.bin; ids are stored as
    uint16 when the vocabulary fits in 16 bits (it does for vocab <= 65536),
    else uint32. The sidecar .json records the char count, the dtype, and a
    fingerprint of the inputs — a mismatch (or unreadable .json) re-encodes
    rather than trusting a stale or corrupt cache.
    """
    dtype = np.uint16 if tok.vocab_size <= 65536 else np.uint32
    cache = corpus.DATA_DIR / "tokenized" / tokenizer_name / f"{split}.bin"
    meta = cache.with_suffix(".json")
    fingerprint = _cache_fingerprint(tok_path, files) if use_cache else None

    if use_cache and cache.exists() and meta.exists():
        try:
            info = json.loads(meta.read_text())
        except (json.JSONDecodeError, ValueError):
            info = None
        if info and info.get("fingerprint") == fingerprint:
            arr = np.fromfile(cache, dtype=np.dtype(info["dtype"]))
            return arr.astype(np.int64), info["n_chars"]

    ids, n_chars = [], 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        n_chars += len(text)
        ids.extend(tok.encode(text, add_bos=True, add_eos=True))
    arr = np.array(ids, dtype=dtype)
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        arr.tofile(cache)
        meta.write_text(json.dumps({"n_chars": n_chars,
                                    "dtype": np.dtype(dtype).name,
                                    "fingerprint": fingerprint}))
    return arr.astype(np.int64), n_chars


def get_batch(data, batch_size, block_size, device, generator):
    """Sample a batch of (input, target) windows from the token stream."""
    if data.numel() < block_size + 1:
        raise ValueError(
            f"token stream has {data.numel()} tokens — need at least "
            f"{block_size + 1} for one window of block_size {block_size}")
    # === EXERCISE START: get-batch ========================================
    # Concept: a language model trains on (input, target) pairs where the
    #   target is the input shifted one token to the left -- at every
    #   position the model is asked to predict the *next* token. Build a
    #   batch by sampling random fixed-length windows from the flat stream.
    # Given:   data -- 1-D LongTensor of token ids; batch_size; block_size;
    #          device; generator -- a seeded torch.Generator for the picks.
    # Produce: x, y -- two (batch_size, block_size) LongTensors on `device`;
    #          y is x shifted left by one token.
    # Steps:   1) pick batch_size random starts in [0, len(data)-block_size-1]
    #          2) x row = data[start : start+block_size]
    #          3) y row = data[start+1 : start+block_size+1]
    #          4) stack the rows; move both to `device`
    # Learning mode: delete the body below and rewrite it from the spec;
    #   the committed code is the reference (`git diff` shows your delta).
    # ----------------------------------------------------------------------
    max_start = data.numel() - block_size
    starts = torch.randint(max_start, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in starts.tolist()])
    # === EXERCISE END: get-batch ==========================================
    return x.to(device), y.to(device)


def lr_at(step, base_lr, warmup, max_steps):
    """Learning rate schedule: linear warmup, then cosine decay to 10%."""
    if step < warmup:
        return base_lr * step / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device, eval_steps, seed):
    """Mean next-token loss over `eval_steps` batches. The seed is fixed, so
    every call scores the same batches — the perplexity curve is comparable
    step to step rather than jittering with the sample."""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    total = 0.0
    for _ in range(eval_steps):
        x, y = get_batch(data, batch_size, block_size, device, gen)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / eval_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML experiment config")
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny model on a few files for a quick end-to-end run")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.smoke_test:
        cfg = {**cfg, **SMOKE, "name": "smoke"}
    name, tokenizer_name = cfg["name"], cfg["tokenizer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])

    # -- tokenizer + data --------------------------------------------------
    tok_path = corpus.DATA_DIR / "tokenizers" / tokenizer_name / "tokenizer.json"
    if not tok_path.exists():
        sys.exit(f"tokenizer '{tokenizer_name}' not found at {tok_path}\n"
                 f"  run 01_tokenizer/train_tokenizer.py first.")
    tok = bpe.ByteBPE.load(tok_path)
    failed = [cname for cname, ok, _ in tok.sanity_check() if not ok]
    if failed:
        sys.exit(f"tokenizer '{tokenizer_name}' failed sanity checks: {failed}")

    train_files = corpus.split_files("train")
    val_files = corpus.split_files("val")
    use_cache = not args.smoke_test
    if args.smoke_test:
        train_files = train_files[:SMOKE_TRAIN_FILES]
        val_files = val_files[:SMOKE_VAL_FILES]

    print(f"train run '{name}'  (device: {device})")
    print(f"  tokenizer  : {tokenizer_name}  ({tok.vocab_size:,} tokens)")
    t_enc = time.time()
    train_arr, _ = load_tokens(tok, tokenizer_name, "train", train_files,
                               tok_path, use_cache)
    val_arr, val_chars = load_tokens(tok, tokenizer_name, "val", val_files,
                                     tok_path, use_cache)
    train_data = torch.from_numpy(train_arr)
    val_data = torch.from_numpy(val_arr)
    print(f"  corpus     : {train_data.numel():,} train tokens, "
          f"{val_data.numel():,} val tokens  ({time.time() - t_enc:.1f}s)")

    # -- model -------------------------------------------------------------
    gcfg = model_lib.GPTConfig.from_scale(
        cfg["scale"], vocab_size=tok.vocab_size,
        block_size=cfg["block_size"], dropout=cfg["dropout"])
    model = model_lib.GPT(gcfg).to(device)
    n_tr = model.num_params(non_embedding=True)
    n_all = model.num_params(non_embedding=False)
    print(f"  model      : scale={cfg['scale']}  "
          f"{n_tr / 1e6:.1f}M transformer + {(n_all - n_tr) / 1e6:.1f}M "
          f"embedding = {n_all / 1e6:.1f}M params")

    # -- optimiser + run state --------------------------------------------
    # Weight decay applies to the 2-D+ matmul / embedding weights only; the
    # 1-D parameters (LayerNorm gains, every bias) are left undecayed — the
    # GPT-2 recipe. model.parameters() yields the tied embed/head weight once.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=(0.9, 0.95))
    micro_batch, block_size = cfg["batch_size"], gcfg.block_size
    grad_accum = cfg.get("grad_accum", 1)  # micro-batches per optimiser step
    max_steps, grad_clip = cfg["max_steps"], cfg["grad_clip"]
    gen = torch.Generator().manual_seed(cfg["seed"])

    ckpt_dir = corpus.DATA_DIR / "checkpoints" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt.pt"
    best_val = float("inf")

    print(f"  training   : {max_steps:,} steps, batch {micro_batch}x"
          f"{grad_accum} (effective {micro_batch * grad_accum}) x "
          f"block {block_size}\n")
    t0 = time.time()
    model.train()
    for step in range(1, max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg["learning_rate"],
                            cfg["warmup_steps"], max_steps)
        # === EXERCISE START: train-step ===================================
        # Concept: one optimisation step, with gradient accumulation. To
        #   train at an effective batch larger than fits in GPU memory, run
        #   grad_accum micro-batches and sum their gradients before a single
        #   optimiser update. Each micro-loss is divided by grad_accum so the
        #   accumulated gradient equals that of one full-size batch.
        # Given:   model (train mode); train_data; get_batch(); gen;
        #          micro_batch, block_size, grad_accum; opt; grad_clip; device.
        # Produce: one optimiser update built from grad_accum micro-batches.
        # Steps:   1) opt.zero_grad(set_to_none=True)
        #          2) grad_accum times: sample a micro-batch with get_batch,
        #             forward to a loss, then (loss / grad_accum).backward()
        #             -- .backward() *adds* to the existing gradients
        #          3) torch.nn.utils.clip_grad_norm_(params, grad_clip)
        #          4) opt.step()
        # Learning mode: delete the body below and rewrite it from the spec;
        #   the committed code is the reference (`git diff` shows your delta).
        # ------------------------------------------------------------------
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = get_batch(train_data, micro_batch, block_size, device, gen)
            _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        # === EXERCISE END: train-step =====================================

        if step % cfg["eval_interval"] == 0 or step == max_steps:
            # Measure train and val loss the same way — averaged over
            # eval_steps batches with dropout off — so the train/val gap (the
            # overfitting signal) is a like-for-like comparison.
            train_loss = estimate_loss(model, train_data, micro_batch,
                                       block_size, device,
                                       cfg["eval_steps"], cfg["seed"])
            val_loss = estimate_loss(model, val_data, micro_batch, block_size,
                                     device, cfg["eval_steps"], cfg["seed"])
            train_ppl = math.exp(min(20.0, train_loss))
            val_ppl = math.exp(min(20.0, val_loss))
            tag = ""
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"config": vars(gcfg), "model": model.state_dict(),
                            "tokenizer": tokenizer_name, "step": step,
                            "val_loss": val_loss}, ckpt_path)
                tag = "  <- saved (best val)"
            print(f"  step {step:>6}/{max_steps}  "
                  f"train ppl {train_ppl:8.2f}  val ppl {val_ppl:8.2f}{tag}")

    # bits-per-character is comparable across tokenizers; perplexity is not
    # (a token means a different amount of text under each vocabulary). The
    # token count includes the per-document BOS/EOS sentinels (no characters
    # of their own) — a <0.1% inflation, near-constant across the sweep.
    val_bpc = best_val / math.log(2) * (val_data.numel() / val_chars)
    print(f"\n  done in {(time.time() - t0) / 60:.1f} min")
    print(f"  best val   : ppl {math.exp(min(20.0, best_val)):.2f}  "
          f"({val_bpc:.3f} bits/char)")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  sample     : python3 02_pretrain/sample.py --name {name} "
          f"--prompt '罗德岛'")


if __name__ == "__main__":
    main()
