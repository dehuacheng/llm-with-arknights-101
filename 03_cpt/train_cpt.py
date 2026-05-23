#!/usr/bin/env python3
"""Stage 03 — continued pretraining of Qwen3-0.6B-Base on Arknights cn/.

Same training objective as Stage 02 (next-token prediction on packed
sequences); the only differences are (a) the weights start from a 600M
HuggingFace checkpoint instead of random init, (b) PEFT/LoRA can wrap the
model, (c) a replay-mix dataloader can interleave general-Chinese batches.

    python3 03_cpt/train_cpt.py --config 03_cpt/configs/lora_replay.yaml
    python3 03_cpt/train_cpt.py --config <any> --smoke-test

Reads the token streams produced by 03_cpt/prepare_data.py. Checkpoints land
in data/checkpoints/<run>/ (git-ignored).

Pedagogically central regions wear EXERCISE markers. In learning mode you
delete the body between the markers and rewrite it from the spec — `git diff`
recovers the reference implementation.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --smoke-test overrides: tiny model is not available, so we keep the base but
# clamp steps/eval to a <5-minute run. LoRA + small block keeps it cheap.
SMOKE = dict(block_size=256, batch_size=2, grad_accum=2,
             max_steps=20, warmup_steps=5, eval_interval=10, eval_steps=4,
             patience=5)


# ---------------------------------------------------------------------------
# Token streams
# ---------------------------------------------------------------------------

def open_stream(path: Path) -> np.memmap:
    """Memory-map a uint32 .bin stream produced by prepare_data.py."""
    return np.memmap(path, dtype=np.uint32, mode="r")


# === EXERCISE START: pack-sequences ==================================
# Concept: turn a flat token stream into one window per row, ready to
#          feed a HuggingFace causal LM.
# Given:   data        — a 1-D uint32 array of token ids
#          block_size  — window length L
#          batch_size  — number of windows per batch
#          device      — torch device
# Produce: x  shape (B, L) int64  — the input tokens; the caller passes
#          this AS BOTH input_ids AND labels to the HF model. The model
#          internally shifts the labels by one position (see
#          transformers.loss_utils.ForCausalLMLoss), so DO NOT pre-shift
#          here — that would train the model to predict t+2 from t.
#          (This is the key contract difference from Stage 02, where the
#          from-scratch GPT expects (x, y) with y already shifted left.)
# Steps:   1) sample `batch_size` random offsets in [0, len(data) - L]
#          2) slice each into a length-L chunk
#          3) stack, cast to int64, move to device (non_blocking on CUDA)
# Learning mode: delete the body below and rewrite it from the spec.
# ----------------------------------------------------------------------
def get_batch(data: np.memmap, block_size: int, batch_size: int,
              device: torch.device, rng: np.random.Generator):
    n = len(data) - block_size
    ix = rng.integers(0, n, size=batch_size, dtype=np.int64)
    chunks = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    x = torch.from_numpy(chunks)
    return x.to(device, non_blocking=True)
# === EXERCISE END: pack-sequences ====================================


# === EXERCISE START: replay-mix ======================================
# Concept: with probability p, draw the batch from a general-Chinese stream
#          instead of the Arknights stream. Mixing per *micro-batch* (not
#          per sequence) is simpler and lets gradient accumulation average
#          a few corpora into one optimiser step.
# Given:   ark, replay   — uint32 memmaps (replay may be None)
#          mix_ratio     — P(use replay) per call; 0 disables
#          block_size, batch_size, device, rng — as in get_batch
# Produce: (x, source)  — source ∈ {"arknights", "replay"}; x is fed as
#          BOTH input_ids and labels (see pack-sequences contract).
# Steps:   1) if replay is None or rng.random() >= mix_ratio: draw from ark
#          2) else: draw from replay
#          3) return tensor + source label (for logging)
# Learning mode: rewrite from the spec.
# ----------------------------------------------------------------------
def get_mixed_batch(ark: np.memmap, replay: np.memmap | None, mix_ratio: float,
                    block_size: int, batch_size: int,
                    device: torch.device, rng: np.random.Generator):
    if replay is None or rng.random() >= mix_ratio:
        return get_batch(ark, block_size, batch_size, device, rng), "arknights"
    return get_batch(replay, block_size, batch_size, device, rng), "replay"
# === EXERCISE END: replay-mix ========================================


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------

# === EXERCISE START: lora-injection ==================================
# Concept: wrap the base Qwen3 model with low-rank adapters on every
#          listed projection matrix. The base weights stay frozen — only
#          the LoRA A/B matrices receive gradients.
# Given:   model          — a HuggingFace causal LM
#          lora_cfg       — dict with r, alpha, dropout, target_modules
# Produce: a PeftModel that exposes the same forward signature as the base
# Steps:   1) build a peft.LoraConfig with task_type=CAUSAL_LM
#          2) call peft.get_peft_model(model, lora_config)
#          3) call print_trainable_parameters() — sanity-check the count
# Learning mode: rewrite from the spec.
# ----------------------------------------------------------------------
def inject_lora(model, lora_cfg):
    from peft import LoraConfig, TaskType, get_peft_model
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
# === EXERCISE END: lora-injection ====================================


def load_base_model(base_model_path: Path, dtype_str: str, device: torch.device):
    """Loads Qwen3-0.6B-Base in bf16 (its native dtype) and moves to device."""
    from transformers import AutoModelForCausalLM
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    print(f"loading base model: {base_model_path} (dtype={dtype_str})")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path), torch_dtype=dtype, attn_implementation="sdpa",
    )
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# LR schedule + evaluation
# ---------------------------------------------------------------------------

def lr_at(step: int, base_lr: float, warmup: int) -> float:
    """Linear warmup, then held constant. Same rationale as Stage 02's
    train.py (see 02_pretrain/README.md §1): a flat LR keeps 'val loss turned
    upward' a clean early-stop signal."""
    if step < warmup:
        return base_lr * step / warmup
    return base_lr


# === EXERCISE START: eval-loss =======================================
# Concept: held-out cross-entropy on a number of eval batches, no grad.
# Given:   model, data (memmap), block_size, batch_size, n_batches, device,
#          seed — used to seed a fresh RNG each call so eval scores the
#          same windows step-to-step (the train RNG must NOT be reused —
#          that both advances training stochastics and makes eval jitter).
# Produce: mean loss (float), mean ppl (float)
# Steps:   1) build a fresh np.random.default_rng(seed)
#          2) model.eval(); torch.no_grad()
#          3) for n_batches: get_batch -> model(input_ids=x, labels=x) -> loss
#             (labels=input_ids — HF shifts internally; see pack-sequences)
#          4) accumulate, divide, return loss and exp(min(20, loss))
#             (clamp guards against early-training NaN/inf overflowing exp)
#          5) model.train() before returning
# Learning mode: rewrite from the spec.
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data: np.memmap, block_size: int, batch_size: int,
             n_batches: int, device: torch.device, seed: int):
    eval_rng = np.random.default_rng(seed)
    model.eval()
    losses = torch.zeros(n_batches, device=device)
    for i in range(n_batches):
        x = get_batch(data, block_size, batch_size, device, eval_rng)
        out = model(input_ids=x, labels=x)
        losses[i] = out.loss
    model.train()
    mean = losses.mean().item()
    return mean, math.exp(min(20.0, mean))
# === EXERCISE END: eval-loss =========================================


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke_test:
        cfg.update(SMOKE)
        cfg["name"] = cfg["name"] + "_smoke"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # --- data
    tok_root = ROOT / "data/tokenized/qwen3-0.6b-base"
    train = open_stream(tok_root / f"{cfg['train_stream']}.bin")
    val = open_stream(tok_root / f"{cfg['val_stream']}.bin")
    general_val = None
    gv_path = tok_root / f"{cfg['general_val_stream']}.bin"
    if gv_path.exists():
        general_val = open_stream(gv_path)
    replay = None
    if cfg.get("replay", {}).get("enabled"):
        replay = open_stream(tok_root / f"{cfg['replay']['stream']}.bin")
    print(f"streams: train={len(train):,} val={len(val):,}"
          + (f" replay={len(replay):,}" if replay is not None else "")
          + (f" general_val={len(general_val):,}" if general_val is not None else ""))

    # --- model
    model = load_base_model(ROOT / cfg["base_model"], cfg["dtype"], device)
    if cfg.get("lora", {}).get("enabled"):
        model = inject_lora(model, cfg["lora"])
    model.train()

    # --- optimiser
    # Weight decay applies to the 2-D+ matmul / embedding weights only; the
    # 1-D parameters (LayerNorm gains, every bias) are left undecayed — the
    # GPT-2 recipe also used in 02_pretrain/train.py. For LoRA this is moot
    # (only the A/B matrices receive gradients, all 2-D) but we keep the
    # split uniform so full-FT and LoRA take the same code path.
    trainable = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.dim() >= 2]
    no_decay = [p for p in trainable if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    # --- training loop
    block = cfg["block_size"]
    bs = cfg["batch_size"]
    accum = cfg["grad_accum"]
    max_steps = cfg["max_steps"]
    warmup = cfg["warmup_steps"]
    eval_interval = cfg["eval_interval"]
    eval_steps = cfg["eval_steps"]
    patience = cfg.get("patience", 8)
    grad_clip = cfg["grad_clip"]
    replay_p = cfg.get("replay", {}).get("mix_ratio", 0.0)
    eval_seed = cfg["seed"]  # used by evaluate() — fresh RNG each call

    best_val, best_step, evals_no_improve, stop_reason = float("inf"), -1, 0, "max_steps"
    ckpt_dir = ROOT / "data/checkpoints" / cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\ntrain run '{cfg['name']}'  (device: {device})")
    print(f"  base       : {cfg['base_model']}")
    print(f"  training   : up to {max_steps:,} steps, batch {bs}x{accum} "
          f"(effective {bs*accum}) x block {block}; early-stop patience {patience} evals")
    t0 = time.time()

    # === EXERCISE START: train-loop ====================================
    # Concept: warmup + constant LR; gradient accumulation; eval every N
    #          steps; early-stop after `patience` non-improvements.
    # Steps:   1) set LR via lr_at(step, base_lr, warmup) on optim.param_groups
    #          2) zero_grad
    #          3) for accum micro-batches: get_mixed_batch -> model(input_ids=x,
    #             labels=x) -> (loss / accum).backward()
    #             (pass labels=input_ids — the HF model shifts internally;
    #              see the pack-sequences contract)
    #          4) clip_grad_norm_(trainable, grad_clip), optim.step()
    #          5) every eval_interval steps:
    #               val_loss, _ = evaluate(model, val, block, bs, eval_steps,
    #                                      device, eval_seed)
    #               if val_loss < best_val: save checkpoint, reset counter
    #               else: evals_no_improve += 1
    #               if evals_no_improve >= patience: break (stop_reason = "early_stop")
    # Saving: use model.save_pretrained(ckpt_dir) — for a PeftModel this writes
    #         only the adapter (~5–20 MB); for a full-FT model it writes the
    #         full base (~1.2 GB). Avoid torch.save(state_dict()) here — that
    #         path saves the full model even for LoRA, defeating the size win.
    #         Write atomically: save to ckpt_dir.with_suffix(".tmp") then
    #         os.replace, so a Ctrl-C mid-save cannot corrupt the best ckpt.
    # Learning mode: rewrite from the spec. The Stage-02 train.py main loop is
    # a very close template (modulo the labels contract) — diff against it.
    # --------------------------------------------------------------------
    raise NotImplementedError(
        "EXERCISE: implement the training loop. See spec above; the "
        "Stage 02 train.py main loop is the reference."
    )
    # === EXERCISE END: train-loop ======================================

    dt = (time.time() - t0) / 60
    print(f"\n  done in {dt:.1f} min  ({stop_reason})")
    print(f"  best val   : ppl {math.exp(best_val):.2f}  at step {best_step}")


if __name__ == "__main__":
    main()
