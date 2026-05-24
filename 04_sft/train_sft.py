#!/usr/bin/env python3
"""Stage 04 — SFT distillation: teach the Stage-03 CPT checkpoint to answer
Arknights questions in Qwen3 chat format.

Reads agent-produced Q&A pairs from data/sft/qa_{train,val}.jsonl (one JSON
object per line, schema in data_gen/AGENT_BRIEF.md §4). Applies Qwen3's
chat template, masks prompt tokens with -100 so the loss only fires on the
assistant response, runs a hand-rolled training loop matching the Stage 02
/ Stage 03 conventions (warmup + constant LR, gradient accumulation,
periodic eval, early-stop on best val).

    python3 04_sft/train_sft.py --config 04_sft/configs/sft_lora.yaml
    python3 04_sft/train_sft.py --config <any> --smoke-test

Checkpoints land at data/checkpoints/<run>/ via model.save_pretrained:
LoRA writes the adapter (~5-20 MB); full-FT writes the full ~1.2 GB
checkpoint.

Pedagogically central regions wear EXERCISE markers. In learning mode you
delete the body between the markers and rewrite it from the spec — git diff
recovers the reference implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --smoke-test overrides: small batches, few steps, LoRA forced on regardless
# of the config (so the smoke loads cheap even for sft_full.yaml).
SMOKE = dict(batch_size=2, grad_accum=2, max_steps=30, warmup_steps=5,
             eval_interval=10, eval_steps=4, patience=2, max_length=512,
             force_lora=True)
SMOKE_TRAIN_ROWS, SMOKE_VAL_ROWS = 40, 16


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """JSONL is small (~5000 lines); a full in-memory list is fine and keeps
    the loader trivial. Stage 03 needed memmaps because the token stream was
    big and contiguous; SFT pairs are structured, on the order of 10 MB."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# === EXERCISE START: sft-format ======================================
# Concept: turn one JSON row {"messages": [system, user, assistant]} into
#          (input_ids, labels) where the assistant tokens contribute to the
#          loss and the prompt tokens get label = -100 (HF's ignore_index).
#          The crux: render the prompt-only chat template, count its length,
#          then render the full template and slice — everything before the
#          prompt cutoff is masked.
# Given:   tok         — Qwen3 AutoTokenizer
#          row         — one JSONL row with a "messages" list
#          max_length  — truncate longer examples (right-truncate so the
#                        assistant response is preserved as long as possible)
# Produce: input_ids (1-D LongTensor), labels (1-D LongTensor of same len)
# Steps:   1) prompt_text = tok.apply_chat_template(row["messages"][:-1],
#                                                   tokenize=False,
#                                                   add_generation_prompt=True)
#             — the template's "now generate the assistant turn" hook
#          2) full_text = tok.apply_chat_template(row["messages"],
#                                                 tokenize=False)
#          3) prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
#             full_ids   = tok(full_text,   add_special_tokens=False).input_ids
#          4) labels = list(full_ids); for i in range(len(prompt_ids)):
#                  labels[i] = -100
#          5) if len(full_ids) > max_length: right-truncate both
#          6) return torch.tensor(full_ids, long), torch.tensor(labels, long)
# Learning mode: rewrite from the spec. Common pitfall: apply_chat_template
# can prepend a BOS that doesn't appear in the tokenized prompt-only output,
# causing a one-token mask shift. Assert prompt_ids is a prefix of full_ids
# in your implementation.
# ----------------------------------------------------------------------
def format_row(tok, row: dict, max_length: int):
    msgs = row["messages"]
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                          add_generation_prompt=True)
    full_text = tok.apply_chat_template(msgs, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    # The prompt-only render should be a prefix of the full render — assert
    # so a Qwen tokenizer change is caught here rather than silently shifting
    # the loss mask.
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "chat template prompt-only render is not a prefix of full render — "
            "the label mask would shift; check tokenizer / template version")
    labels = list(full_ids)
    for i in range(len(prompt_ids)):
        labels[i] = -100
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
# === EXERCISE END: sft-format ========================================


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int):
    """Right-pad to the longest example in the batch. Labels pad with -100
    (the ignore_index) so padding never contributes to loss."""
    max_len = max(len(x) for x, _ in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        input_ids[i, :len(x)] = x
        labels[i, :len(y)] = y
        attn[i, :len(x)] = 1
    return input_ids, labels, attn


def iter_batches(rows: list[dict], tok, batch_size: int, max_length: int,
                 rng: np.random.Generator) -> Iterator[tuple]:
    """Infinite shuffled iterator over the row list, yielding collated batches."""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    while True:
        order = rng.permutation(len(rows))
        for i in range(0, len(order) - batch_size + 1, batch_size):
            batch = [format_row(tok, rows[j], max_length) for j in order[i:i + batch_size]]
            yield collate(batch, pad_id)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(base_path: Path, dtype_str: str, device: torch.device):
    from transformers import AutoModelForCausalLM
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    print(f"loading base: {base_path} (dtype={dtype_str})")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), torch_dtype=dtype, attn_implementation="sdpa",
    )
    model.to(device)
    return model


def inject_lora(model, lora_cfg):
    """Same LoRA wrap as Stage 03 — the base stays frozen, only A/B matrices
    receive gradients."""
    from peft import LoraConfig, TaskType, get_peft_model
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.0),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# LR schedule + evaluation
# ---------------------------------------------------------------------------

def lr_at(step: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / warmup
    return base_lr


@torch.no_grad()
def evaluate(model, val_rows: list[dict], tok, batch_size: int, max_length: int,
             n_batches: int, device: torch.device, seed: int):
    """Mean cross-entropy on val rows (masked to response tokens only). A
    fresh RNG per call so eval scores the same windows step-to-step (the
    train RNG is not reused — Stage 03 hit that bug)."""
    eval_rng = np.random.default_rng(seed)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    losses = []
    model.eval()
    order = eval_rng.permutation(len(val_rows))
    for i in range(min(n_batches, len(order) // batch_size)):
        batch = [format_row(tok, val_rows[j], max_length)
                 for j in order[i * batch_size:(i + 1) * batch_size]]
        input_ids, labels, attn = collate(batch, pad_id)
        input_ids, labels, attn = (t.to(device) for t in (input_ids, labels, attn))
        out = model(input_ids=input_ids, labels=labels, attention_mask=attn)
        losses.append(out.loss.item())
    model.train()
    mean = sum(losses) / max(1, len(losses))
    return mean, math.exp(min(20.0, mean))


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
        cfg.update({k: v for k, v in SMOKE.items() if k != "force_lora"})
        cfg["name"] = cfg["name"] + "_smoke"
        # In smoke mode, force LoRA on even for sft_full configs so we don't
        # blow VRAM loading + full-FT optimising a 600M model in 30 steps.
        if SMOKE.get("force_lora") and not cfg.get("lora", {}).get("enabled"):
            cfg["lora"] = {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.0,
                           "target_modules": ["q_proj", "v_proj"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # --- data
    sft_root = ROOT / "data/sft"
    train_rows = load_jsonl(sft_root / cfg["train_file"],
                            limit=SMOKE_TRAIN_ROWS if args.smoke_test else None)
    val_rows = load_jsonl(sft_root / cfg["val_file"],
                          limit=SMOKE_VAL_ROWS if args.smoke_test else None)
    print(f"data: {len(train_rows):,} train rows, {len(val_rows):,} val rows")

    # --- tokenizer + model
    from transformers import AutoTokenizer
    base_path = ROOT / cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(str(base_path))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id  # Qwen3 has no pad token by default
    model = load_model(base_path, cfg["dtype"], device)
    if cfg.get("lora", {}).get("enabled"):
        model = inject_lora(model, cfg["lora"])

    # Gradient checkpointing — same Qwen3 151K-vocab logits gotcha as Stage 03.
    # SFT at max_length=1024, batch=4 has identical tokens/micro-batch (4096)
    # as Stage 03's 2×2048, so the same fp32-logits transient (~2.4 GB) bites.
    # Default on; configs can set gradient_checkpointing: false if VRAM allows.
    if cfg.get("gradient_checkpointing", True):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.train()

    # --- optimiser (decay vs no-decay split — GPT-2 recipe, same as 03_cpt)
    trainable = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.dim() >= 2]
    no_decay = [p for p in trainable if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    # --- run state
    max_length = cfg["max_length"]
    bs = cfg["batch_size"]
    accum = cfg["grad_accum"]
    max_steps = cfg["max_steps"]
    warmup = cfg["warmup_steps"]
    eval_interval = cfg["eval_interval"]
    eval_steps = cfg["eval_steps"]
    patience = cfg.get("patience", 5)
    grad_clip = cfg["grad_clip"]
    eval_seed = cfg["seed"]

    best_val, best_step, evals_no_improve, stop_reason = float("inf"), -1, 0, "max_steps"
    ckpt_dir = ROOT / "data/checkpoints" / cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Tmp dir for atomic save — string concat (not with_suffix) so dotted names
    # like 'sft_full_v1.0' don't lose their trailing version when we add .tmp.
    tmp_dir = Path(str(ckpt_dir) + ".tmp")

    print(f"\ntrain run '{cfg['name']}'  (device: {device})")
    print(f"  base       : {cfg['base_model']}")
    print(f"  training   : up to {max_steps:,} steps, batch {bs}x{accum} "
          f"(effective {bs*accum}) max_length {max_length}; "
          f"early-stop patience {patience} evals")
    t0 = time.time()

    train_iter = iter_batches(train_rows, tok, bs, max_length, rng)

    # === EXERCISE START: train-loop ====================================
    # Concept: warmup + constant LR; gradient accumulation; eval every N
    #          steps; early-stop after `patience` non-improvements.
    # Steps:   1) set LR via lr_at(step, base_lr, warmup) on optim.param_groups
    #          2) optim.zero_grad(set_to_none=True)
    #          3) for accum micro-batches:
    #               (input_ids, labels, attn) = next(train_iter)
    #               move tensors to device
    #               out = model(input_ids=input_ids, labels=labels,
    #                           attention_mask=attn)
    #               (out.loss / accum).backward()
    #          4) torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
    #          5) optim.step()
    #          6) every eval_interval steps:
    #               val_loss, val_ppl = evaluate(model, val_rows, tok, bs,
    #                                            max_length, eval_steps,
    #                                            device, eval_seed)
    #               if val_loss < best_val:
    #                   save to ckpt_dir via model.save_pretrained
    #                   (PeftModel writes adapter only; full-FT writes full)
    #                   write to .tmp and os.replace for atomicity
    #                   evals_no_improve = 0
    #               else:
    #                   evals_no_improve += 1
    #               if evals_no_improve >= patience: break (stop_reason = "early_stop")
    # Learning mode: rewrite from the spec. The 02_pretrain/train.py and
    # 03_cpt/train_cpt.py loops are close templates — diff against them.
    # --------------------------------------------------------------------
    for step in range(1, max_steps + 1):
        for g in optim.param_groups:
            g["lr"] = lr_at(step, cfg["learning_rate"], warmup)

        optim.zero_grad(set_to_none=True)
        for _ in range(accum):
            input_ids, labels, attn = next(train_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            out = model(input_ids=input_ids, labels=labels, attention_mask=attn)
            (out.loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optim.step()

        if step % eval_interval == 0 or step == max_steps:
            val_loss, val_ppl = evaluate(model, val_rows, tok, bs, max_length,
                                         eval_steps, device, eval_seed)
            # NaN guard — `nan < best_val` is False, so without an explicit
            # branch a divergence silently lands in the no-improve path and
            # the final summary prints exp(20)≈4.85e8 as if it were real ppl.
            if not math.isfinite(val_loss):
                stop_reason = f"non-finite val loss ({val_loss}) at step {step}"
                print(f"  step {step:>6}/{max_steps}  val loss {val_loss}"
                      f"  <- ABORTING")
                break
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                evals_no_improve = 0
                # Atomic-ish save: write to tmp_dir, then rmtree(ckpt_dir) +
                # rename. Same caveat as Stage 03 — small non-atomic gap, but
                # the new save is durable on disk before the rmtree, so a
                # crash in the gap leaves a recoverable `<name>.tmp` dir.
                # For PeftModel save_pretrained writes the adapter only
                # (~20-40 MB); full-FT writes the full ~1.2 GB.
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                model.save_pretrained(str(tmp_dir))
                tok.save_pretrained(str(tmp_dir))
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                os.rename(tmp_dir, ckpt_dir)
                tag = "  <- saved (best val)"
            else:
                evals_no_improve += 1
                tag = f"  (no improvement {evals_no_improve}/{patience})"
            print(f"  step {step:>6}/{max_steps}  val loss {val_loss:6.4f}"
                  f"  ppl {val_ppl:7.2f}{tag}")
            if evals_no_improve >= patience:
                stop_reason = (f"early-stopped — val loss did not improve for "
                               f"{patience} evals")
                break
    # === EXERCISE END: train-loop ======================================

    dt = (time.time() - t0) / 60
    print(f"\n  done in {dt:.1f} min  ({stop_reason})")
    if math.isfinite(best_val):
        print(f"  best val   : ppl {math.exp(min(20.0, best_val)):.2f}"
              f"  at step {best_step}")
        print(f"  checkpoint : {ckpt_dir}")
    else:
        print(f"  best val   : (no improving eval — no checkpoint saved)")


if __name__ == "__main__":
    main()
