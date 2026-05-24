#!/usr/bin/env python3
"""Stage 05 — DPO (and IPO) against plausible hallucinations.

Reads agent-produced preference pairs from data/dpo/<flavour>_{train,val}.jsonl
(one JSON object per line; schema in data_gen/AGENT_BRIEF.md §5). Holds the
policy model (trainable, starts from Stage 04 SFT) and a frozen reference
(the same SFT checkpoint) in memory. Implements both the DPO log-sigmoid
loss (Rafailov 2023) and the IPO identity-link loss (Azar 2023) — pick via
the `objective: dpo|ipo` config field.

    python3 05_dpo/train_dpo.py --config 05_dpo/configs/dpo_curated.yaml
    python3 05_dpo/train_dpo.py --config <any> --smoke-test

Checkpoints land at data/checkpoints/<run>/. LoRA writes adapter only
(~5-20 MB); full-FT writes the full ~1.2 GB. The reference is *not*
checkpointed — it stays the immutable Stage 04 SFT result.

Pedagogically central regions wear EXERCISE markers. In learning mode you
delete the body between the markers and rewrite from the spec.
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
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --smoke-test overrides: tiny batch, few steps, LoRA forced on.
SMOKE = dict(batch_size=2, grad_accum=2, max_steps=20, warmup_steps=5,
             eval_interval=10, eval_steps=4, patience=2, max_length=512,
             force_lora=True)
SMOKE_TRAIN_ROWS, SMOKE_VAL_ROWS = 32, 16


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# === EXERCISE START: dpo-format ======================================
# Concept: build (prompt + response) input_ids with a -100 mask on prompt
#          tokens, identical in shape to the Stage 04 sft-format block —
#          DPO and SFT eat the same kind of tensor; only the loss differs.
#          We need three tensors per row: prompt, prompt+chosen, prompt+rejected.
# Given:   tok, row (with "prompt", "chosen", "rejected" lists of messages),
#          max_length
# Produce: (ids_chosen, labels_chosen, ids_rejected, labels_rejected)
#          all 1-D LongTensors. labels mask prompt tokens with -100.
# Steps:   1) prompt_text = tok.apply_chat_template(row["prompt"], tokenize=False,
#                                                   add_generation_prompt=True)
#          2) for each of chosen/rejected:
#               full_msgs = row["prompt"] + row[side]
#               full_text = tok.apply_chat_template(full_msgs, tokenize=False)
#               prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
#               full_ids   = tok(full_text,   add_special_tokens=False).input_ids
#               labels = list(full_ids); for i in range(len(prompt_ids)):
#                   labels[i] = -100
#               (right-truncate to max_length, keep response visible)
#          3) return all four tensors
# Learning mode: assert prompt_ids is a prefix of full_ids; same gotcha as
# the SFT formatter.
# ----------------------------------------------------------------------
def format_pair(tok, row: dict, max_length: int):
    prompt_text = tok.apply_chat_template(row["prompt"], tokenize=False,
                                          add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids

    def _side(side: str):
        full_msgs = list(row["prompt"]) + list(row[side])
        full_text = tok.apply_chat_template(full_msgs, tokenize=False)
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                "chat template prompt-only render is not a prefix of full render "
                f"on the {side} side; tokenizer/template mismatch")
        labels = list(full_ids)
        for i in range(len(prompt_ids)):
            labels[i] = -100
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]
        return (torch.tensor(full_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))

    ids_c, lbl_c = _side("chosen")
    ids_r, lbl_r = _side("rejected")
    return ids_c, lbl_c, ids_r, lbl_r
# === EXERCISE END: dpo-format ========================================


def collate_pairs(batch, pad_id: int):
    """Right-pad each side independently to its own max length. We pack
    chosen and rejected as two separate batched tensors rather than
    concatenating — keeps the masked log-prob computation clean."""
    def _pack(items):
        max_len = max(len(x) for x, _ in items)
        ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
        lbl = torch.full((len(items), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(items), max_len), dtype=torch.long)
        for i, (x, y) in enumerate(items):
            ids[i, :len(x)] = x
            lbl[i, :len(y)] = y
            attn[i, :len(x)] = 1
        return ids, lbl, attn
    chosen = [(c[0], c[1]) for c in batch]
    rejected = [(c[2], c[3]) for c in batch]
    return _pack(chosen), _pack(rejected)


def iter_batches(rows, tok, batch_size: int, max_length: int, rng):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    while True:
        order = rng.permutation(len(rows))
        for i in range(0, len(order) - batch_size + 1, batch_size):
            batch = [format_pair(tok, rows[j], max_length)
                     for j in order[i:i + batch_size]]
            yield collate_pairs(batch, pad_id)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(base_path: Path, dtype_str: str, device: torch.device):
    from transformers import AutoModelForCausalLM
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    print(f"loading: {base_path} (dtype={dtype_str})")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), torch_dtype=dtype, attn_implementation="sdpa",
    )
    model.to(device)
    return model


def inject_lora(model, lora_cfg):
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
# Log-prob computation
# ---------------------------------------------------------------------------

# === EXERCISE START: dpo-logprobs ====================================
# Concept: sum the log-probabilities of the response tokens (those with
#          label != -100) under one model. This is the per-sequence log-π
#          that the DPO/IPO objectives compare across (policy vs ref) and
#          (chosen vs rejected). The mean is over batch only, NOT over
#          tokens — DPO is a sequence-level objective.
# Given:   model, input_ids (B, T), labels (B, T)   [labels has -100 on prompt]
#          attention_mask (B, T)
# Produce: log_probs (B,) — sum of log π(response token | prefix) per row
# Steps:   1) outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#          2) logits = outputs.logits   # (B, T, V)
#          3) Shift: predict token t from logits at position t-1.
#             shift_logits = logits[:, :-1, :]
#             shift_labels = labels[:, 1:]
#          4) log_probs_all = F.log_softmax(shift_logits, dim=-1)
#          5) gather the label at each position: per_token_logp =
#             log_probs_all.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
#          6) mask out the -100 positions: per_token_logp[mask] = 0 where mask = shift_labels == -100
#          7) sum over T → (B,) log-prob of the response
# Notes: clamp(min=0) before gather is a standard trick to avoid the gather
# crashing on -100; the mask zeroes those entries before the sum.
# ----------------------------------------------------------------------
def sequence_logprobs(model, input_ids: torch.Tensor, labels: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    # Gather; clamp -100 to 0 so gather doesn't crash, then mask out below.
    gathered = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100)
    return (gathered * mask).sum(dim=-1)
# === EXERCISE END: dpo-logprobs ======================================


# ---------------------------------------------------------------------------
# DPO + IPO losses
# ---------------------------------------------------------------------------

# === EXERCISE START: dpo-loss ========================================
# Concept: DPO loss — the policy's log-ratio on `chosen` should exceed
#          its log-ratio on `rejected` (both ratios relative to the
#          frozen reference), filtered through a log-sigmoid.
# Inputs:  logπ_pol_chosen, logπ_pol_rejected, logπ_ref_chosen,
#          logπ_ref_rejected  — each (B,), already log-summed via
#          sequence_logprobs
#          beta — the DPO temperature (higher → stronger preference signal)
# Loss:    log_ratio_chosen   = logπ_pol_chosen   - logπ_ref_chosen
#          log_ratio_rejected = logπ_pol_rejected - logπ_ref_rejected
#          logits = beta * (log_ratio_chosen - log_ratio_rejected)
#          loss = -F.logsigmoid(logits).mean()
# Also returned for logging: implicit rewards (β · log_ratio_*) and the
# pairwise margin (mean of `logits` — positive = preferring chosen).
# ----------------------------------------------------------------------
def dpo_loss(logp_pol_c, logp_pol_r, logp_ref_c, logp_ref_r, beta: float):
    log_ratio_c = logp_pol_c - logp_ref_c
    log_ratio_r = logp_pol_r - logp_ref_r
    logits = beta * (log_ratio_c - log_ratio_r)
    loss = -F.logsigmoid(logits).mean()
    return loss, {
        "implicit_reward_chosen": (beta * log_ratio_c).mean().item(),
        "implicit_reward_rejected": (beta * log_ratio_r).mean().item(),
        "margin": logits.mean().item(),
        "pair_acc": (logits > 0).float().mean().item(),
    }
# === EXERCISE END: dpo-loss ==========================================


# === EXERCISE START: ipo-loss ========================================
# Concept: IPO (Azar 2023) — same data as DPO, identity link instead of
#          sigmoid. More robust to label noise because the loss penalises
#          *over-confidence* in either direction, not just under-confidence
#          on the chosen side.
# Loss:    log_ratio_c = logπ_pol_c - logπ_ref_c
#          log_ratio_r = logπ_pol_r - logπ_ref_r
#          loss = ((log_ratio_c - log_ratio_r) - 1/(2 * beta))^2  .mean()
# Note: at small β (~0.1) the (2β)^(-1) term is ~5; a well-trained model
# pushes the log-ratio diff toward that value rather than infinity.
# ----------------------------------------------------------------------
def ipo_loss(logp_pol_c, logp_pol_r, logp_ref_c, logp_ref_r, beta: float):
    log_ratio_c = logp_pol_c - logp_ref_c
    log_ratio_r = logp_pol_r - logp_ref_r
    diff = log_ratio_c - log_ratio_r
    target = 1.0 / (2.0 * beta)
    loss = ((diff - target) ** 2).mean()
    return loss, {
        "log_ratio_diff": diff.mean().item(),
        "ipo_target": target,
        "pair_acc": (diff > 0).float().mean().item(),
    }
# === EXERCISE END: ipo-loss ==========================================


# ---------------------------------------------------------------------------
# LR schedule + evaluation
# ---------------------------------------------------------------------------

def lr_at(step: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / warmup
    return base_lr


@torch.no_grad()
def evaluate(policy, ref, val_rows, tok, batch_size, max_length,
             n_batches, device, seed, objective, beta):
    """Eval loss + pairwise accuracy on held-out pairs."""
    eval_rng = np.random.default_rng(seed)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    policy.eval()
    losses, accs = [], []
    order = eval_rng.permutation(len(val_rows))
    loss_fn = dpo_loss if objective == "dpo" else ipo_loss
    for i in range(min(n_batches, len(order) // batch_size)):
        batch = [format_pair(tok, val_rows[j], max_length)
                 for j in order[i * batch_size:(i + 1) * batch_size]]
        (ids_c, lbl_c, attn_c), (ids_r, lbl_r, attn_r) = collate_pairs(batch, pad_id)
        ids_c, lbl_c, attn_c = (t.to(device) for t in (ids_c, lbl_c, attn_c))
        ids_r, lbl_r, attn_r = (t.to(device) for t in (ids_r, lbl_r, attn_r))
        logp_pol_c = sequence_logprobs(policy, ids_c, lbl_c, attn_c)
        logp_pol_r = sequence_logprobs(policy, ids_r, lbl_r, attn_r)
        logp_ref_c = sequence_logprobs(ref, ids_c, lbl_c, attn_c)
        logp_ref_r = sequence_logprobs(ref, ids_r, lbl_r, attn_r)
        loss, info = loss_fn(logp_pol_c, logp_pol_r, logp_ref_c, logp_ref_r, beta)
        losses.append(loss.item())
        accs.append(info["pair_acc"])
    policy.train()
    return (sum(losses) / max(1, len(losses)),
            sum(accs) / max(1, len(accs)))


# --- SFT-preservation tripwire (README §5 third eval signal) -----------------
# DPO's named failure mode #1: the policy cheaply pushes chosen > rejected by
# making both unlikely. We catch that by re-scoring `data/sft/qa_val.jsonl`
# under the policy and watching for the assistant-token CE drifting upward
# from the Stage-04 baseline.

def _format_sft_row(tok, row: dict, max_length: int):
    """Same shape as 04_sft/train_sft.py format_row, inlined to avoid a
    cross-stage import. Returns (input_ids, labels) with -100 on prompt,
    or None if right-truncation leaves no response tokens (skip silently)."""
    msgs = row["messages"]
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                          add_generation_prompt=True)
    full_text = tok.apply_chat_template(msgs, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    # Mirror the Stage 04 invariant: prompt-only render must be a prefix of
    # the full render. A tokenizer/template upgrade that breaks this would
    # silently shift the label mask and the SFT-preservation CE would
    # become meaningless — exactly the tripwire we're trying to make
    # trustworthy. Raise loud instead of masking the wrong tokens.
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "SFT-preservation eval: chat-template prompt-only render is not "
            "a prefix of the full render — label mask would shift; check "
            "tokenizer / template version")
    labels = list(full_ids)
    for i in range(len(prompt_ids)):
        labels[i] = -100
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
    # Defensive: if right-truncation wiped every response token, HF returns
    # NaN loss (no unignored labels) and poisons the mean. Skip the row.
    if all(l == -100 for l in labels):
        return None
    return (torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


@torch.no_grad()
def evaluate_sft_preservation(policy, sft_val_rows, tok, batch_size,
                              max_length, n_batches, device, seed):
    """Mean assistant-token CE on the SFT val set, under the *policy*. The
    Stage-04 baseline is 2.96; significant rise → DPO is eroding the
    instruction-following capability the SFT step installed."""
    if not sft_val_rows:
        return float("nan")
    eval_rng = np.random.default_rng(seed)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    policy.eval()
    losses = []
    order = eval_rng.permutation(len(sft_val_rows))
    n = min(n_batches, len(order) // batch_size)
    for i in range(n):
        rows = [sft_val_rows[j] for j in order[i * batch_size:(i + 1) * batch_size]]
        formatted = [_format_sft_row(tok, r, max_length) for r in rows]
        batch = [b for b in formatted if b is not None]   # skip all-mask rows
        if not batch:
            continue
        max_len = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        lbl = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for k, (x, y) in enumerate(batch):
            ids[k, :len(x)] = x
            lbl[k, :len(y)] = y
            attn[k, :len(x)] = 1
        ids, lbl, attn = (t.to(device) for t in (ids, lbl, attn))
        out = policy(input_ids=ids, labels=lbl, attention_mask=attn)
        losses.append(out.loss.item())
    policy.train()
    # Return NaN when zero batches actually ran (e.g., batch_size > rows)
    # so the caller's math.isnan check fires — 0.0 would masquerade as a
    # perfect preservation score.
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


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
        if SMOKE.get("force_lora") and not cfg.get("lora", {}).get("enabled"):
            cfg["lora"] = {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.0,
                           "target_modules": ["q_proj", "v_proj"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    objective = cfg.get("objective", "dpo").lower()
    if objective not in {"dpo", "ipo"}:
        raise SystemExit(f"objective must be 'dpo' or 'ipo'; got {objective!r}")
    beta = float(cfg["beta"])

    # --- data
    dpo_root = ROOT / "data/dpo"
    train_rows = load_jsonl(dpo_root / cfg["train_file"],
                            limit=SMOKE_TRAIN_ROWS if args.smoke_test else None)
    val_rows = load_jsonl(dpo_root / cfg["val_file"],
                          limit=SMOKE_VAL_ROWS if args.smoke_test else None)
    print(f"data: {len(train_rows):,} train pairs, {len(val_rows):,} val pairs")
    print(f"objective: {objective}  β={beta}")

    # --- SFT-preservation eval set (README §5 tripwire). Hard path; warn
    # rather than crash if absent, so the loop still works against a base
    # that lacks Stage-04 artefacts (unlikely, but explicit).
    sft_val_path = ROOT / "data/sft/qa_val.jsonl"
    sft_val_rows = (load_jsonl(sft_val_path,
                               limit=SMOKE_VAL_ROWS if args.smoke_test else None)
                    if sft_val_path.exists() else [])
    if sft_val_rows:
        print(f"sft-preservation: {len(sft_val_rows):,} rows from {sft_val_path.relative_to(ROOT)}")
    else:
        print(f"sft-preservation: skipped ({sft_val_path} not found)")

    # --- tokenizer + models (policy + frozen reference)
    from transformers import AutoTokenizer
    base_path = ROOT / cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(str(base_path))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    policy = load_model(base_path, cfg["dtype"], device)
    if cfg.get("lora", {}).get("enabled"):
        policy = inject_lora(policy, cfg["lora"])

    # Gradient checkpointing on the policy — same Qwen3 151K-vocab fp32-
    # logits-cast pressure as Stage 03/04, hit FOUR times per DPO step
    # (chosen/rejected × policy/ref). Ref doesn't need it (no_grad). LoRA
    # also requires enable_input_require_grads so the activation graph
    # reaches the trainable adapters under checkpointing.
    if cfg.get("gradient_checkpointing", True):
        policy.config.use_cache = False
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()
    policy.train()

    # Reference is a frozen second copy of the SAME base — never updated.
    # The reference holds the policy's prior; DPO measures policy's deviation
    # from it. Loading twice trades VRAM for memory-shared correctness.
    print("loading frozen reference (second copy of base):")
    ref_model = load_model(base_path, cfg["dtype"], device)
    ref_model.config.use_cache = False  # no KV-cache allocs on ref forward
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # --- optimiser (decay vs no-decay split — GPT-2 recipe)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.dim() >= 2]
    no_decay = [p for p in trainable if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    # --- run state
    bs = cfg["batch_size"]
    accum = cfg["grad_accum"]
    max_steps = cfg["max_steps"]
    warmup = cfg["warmup_steps"]
    eval_interval = cfg["eval_interval"]
    eval_steps = cfg["eval_steps"]
    patience = cfg.get("patience", 5)
    grad_clip = cfg["grad_clip"]
    max_length = cfg["max_length"]
    eval_seed = cfg["seed"]
    loss_fn = dpo_loss if objective == "dpo" else ipo_loss

    best_val, best_step, evals_no_improve, stop_reason = float("inf"), -1, 0, "max_steps"
    ckpt_dir = ROOT / "data/checkpoints" / cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\ntrain run '{cfg['name']}'  (device: {device})")
    print(f"  base       : {cfg['base_model']}")
    print(f"  training   : up to {max_steps:,} steps, batch {bs}x{accum} "
          f"(effective {bs*accum}) max_length {max_length}; "
          f"early-stop patience {patience} evals")
    t0 = time.time()

    train_iter = iter_batches(train_rows, tok, bs, max_length, rng)

    # === EXERCISE START: train-loop ====================================
    # Concept: warmup + constant LR; gradient accumulation; per-batch
    #          DPO/IPO forward through both policy and ref; eval + early-stop.
    # Steps:   1) set LR via lr_at on optim.param_groups
    #          2) optim.zero_grad(set_to_none=True)
    #          3) for accum micro-batches:
    #               (ids_c, lbl_c, attn_c), (ids_r, lbl_r, attn_r) = next(train_iter)
    #               move all 6 tensors to device
    #               with torch.no_grad():
    #                   logp_ref_c = sequence_logprobs(ref_model, ids_c, lbl_c, attn_c)
    #                   logp_ref_r = sequence_logprobs(ref_model, ids_r, lbl_r, attn_r)
    #               logp_pol_c = sequence_logprobs(policy, ids_c, lbl_c, attn_c)
    #               logp_pol_r = sequence_logprobs(policy, ids_r, lbl_r, attn_r)
    #               loss, info = loss_fn(logp_pol_c, logp_pol_r,
    #                                    logp_ref_c, logp_ref_r, beta)
    #               (loss / accum).backward()
    #          4) clip_grad_norm_(trainable, grad_clip); optim.step()
    #          5) every eval_interval steps:
    #               val_loss, pair_acc = evaluate(policy, ref_model, val_rows,
    #                                             tok, bs, max_length,
    #                                             eval_steps, device,
    #                                             eval_seed, objective, beta)
    #               if val_loss < best_val:
    #                   policy.save_pretrained(ckpt_dir.with_suffix(".tmp"))
    #                   os.replace(ckpt_dir.with_suffix(".tmp"), ckpt_dir)
    #                   (PeftModel writes adapter only; full-FT writes full)
    #                   evals_no_improve = 0
    #               else: evals_no_improve += 1
    #               if evals_no_improve >= patience: stop_reason = "early_stop"; break
    # Learning mode: rewrite from the spec. The 03_cpt/train_cpt.py loop is
    # the closest template; the DPO loop differs only in carrying the
    # two-side batch and the two-model forward.
    # --------------------------------------------------------------------
    # NB: Path.with_suffix(".tmp") strips dotted segments (a name like
    # "dpo_v1.0" → "dpo_v1.tmp" silently). Use string concat — same fix as
    # Stage 03/04.
    tmp_dir = Path(str(ckpt_dir) + ".tmp")
    for step in range(1, max_steps + 1):
        for g in optim.param_groups:
            g["lr"] = lr_at(step, cfg["learning_rate"], warmup)
        optim.zero_grad(set_to_none=True)

        running = {"loss": 0.0, "acc": 0.0}
        aborted = False
        for _ in range(accum):
            (ids_c, lbl_c, attn_c), (ids_r, lbl_r, attn_r) = next(train_iter)
            ids_c, lbl_c, attn_c = (t.to(device, non_blocking=True) for t in (ids_c, lbl_c, attn_c))
            ids_r, lbl_r, attn_r = (t.to(device, non_blocking=True) for t in (ids_r, lbl_r, attn_r))

            # Reference forward — no grad, no graph, kept in fp32 logits via
            # sequence_logprobs' internal cast. Done first so the ref tensors
            # live without depending on the policy graph.
            with torch.no_grad():
                logp_ref_c = sequence_logprobs(ref_model, ids_c, lbl_c, attn_c)
                logp_ref_r = sequence_logprobs(ref_model, ids_r, lbl_r, attn_r)

            logp_pol_c = sequence_logprobs(policy, ids_c, lbl_c, attn_c)
            logp_pol_r = sequence_logprobs(policy, ids_r, lbl_r, attn_r)
            loss, info = loss_fn(logp_pol_c, logp_pol_r,
                                 logp_ref_c, logp_ref_r, beta)

            # Catch a non-finite micro-batch loss *before* it pollutes the
            # weights. Otherwise NaN/Inf rides through backward + step() and
            # corrupts the policy for up to eval_interval-1 steps before
            # the val-loss isfinite check finally trips.
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                aborted = True
                stop_reason = (f"non-finite train loss ({loss_val}) at "
                               f"step {step} (accum micro-batch)")
                print(f"  step {step:5d}/{max_steps}  ABORT — {stop_reason}")
                break

            (loss / accum).backward()

            running["loss"] += loss_val / accum
            running["acc"] += info["pair_acc"] / accum

        if aborted:
            break  # outer: skip clip/step/eval and exit train loop
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optim.step()

        if step % eval_interval == 0 or step == max_steps:
            val_loss, val_acc = evaluate(policy, ref_model, val_rows, tok,
                                         bs, max_length, eval_steps, device,
                                         eval_seed, objective, beta)
            sft_ce = evaluate_sft_preservation(policy, sft_val_rows, tok,
                                               bs, max_length, eval_steps,
                                               device, eval_seed)
            wall = (time.time() - t0) / 60
            sft_str = f"sft_val_ce {sft_ce:.3f}" if not math.isnan(sft_ce) else "sft_val_ce —"
            print(f"  step {step:5d}/{max_steps}  "
                  f"train_loss {running['loss']:.4f}  "
                  f"train_acc {running['acc']:.2f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.2f}  "
                  f"{sft_str}  "
                  f"({wall:.1f} min)")

            if not math.isfinite(val_loss):
                stop_reason = f"non-finite val loss ({val_loss}) at step {step}"
                break

            if val_loss < best_val:
                best_val, best_step = val_loss, step
                evals_no_improve = 0
                # Atomic save: write to .tmp, swap. Same pattern as Stage 04.
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                policy.save_pretrained(str(tmp_dir))
                tok.save_pretrained(str(tmp_dir))
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                os.rename(tmp_dir, ckpt_dir)
            else:
                evals_no_improve += 1

            if evals_no_improve >= patience:
                stop_reason = (f"early-stopped — val loss did not improve "
                               f"for {patience} evals")
                break
    # === EXERCISE END: train-loop ======================================

    dt = (time.time() - t0) / 60
    print(f"\n  done in {dt:.1f} min  ({stop_reason})")
    print(f"  best val   : loss {best_val:.4f}  at step {best_step}")
    print(f"  checkpoint : {ckpt_dir}")


if __name__ == "__main__":
    main()
