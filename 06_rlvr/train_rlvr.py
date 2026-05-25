#!/usr/bin/env python3
"""Stage 06 — RLVR / GRPO with verifiable rewards.

Reads agent-produced RL prompts from data/rl/prompts_train.jsonl (and val
split from prompts_val.jsonl); samples N completions per prompt from the
policy; scores each sample programmatically against `key_facts` /
`must_not_contain` (see reward.py); applies group-relative advantage
(z-score within the group) and a PPO-clipped surrogate with KL penalty
to a frozen reference (Stage 04 SFT). Hand-rolled — not `trl.GRPOTrainer`.

    python3 06_rlvr/train_rlvr.py --config 06_rlvr/configs/grpo_baseline.yaml
    python3 06_rlvr/train_rlvr.py --config <any>             --smoke-test

Checkpoints land at data/checkpoints/<run>/ (LoRA adapter, 5-20 MB; sits
on data/checkpoints/sft_full). Probe with:
    .venv/bin/python 04_sft/chat.py --adapter data/checkpoints/<run> --probes

Pedagogically central regions wear EXERCISE markers. In learning mode you
delete the body between the markers and rewrite from the spec.

Failure-mode design + hyperparameter rationale: 06_rlvr/README.md.
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

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "06_rlvr"))
from reward import reward as reward_fn  # noqa: E402

# --smoke-test overrides: tiny everything, LoRA forced on, ~8 min on a 4090.
SMOKE = dict(
    n_samples=4, batch_size=1, grad_accum=2, max_steps=20, warmup_steps=2,
    eval_interval=10, eval_samples_per_val_prompt=2, patience=2,
    max_new_tokens=64, force_lora=True,
)
SMOKE_TRAIN_ROWS, SMOKE_VAL_ROWS = 16, 8


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


def filter_out_val_ids(train_rows: list[dict], val_rows: list[dict]) -> list[dict]:
    """The agent's source of truth is prompts_train.jsonl; val is derived
    from a subset of it. Train rows whose id is in val are dropped here so
    we never train on a val prompt."""
    val_ids = {r["id"] for r in val_rows}
    return [r for r in train_rows if r["id"] not in val_ids]


def iter_prompts(rows: list[dict], rng):
    """Infinite iterator over prompts — one row per yield. The GRPO step
    consumes batch_size prompts per micro-step and grad_accum micro-steps,
    yielding (batch_size * grad_accum) prompts per gradient update."""
    while True:
        order = rng.permutation(len(rows))
        for i in order:
            yield rows[int(i)]


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
# EXERCISE blocks
# ---------------------------------------------------------------------------

# === EXERCISE START: grpo-sample =====================================
# Concept: for one prompt, draw N independent completions from the
#          *current* policy at the configured temperature, and cache the
#          per-response-token log-probs of the sampled tokens under the
#          policy at sample time. These cached log-probs are the
#          "old log-probs" the PPO clip ratio is built against in
#          grpo-loss; they are NOT recomputed in the loss path.
# Given:   policy, tok, prompt_messages (list[{role,content}]), n_samples,
#          temperature, top_p, max_new_tokens, device
# Produce: list of N dicts, each:
#     {"prompt_ids": LongTensor (P,)  on device,
#      "response_ids": LongTensor (R,) on device,
#      "old_logprobs": Tensor (R,) on device, fp32,
#      "response_text": str}
# Steps:   1) Render the prompt:
#               prompt_text = tok.apply_chat_template(prompt_messages,
#                              tokenize=False, add_generation_prompt=True)
#          2) Tokenise → prompt_ids (1, P); expand to (N, P) for parallel gen.
#          3) policy.eval(); use_cache=True; with torch.no_grad():
#               out = policy.generate(input_ids=..., attention_mask=...,
#                       do_sample=True, temperature=T, top_p=top_p,
#                       max_new_tokens=max_new_tokens,
#                       return_dict_in_generate=True, output_scores=True,
#                       pad_token_id=pad_id, eos_token_id=[eos, im_end])
#          4) Per sample: slice response_ids from sequences[i, P:];
#             find the first eos/im_end → end index (inclusive); truncate.
#          5) old_logprobs[t] = log_softmax(out.scores[t][i].float())[response_ids[t]]
#             for t in [0, end). Stack into a (R,) tensor.
#          6) Detokenise: response_text = tok.decode(response_ids,
#                                                    skip_special_tokens=True)
#          7) Detach everything; free out.scores buffer + torch.cuda.empty_cache().
# Memory: out.scores is the biggest single object (R × N × V × 4B fp32 if cast).
#         Free it as soon as old_logprobs are gathered, before returning.
# Notes:  policy stays in .eval() during sampling (no_grad means dropout off
#         too — fine; sample-time and loss-time forwards both want dropout off).
# ---------------------------------------------------------------------
def grpo_sample(policy, tok, prompt_messages: list[dict], n_samples: int,
                temperature: float, top_p: float, max_new_tokens: int,
                device: torch.device) -> list[dict]:
    prompt_text = tok.apply_chat_template(prompt_messages, tokenize=False,
                                          add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False,
                     return_tensors="pt").input_ids.to(device)  # (1, P)
    P = prompt_ids.shape[1]

    # Build stop-token list: standard eos plus Qwen3's chat-template <|im_end|>
    # (a separate ID; the model emits it to close the assistant turn).
    eos_ids: list[int] = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tok.unk_token_id and im_end not in eos_ids:
        eos_ids.append(im_end)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Generate N samples in parallel. (N, P) → (N, P + R_max).
    input_batch = prompt_ids.expand(n_samples, -1).contiguous()
    attn_batch = torch.ones_like(input_batch)

    was_training = policy.training
    policy.eval()
    # use_cache must be True for generate. Stage 05's grad-checkpoint setup
    # had use_cache=False; restore here for sampling only.
    prev_use_cache = policy.config.use_cache
    policy.config.use_cache = True
    try:
        with torch.no_grad():
            out = policy.generate(
                input_ids=input_batch,
                attention_mask=attn_batch,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=eos_ids,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        policy.config.use_cache = prev_use_cache
        if was_training:
            policy.train()

    sequences = out.sequences  # (N, P + R_max)
    R_max = sequences.shape[1] - P

    samples = []
    for i in range(n_samples):
        # Find response end at first eos/im_end (inclusive); fall through to
        # R_max if no stop token was emitted within the budget.
        response = sequences[i, P:].clone()  # (R_max,)
        end = R_max
        for t in range(R_max):
            tok_id = response[t].item()
            if tok_id in eos_ids:
                end = t + 1
                break
        response_ids = response[:end]  # (R,)

        # Old log-probs: cast scores to fp32 (Qwen3 151K-vocab pressure),
        # log_softmax, gather the sampled token. Per-token loop is fine —
        # R is bounded by max_new_tokens (≤128 typically); the dominant
        # cost is the log_softmax over V=151936 per token.
        old_lp = torch.empty(end, dtype=torch.float32, device=device)
        for t in range(end):
            log_p = F.log_softmax(out.scores[t][i].float(), dim=-1)
            old_lp[t] = log_p[response_ids[t]]

        response_text = tok.decode(response_ids, skip_special_tokens=True)
        samples.append({
            "prompt_ids": prompt_ids[0].clone(),
            "response_ids": response_ids,
            "old_logprobs": old_lp,
            "response_text": response_text,
        })

    # Drop the (R_max, N, V) scores buffer. Without this, subsequent forwards
    # under gradient checkpointing OOM.
    del out
    torch.cuda.empty_cache()
    return samples
# === EXERCISE END: grpo-sample =======================================


# === EXERCISE START: grpo-reward =====================================
# Concept: score each of the N sampled responses against the prompt's
#          key_facts / must_not_contain via the verifiable reward
#          function (reward.py). Pure CPU; deterministic; unit-tested.
# Given:   samples (list of dicts from grpo-sample), item (the RL prompt
#          row, with key_facts + must_not_contain), reward_kwargs
#          (length_penalty_threshold, length_penalty_rate, trap_weight,
#           refusal_phrases)
# Produce: rewards: Tensor (N,) fp32 on cpu, debug: list[dict]
# Note: the entire scoring path is offline-reproducible — you can re-run
# reward() on the logged responses post-hoc to A/B different reward
# weights without re-training. That's the verifiable-reward design payoff.
# ---------------------------------------------------------------------
def grpo_reward(samples: list[dict], item: dict, **reward_kwargs
                ) -> tuple[torch.Tensor, list[dict]]:
    out, dbg = [], []
    for s in samples:
        r, d = reward_fn(s["response_text"], item, **reward_kwargs)
        out.append(r)
        dbg.append(d)
    return torch.tensor(out, dtype=torch.float32), dbg
# === EXERCISE END: grpo-reward =======================================


# === EXERCISE START: grpo-advantage ==================================
# Concept: GRPO replaces PPO's critic with the in-group reward statistics.
#          Per-sample advantage is the z-score of the reward within the
#          group of N samples for one prompt.
# Given:   rewards (N,)
# Produce: advantages (N,)
# Steps:   1) mean = rewards.mean()
#          2) std  = rewards.std(unbiased=False)
#          3) advantages = (rewards - mean) / (std + 1e-8)
# Edge:    if all N rewards are equal (group collapsed, std=0), advantages
#          are all 0 — that prompt contributes no gradient. Correct; log
#          the fraction of "dead groups" as a diversity tripwire.
# ---------------------------------------------------------------------
def grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + 1e-8)
# === EXERCISE END: grpo-advantage ====================================


# === EXERCISE START: grpo-loss =======================================
# Concept: PPO-clipped surrogate + KL penalty on a group of N samples.
#          For each sampled response:
#            ratio_t  = exp(logπ_pol(t) - old_logprobs(t))
#            surr1_t  = ratio_t * A                        # A from grpo-advantage
#            surr2_t  = clip(ratio_t, 1-ε, 1+ε) * A
#            policy_loss_t = -min(surr1_t, surr2_t)
#            kl_t          = logπ_pol(t) - logπ_ref(t)     # k1 estimator
#            per_token     = policy_loss_t + β · kl_t
#          Reduce: mean over masked response tokens per sequence, then
#          mean over N samples in the group. (Outer mean over batch
#          happens in the train-loop via grad_accum.)
# Given:   samples (N dicts: prompt_ids, response_ids, old_logprobs),
#          advantages (N,), policy (trainable), ref_model (frozen),
#          beta, clip_epsilon, device, pad_id
# Produce: loss (scalar tensor with grad), info dict
# Notes:   - Logits cast to fp32 before log_softmax — same Qwen3 151K-vocab
#            pressure as Stages 03-05.
#          - Forward fires twice per group (policy + ref); plus the
#            sampling forward already happened in grpo-sample. 3× per-token
#            compute per training step.
#          - **Single-step GRPO**: old_logprobs come from the same policy
#            forward as the loss path (no inner update-loop), so ratio ≈ 1
#            on entry and the clip rarely fires. clipfrac ≈ 0 is correct.
#            If extending to multi-step, you must recompute old_logprobs
#            after each inner step or document the staleness.
# ---------------------------------------------------------------------
def grpo_loss(samples: list[dict], advantages: torch.Tensor,
              policy, ref_model, beta: float, clip_epsilon: float,
              device: torch.device, pad_id: int
              ) -> tuple[torch.Tensor, dict]:
    n = len(samples)
    # Same prompt for all N samples in a group, so P is constant.
    P = samples[0]["prompt_ids"].shape[0]
    R_max = max(s["response_ids"].shape[0] for s in samples)
    T = P + R_max

    ids = torch.full((n, T), pad_id, dtype=torch.long, device=device)
    response_mask = torch.zeros((n, T), dtype=torch.bool, device=device)
    old_lp_pad = torch.zeros((n, R_max), dtype=torch.float32, device=device)

    for i, s in enumerate(samples):
        prompt_ids = s["prompt_ids"]
        resp_ids = s["response_ids"]
        R = resp_ids.shape[0]
        ids[i, :P] = prompt_ids
        ids[i, P:P + R] = resp_ids
        response_mask[i, P:P + R] = True
        old_lp_pad[i, :R] = s["old_logprobs"]

    attn = (ids != pad_id).long()
    # Edge: a response token that happens to equal pad_id would be masked
    # out of attention. Vanishingly unlikely for Qwen3 (pad = eos), but
    # we restore those positions to attended explicitly.
    attn = torch.maximum(attn, response_mask.long())

    # Policy forward (with grad). Gradient checkpointing on (handled at
    # model setup) — activations are recomputed on backward.
    pol_out = policy(input_ids=ids, attention_mask=attn)
    pol_logits = pol_out.logits.float()  # (N, T, V)

    # Reference forward (frozen, no grad, no graph). The fp32 logits cast
    # is the binding VRAM constraint here as in Stages 03-05.
    with torch.no_grad():
        ref_out = ref_model(input_ids=ids, attention_mask=attn)
        ref_logits = ref_out.logits.float()  # (N, T, V)

    # logits[:, t, :] predicts ids[:, t+1] — shift to align.
    shift_pol = pol_logits[:, :-1, :]   # (N, T-1, V)
    shift_ref = ref_logits[:, :-1, :]
    shift_ids = ids[:, 1:]              # (N, T-1)
    shift_mask = response_mask[:, 1:]   # (N, T-1)

    log_pol_all = F.log_softmax(shift_pol, dim=-1)
    log_ref_all = F.log_softmax(shift_ref, dim=-1)
    log_pol = log_pol_all.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
    log_ref = log_ref_all.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)

    # Align old_logprobs to shift_ids: response_ids[t] sits at ids[:, P+t]
    # which is shift_ids[:, P-1+t]. So old_lp_pad[:, :R_max] slots at
    # shift positions [P-1, P-1+R_max).
    old_lp_shifted = torch.zeros_like(log_pol)
    old_lp_shifted[:, P - 1:P - 1 + R_max] = old_lp_pad

    ratio = torch.exp(log_pol - old_lp_shifted)  # (N, T-1)
    A = advantages.to(device).unsqueeze(-1)      # (N, 1)
    surr1 = ratio * A
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * A
    policy_loss_per_tok = -torch.min(surr1, surr2)
    kl_per_tok = log_pol - log_ref

    per_tok = policy_loss_per_tok + beta * kl_per_tok

    # Reduce: per-sequence mean over response tokens, then mean over group.
    mask_f = shift_mask.float()
    tok_counts = mask_f.sum(dim=-1).clamp(min=1)  # (N,) — never divide by 0
    per_seq = (per_tok * mask_f).sum(dim=-1) / tok_counts
    loss = per_seq.mean()

    with torch.no_grad():
        total_mask = mask_f.sum().clamp(min=1)
        mean_kl = (kl_per_tok * mask_f).sum() / total_mask
        mean_ratio = (ratio * mask_f).sum() / total_mask
        clipfrac = ((torch.abs(ratio - 1.0) > clip_epsilon).float() * mask_f).sum() / total_mask
        mean_policy_loss = (policy_loss_per_tok * mask_f).sum() / total_mask

    return loss, {
        "policy_loss": mean_policy_loss.item(),
        "kl": mean_kl.item(),
        "clipfrac": clipfrac.item(),
        "ratio_mean": mean_ratio.item(),
    }
# === EXERCISE END: grpo-loss =========================================


# ---------------------------------------------------------------------------
# LR schedule + evaluation
# ---------------------------------------------------------------------------

def lr_at(step: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    return base_lr


@torch.no_grad()
def evaluate_reward(policy, tok, val_rows, cfg, device, n_samples_per_prompt: int,
                    reward_kwargs: dict) -> dict:
    """Mean reward + per-category breakdown over val prompts. Cheap because
    we use a small n_samples_per_prompt (typically 4) regardless of training N."""
    by_cat: dict[str, list[float]] = {}
    all_rewards, all_lens = [], []
    n_dead = 0  # groups where all sampled rewards are equal

    for item in val_rows:
        samples = grpo_sample(
            policy, tok, item["prompt"], n_samples_per_prompt,
            cfg["temperature"], cfg["top_p"], cfg["max_new_tokens"], device,
        )
        rewards, _ = grpo_reward(samples, item, **reward_kwargs)
        r_list = rewards.tolist()
        by_cat.setdefault(item.get("category", "?"), []).extend(r_list)
        all_rewards.extend(r_list)
        all_lens.extend(len(s["response_text"]) for s in samples)
        if max(r_list) - min(r_list) < 1e-6:
            n_dead += 1

    return {
        "val_reward": sum(all_rewards) / max(1, len(all_rewards)),
        "mean_response_len": sum(all_lens) / max(1, len(all_lens)),
        "dead_group_frac": n_dead / max(1, len(val_rows)),
        "by_category": {c: sum(rs) / len(rs) for c, rs in by_cat.items()},
    }


# --- SFT-preservation tripwire (same as Stage 05; see 05_dpo/train_dpo.py) ---
# Catches the "GRPO bleeds into off-target weights" mode by re-scoring the
# SFT val set under the policy and watching for the assistant-token CE
# drifting upward from the Stage-04 baseline of ~2.96.

def _format_sft_row(tok, row: dict, max_length: int):
    msgs = row["messages"]
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                          add_generation_prompt=True)
    full_text = tok.apply_chat_template(msgs, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    # Prefix invariant — same gotcha as Stages 04 / 05. Raise loud rather
    # than silently shift the label mask.
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "SFT-preservation eval: chat-template prompt-only render is not "
            "a prefix of the full render; tokenizer/template mismatch")
    labels = list(full_ids)
    for i in range(len(prompt_ids)):
        labels[i] = -100
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
    if all(l == -100 for l in labels):
        return None  # right-truncation wiped all response tokens
    return (torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


@torch.no_grad()
def evaluate_sft_preservation(policy, sft_val_rows, tok, batch_size, max_length,
                              n_batches, device) -> float:
    if not sft_val_rows:
        return float("nan")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    was_training = policy.training
    policy.eval()
    prev_use_cache = policy.config.use_cache
    policy.config.use_cache = False
    losses = []
    try:
        n = min(n_batches, len(sft_val_rows) // batch_size)
        for i in range(n):
            rows = sft_val_rows[i * batch_size:(i + 1) * batch_size]
            formatted = [_format_sft_row(tok, r, max_length) for r in rows]
            batch = [b for b in formatted if b is not None]
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
    finally:
        policy.config.use_cache = prev_use_cache
        if was_training:
            policy.train()
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
        cfg["train_file"] = "06_rlvr/smoke_prompts.jsonl"
        cfg["val_file"] = "06_rlvr/smoke_prompts.jsonl"
        if SMOKE.get("force_lora") and not cfg.get("lora", {}).get("enabled"):
            cfg["lora"] = {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.0,
                           "target_modules": ["q_proj", "v_proj"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    beta = float(cfg["beta"])
    clip_epsilon = float(cfg["clip_epsilon"])
    n_samples = int(cfg["n_samples"])

    reward_kwargs = dict(
        length_penalty_threshold=float(cfg["length_penalty_threshold"]),
        length_penalty_rate=float(cfg["length_penalty_rate"]),
        trap_weight=float(cfg["trap_weight"]),
        refusal_phrases=tuple(cfg.get("refusal_phrases", [])) or None,
    )
    # reward_fn's default is used when refusal_phrases is None
    if reward_kwargs["refusal_phrases"] is None:
        del reward_kwargs["refusal_phrases"]

    # --- data ---
    train_path = ROOT / cfg["train_file"]
    val_path = ROOT / cfg["val_file"]
    raw_train_rows = load_jsonl(
        train_path, limit=SMOKE_TRAIN_ROWS if args.smoke_test else None,
    )
    val_rows = load_jsonl(
        val_path, limit=SMOKE_VAL_ROWS if args.smoke_test else None,
    )
    train_rows = filter_out_val_ids(raw_train_rows, val_rows) if not args.smoke_test else raw_train_rows
    print(f"data: {len(train_rows):,} train prompts, {len(val_rows):,} val prompts")
    print(f"        (loaded {len(raw_train_rows):,} from {train_path.relative_to(ROOT)}; "
          f"dropped {len(raw_train_rows)-len(train_rows)} val overlaps)")

    # --- SFT-preservation eval (README §5 tripwire) ---
    sft_val_path = ROOT / "data/sft/qa_val.jsonl"
    sft_val_rows = (load_jsonl(sft_val_path,
                               limit=SMOKE_VAL_ROWS if args.smoke_test else None)
                    if sft_val_path.exists() else [])
    if sft_val_rows:
        print(f"sft-preservation: {len(sft_val_rows):,} rows from {sft_val_path.relative_to(ROOT)}")
    else:
        print(f"sft-preservation: skipped ({sft_val_path} not found)")

    # --- tokenizer + models (policy + frozen reference) ---
    from transformers import AutoTokenizer
    base_path = ROOT / cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(str(base_path))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    policy = load_model(base_path, cfg["dtype"], device)
    if cfg.get("lora", {}).get("enabled"):
        policy = inject_lora(policy, cfg["lora"])

    # Gradient checkpointing on the policy. Same Qwen3 151K-vocab fp32-
    # logits-cast pressure as Stages 03-05; here the loss does 2 forwards
    # per group (policy + ref) plus the sampling forward already happened.
    # LoRA + grad-checkpointing requires enable_input_require_grads.
    if cfg.get("gradient_checkpointing", True):
        policy.config.use_cache = False
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(policy, "enable_input_require_grads"):
            policy.enable_input_require_grads()
    policy.train()

    print("loading frozen reference (second copy of base):")
    ref_model = load_model(base_path, cfg["dtype"], device)
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # --- optimiser (decay vs no-decay split — GPT-2 recipe) ---
    trainable = [p for p in policy.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.dim() >= 2]
    no_decay = [p for p in trainable if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    bs = cfg["batch_size"]
    accum = cfg["grad_accum"]
    max_steps = cfg["max_steps"]
    warmup = cfg["warmup_steps"]
    eval_interval = cfg["eval_interval"]
    n_eval_samples = cfg["eval_samples_per_val_prompt"]
    patience = cfg.get("patience", 5)
    grad_clip = cfg["grad_clip"]
    kl_cap = float(cfg.get("kl_hard_cap", float("inf")))
    kl_max_cap = float(cfg.get("kl_max_hard_cap", float("inf")))
    sft_drift_cap = float(cfg.get("sft_drift_hard_cap", float("inf")))
    sft_baseline = 2.96  # Stage 04 sft_full's val CE; used for drift comparison

    best_val_r, best_step, evals_no_improve, stop_reason = -float("inf"), -1, 0, "max_steps"
    ckpt_dir = ROOT / "data/checkpoints" / cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "data/rl_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    responses_log = log_dir / f"{cfg['name']}_responses.jsonl"
    if not args.smoke_test and responses_log.exists():
        responses_log.unlink()  # fresh per run

    print(f"\ntrain run '{cfg['name']}'  (device: {device})")
    print(f"  base       : {cfg['base_model']}")
    print(f"  training   : up to {max_steps:,} steps, "
          f"batch {bs}x{accum} (effective {bs*accum} prompts/step), "
          f"N={n_samples} samples/prompt → {bs*accum*n_samples} sequences/step")
    print(f"  reward     : trap_weight={reward_kwargs['trap_weight']}, "
          f"len_pen_rate={reward_kwargs['length_penalty_rate']}")
    print(f"  loss       : β={beta} (KL), ε={clip_epsilon} (clip)")
    print(f"  early-stop : patience {patience} evals, "
          f"hard caps: mean_kl>{kl_cap}, max_kl>{kl_max_cap}, "
          f"sft_drift>{sft_drift_cap}")
    t0 = time.time()
    train_iter = iter_prompts(train_rows, rng)

    # === EXERCISE START: train-loop ====================================
    # Concept: warmup + constant LR; gradient accumulation; per-prompt
    #          GRPO update (sample → reward → advantage → loss → backward);
    #          eval interval; early-stop on val_reward patience OR KL
    #          hard-cap OR SFT-drift hard-cap.
    # Steps:   1) set LR via lr_at on optim.param_groups
    #          2) optim.zero_grad(set_to_none=True)
    #          3) for _ in range(grad_accum):
    #               for _ in range(batch_size):
    #                 item = next(train_iter)
    #                 samples = grpo_sample(policy, tok, item["prompt"],
    #                                        n_samples, temperature, top_p,
    #                                        max_new_tokens, device)
    #                 rewards, dbg = grpo_reward(samples, item, **reward_kwargs)
    #                 advantages = grpo_advantage(rewards)
    #                 loss, info = grpo_loss(samples, advantages, policy,
    #                                         ref_model, beta, clip_epsilon,
    #                                         device, pad_id)
    #                 if not math.isfinite(loss.item()): aborted = True; break
    #                 (loss / (grad_accum * batch_size)).backward()
    #          4) clip_grad_norm_(trainable, grad_clip); optim.step()
    #          5) every eval_interval steps:
    #               metrics = evaluate_reward(policy, tok, val_rows, cfg,
    #                                          device, n_eval_samples,
    #                                          reward_kwargs)
    #               sft_ce = evaluate_sft_preservation(...)
    #               print headline line
    #               apply hard-caps → break with stop_reason
    #               if metrics["val_reward"] > best: save (atomic), reset patience
    #               else: bump patience; if patience exhausted → break
    # Atomic checkpoint save: tmp_dir = Path(str(ckpt_dir) + ".tmp")
    # (NOT .with_suffix — strips dotted segments; see Stage 03/04/05).
    # ------------------------------------------------------------------
    tmp_dir = Path(str(ckpt_dir) + ".tmp")
    for step in range(1, max_steps + 1):
        for g in optim.param_groups:
            g["lr"] = lr_at(step, cfg["learning_rate"], warmup)
        optim.zero_grad(set_to_none=True)

        running = {"loss": 0.0, "reward": 0.0, "kl": 0.0, "clipfrac": 0.0,
                   "n_groups": 0, "n_dead": 0}
        aborted = False
        for _ in range(accum):
            for _ in range(bs):
                item = next(train_iter)
                samples = grpo_sample(
                    policy, tok, item["prompt"], n_samples,
                    cfg["temperature"], cfg["top_p"], cfg["max_new_tokens"], device,
                )
                rewards, _ = grpo_reward(samples, item, **reward_kwargs)
                advantages = grpo_advantage(rewards)
                loss, info = grpo_loss(
                    samples, advantages, policy, ref_model,
                    beta, clip_epsilon, device, pad_id,
                )

                loss_val = loss.item()
                if not math.isfinite(loss_val):
                    aborted = True
                    stop_reason = (f"non-finite train loss ({loss_val}) at "
                                   f"step {step} (group micro-batch)")
                    print(f"  step {step:5d}/{max_steps}  ABORT — {stop_reason}")
                    break

                # Scale by the number of groups in a gradient update so
                # the effective gradient is the mean over (grad_accum * bs).
                (loss / (accum * bs)).backward()

                running["loss"] += loss_val / (accum * bs)
                running["reward"] += rewards.mean().item() / (accum * bs)
                running["kl"] += info["kl"] / (accum * bs)
                running["clipfrac"] += info["clipfrac"] / (accum * bs)
                running["n_groups"] += 1
                if (rewards.max() - rewards.min()).item() < 1e-6:
                    running["n_dead"] += 1
            if aborted:
                break

        if aborted:
            break

        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optim.step()

        if step % eval_interval == 0 or step == max_steps:
            metrics = evaluate_reward(policy, tok, val_rows, cfg, device,
                                      n_eval_samples, reward_kwargs)
            sft_ce = evaluate_sft_preservation(
                policy, sft_val_rows, tok, batch_size=4, max_length=1024,
                n_batches=8 if args.smoke_test else 32, device=device,
            )
            wall = (time.time() - t0) / 60
            sft_str = (f"sft_ce {sft_ce:.3f} ({sft_ce - sft_baseline:+.2f})"
                       if not math.isnan(sft_ce) else "sft_ce —")
            dead_train = running["n_dead"] / max(1, running["n_groups"])
            print(f"  step {step:5d}/{max_steps}  "
                  f"train_r {running['reward']:+.3f}  "
                  f"val_r {metrics['val_reward']:+.3f}  "
                  f"kl {running['kl']:+.3f}  "
                  f"dead {dead_train:.2f}/{metrics['dead_group_frac']:.2f}  "
                  f"resp_len {metrics['mean_response_len']:.0f}  "
                  f"{sft_str}  "
                  f"({wall:.1f} min)")

            # Spot-check artefact: log a few top-reward responses per eval.
            if not args.smoke_test:
                sample_item = val_rows[step % len(val_rows)]
                spot = grpo_sample(policy, tok, sample_item["prompt"], 2,
                                   cfg["temperature"], cfg["top_p"],
                                   cfg["max_new_tokens"], device)
                spot_r, _ = grpo_reward(spot, sample_item, **reward_kwargs)
                with responses_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": step,
                        "id": sample_item["id"],
                        "prompt": sample_item["prompt"][-1]["content"],
                        "samples": [{"response": s["response_text"], "reward": r.item()}
                                    for s, r in zip(spot, spot_r)],
                    }, ensure_ascii=False) + "\n")

            # Hard-cap tripwires (immediate stop).
            if running["kl"] > kl_cap:
                stop_reason = f"hard-cap: mean_kl ({running['kl']:.3f}) > {kl_cap}"
                break
            if not math.isnan(sft_ce) and (sft_ce - sft_baseline) > sft_drift_cap:
                stop_reason = (f"hard-cap: sft drift ({sft_ce - sft_baseline:+.2f}) "
                               f"> {sft_drift_cap}")
                break

            # Best-checkpoint by val_reward (NOT loss — PPO loss isn't directly
            # informative; reward is the primary objective).
            if metrics["val_reward"] > best_val_r:
                best_val_r, best_step = metrics["val_reward"], step
                evals_no_improve = 0
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
                stop_reason = (f"early-stopped — val reward did not improve "
                               f"for {patience} evals")
                break
    # === EXERCISE END: train-loop ======================================

    dt = (time.time() - t0) / 60
    print(f"\n  done in {dt:.1f} min  ({stop_reason})")
    print(f"  best val   : reward {best_val_r:+.4f}  at step {best_step}")
    print(f"  checkpoint : {ckpt_dir}")


if __name__ == "__main__":
    main()
