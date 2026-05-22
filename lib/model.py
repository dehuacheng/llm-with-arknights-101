"""GPT-style decoder-only transformer — implemented from scratch.

Track A's small language model (stage 02). A hand-rolled GPT: token + learned
positional embeddings, pre-norm transformer blocks with multi-head causal
self-attention, and a weight-tied output head. No `transformers` — every
tensor move is visible.

The tensor ops follow the Stanford CS336 convention: named-axis `einops`
(`rearrange` / `einsum`) instead of `.view()` / `.transpose()` / `.reshape()`,
so a shape change reads as the equation it is. `jaxtyping` annotations spell
out the shape each function expects.

Stage 02 (`02_pretrain/train.py`) trains this on the stage-01 tokenizer.

Learning mode: the pedagogically central regions are wrapped in
`# === EXERCISE START/END: <slug> ===` blocks (see AGENTS.md and
01_tokenizer/README.md §7). The committed code is the working reference; a
learner deletes the body and rewrites it from the Concept/Given/Produce/Steps
spec, then checks the delta with `git diff`.

The four EXERCISE blocks here and downstream:
    attention          — scaled-dot-product causal self-attention   (this file)
    transformer-block  — the pre-norm residual wiring               (this file)
    gpt-forward        — embed -> blocks -> logits -> loss           (this file)
    get-batch / train-step / sample-loop                  (02_pretrain/*.py)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor

# Architecture presets — the three model scales swept in stage 02. Each fixes
# only depth and width; the embedding table is sized separately by the
# tokenizer's vocabulary, so a model's *total* parameter count depends on which
# vocab feeds it (see 02_pretrain/README.md). d_ff is 4x d_model throughout.
SCALE_PRESETS = {
    "tiny":  dict(n_layer=4, n_head=4, d_model=256, d_ff=1024),
    "small": dict(n_layer=6, n_head=6, d_model=384, d_ff=1536),
    "large": dict(n_layer=8, n_head=8, d_model=512, d_ff=2048),
}


@dataclass
class GPTConfig:
    """Everything needed to build a GPT. `vocab_size` comes from the trained
    tokenizer; the rest is the architecture (usually via `from_scale`)."""

    vocab_size: int
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.1

    @classmethod
    def from_scale(cls, scale, vocab_size, block_size=512, dropout=0.1):
        """Build a config from a SCALE_PRESETS name plus the tokenizer vocab."""
        if scale not in SCALE_PRESETS:
            raise ValueError(
                f"unknown scale '{scale}'; have {sorted(SCALE_PRESETS)}")
        return cls(vocab_size=vocab_size, block_size=block_size,
                   dropout=dropout, **SCALE_PRESETS[scale])


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention — every head attends only to the past."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.d_model % cfg.n_head != 0:
            raise ValueError(
                f"d_model {cfg.d_model} not divisible by n_head {cfg.n_head}")
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        # One projection produces query, key and value together (3 * d_model).
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Causal mask: position i may attend to 0..i. A registered buffer moves
        # with .to(device) but is not a learned parameter; persistent=False
        # keeps it out of the checkpoint (it is fully derived from block_size).
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size,
                                     dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Float[Tensor, "batch seq d_model"]
                ) -> Float[Tensor, "batch seq d_model"]:
        # === EXERCISE START: attention ========================================
        # Concept: each position builds a query, looks at the key of every
        #   position at or before it, and reads out a weighted mix of their
        #   values. "Causal" = the mask forbids looking ahead. Multi-head =
        #   do this in n_head independent subspaces of width d_head at once.
        # Given:   x (batch, seq, d_model); self.qkv, self.proj; self.n_head,
        #          self.d_head; self.causal_mask (bool, True where allowed);
        #          self.attn_dropout, self.resid_dropout.
        # Produce: the attention output, shape (batch, seq, d_model).
        # Steps:   1) project x to q, k, v and split into heads -> (b h t d)
        #          2) scores = q . k over the head dim, scaled by 1/sqrt(d_head)
        #          3) mask out future positions (set them to -inf)
        #          4) softmax over the key axis -> attention weights; dropout
        #          5) weighted sum of values; merge heads back to d_model
        #          6) output projection, then residual dropout
        # Learning mode: delete the body below and rewrite it from the spec;
        #   the committed code is the reference (`git diff` shows your delta).
        # ----------------------------------------------------------------------
        t = x.shape[1]
        # qkv -> three stacked tensors, each split into heads:
        q, k, v = rearrange(self.qkv(x), "b t (three h d) -> three b h t d",
                            three=3, h=self.n_head)
        scores = einsum(q, k, "b h i d, b h j d -> b h i j") / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.causal_mask[:t, :t], float("-inf"))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        out = einsum(attn, v, "b h i j, b h j d -> b h i d")
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.resid_dropout(self.proj(out))
        # === EXERCISE END: attention ==========================================
        return out


class MLP(nn.Module):
    """Position-wise feed-forward network: d_model -> d_ff -> d_model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Float[Tensor, "batch seq d_model"]
                ) -> Float[Tensor, "batch seq d_model"]:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One pre-norm transformer block: attention sublayer, then MLP sublayer."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: Float[Tensor, "batch seq d_model"]
                ) -> Float[Tensor, "batch seq d_model"]:
        # === EXERCISE START: transformer-block ================================
        # Concept: a block refines a "residual stream". Each sublayer reads a
        #   *normalised* copy of the stream and *adds* its result back — it
        #   never overwrites. Pre-norm (LayerNorm before the sublayer, not
        #   after) is what keeps a deep stack trainable. Two sublayers per
        #   block: attention (mixes across positions) then MLP (per position).
        # Given:   x (batch, seq, d_model); self.ln_1, self.attn, self.ln_2,
        #          self.mlp.
        # Produce: the updated residual stream, same shape as x.
        # Steps:   1) x = x + attn(ln_1(x))
        #          2) x = x + mlp(ln_2(x))
        # Learning mode: delete the body below and rewrite it from the spec;
        #   the committed code is the reference (`git diff` shows your delta).
        # ----------------------------------------------------------------------
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        # === EXERCISE END: transformer-block ==================================
        return x


class GPT(nn.Module):
    """A small decoder-only language model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: the output head reuses the token-embedding matrix.
        # One matrix, two jobs (id -> vector, vector -> id) — fewer parameters
        # and a consistent token geometry. This matters here: at a 32k vocab
        # the embedding table can outweigh the whole transformer.
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the residual-path projections so the residual
        # stream does not blow up with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Int[Tensor, "batch seq"],
                targets: Int[Tensor, "batch seq"] | None = None):
        """Return (logits, loss). `loss` is None when `targets` is None."""
        # === EXERCISE START: gpt-forward ======================================
        # Concept: turn a batch of token-id rows into next-token predictions.
        #   Each id becomes a learned vector; a learned position vector is
        #   added so the model knows *where* each token sits; the transformer
        #   blocks refine that stream; a final norm + the (tied) head turn
        #   each position into a score for every vocabulary token. With
        #   targets, the loss is the next-token cross-entropy.
        # Given:   idx (batch, seq) of token ids; optional targets, same shape;
        #          self.token_emb, self.pos_emb, self.drop, self.blocks,
        #          self.ln_f, self.head; self.cfg.block_size.
        # Produce: logits (batch, seq, vocab_size); loss (scalar) or None.
        # Steps:   1) check seq <= block_size
        #          2) x = drop(token_emb(idx) + pos_emb(positions 0..seq-1))
        #          3) run x through every block, then ln_f
        #          4) logits = head(x)
        #          5) if targets given: cross-entropy of logits vs targets,
        #             flattening (batch, seq) into one axis
        # Learning mode: delete the body below and rewrite it from the spec;
        #   the committed code is the reference (`git diff` shows your delta).
        # ----------------------------------------------------------------------
        t = idx.shape[1]
        if t > self.cfg.block_size:
            raise ValueError(
                f"sequence length {t} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                rearrange(logits, "b t v -> (b t) v"),
                rearrange(targets, "b t -> (b t)"),
                ignore_index=-1,
            )
        # === EXERCISE END: gpt-forward ========================================
        return logits, loss

    def num_params(self, non_embedding=True):
        """Parameter count. `non_embedding=True` drops the token + position
        embedding tables — that is the 'scale' the architecture controls;
        the embedding tables are set by the tokenizer's vocabulary."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel()
            n -= self.pos_emb.weight.numel()
        return n
