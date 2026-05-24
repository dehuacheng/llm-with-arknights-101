#!/usr/bin/env python3
"""Stage 04 — interactive / probe chat against an SFT'd checkpoint.

Two modes:
  --probes   run a hand-picked battery of probe questions and print answers
  --repl     interactive prompt loop (Ctrl-D to exit)

The model is loaded with `apply_chat_template(..., add_generation_prompt=True)`
exactly as in `04_sft/train_sft.py` so the inference-time prompt matches the
training prompt byte-for-byte.

    .venv/bin/python 04_sft/chat.py --probes
    .venv/bin/python 04_sft/chat.py --repl
    .venv/bin/python 04_sft/chat.py --ckpt data/checkpoints/full_ft_replay --probes  # CPT baseline
    # Stage 05: layer a DPO LoRA adapter on top of the SFT base
    .venv/bin/python 04_sft/chat.py --adapter data/checkpoints/dpo_curated --probes
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = "你是罗德岛的档案管理员。基于公开档案准确回答关于泰拉世界的问题；不确定时坦诚说明。"

# Probes span the SFT distribution (factoid/open-ended/refusal), one general-
# Chinese sanity check (李白) to spot regressions from CPT, and a format check.
PROBES = [
    ("factoid",     "凯尔希医生的种族是什么？"),
    ("factoid",     "阿米娅的身高是多少？"),
    ("open_ended",  "请简要介绍博士与阿米娅的关系。"),
    ("event",       "切尔诺伯格事件中发生了什么？"),
    ("relationship","推进之王和荒地双子的关系是什么？"),
    ("refusal_oob", "罗德岛在2030年的股票市值是多少？"),
    ("general_zh",  "李白是哪个朝代的诗人？"),  # off-corpus, sanity for general Chinese
    ("format",      "你是谁？"),
]


def load(ckpt_path: Path, dtype_str: str, device: torch.device,
         adapter_path: Path | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    print(f"loading base {ckpt_path} (dtype={dtype_str}) ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(ckpt_path))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path), dtype=dtype, attn_implementation="sdpa",
    )
    if adapter_path is not None:
        # Stage 05 DPO LoRA adapter sits on top of the Stage 04 SFT base.
        # The adapter dir holds adapter_config.json + adapter_model.safetensors;
        # PeftModel.from_pretrained merges them onto the live base.
        from peft import PeftModel
        print(f"loading adapter {adapter_path} ...", file=sys.stderr)
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.to(device).eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, user_msg: str, *, max_new_tokens: int = 256,
             temperature: float = 0.0, device: torch.device) -> str:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    # Qwen3's chat template terminates assistant turns with <|im_end|>
    # (id 151645), but tok.eos_token is <|endoftext|> (151643). generate()
    # only stops on eos_token_id by default, so without both tokens the
    # model emits <|im_end|> correctly then generation keeps running into
    # garbage. Pass a list of stop ids — HF accepts that.
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [tok.eos_token_id, im_end_id]
    do_sample = temperature > 0.0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=stop_ids,
    )
    new_tokens = out[0, inputs.input_ids.shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    # Qwen3's chat template prepends an empty `<think>\n\n</think>\n\n` to
    # the assistant turn (thinking-mode boilerplate). Our SFT data didn't
    # include real thinking; strip the empty block so the displayed answer
    # is just the response.
    text = text.strip()
    # Qwen3's chat template prepends `<think>\n\n</think>\n\n` to the assistant
    # turn. Our model didn't reliably learn to emit </think> — it often emits
    # a second <think> instead — so strip greedily: drop everything up to
    # `</think>` if present, otherwise drop all leading `<think>` boilerplate.
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end != -1:
            text = text[end + len("</think>"):].lstrip()
        else:
            # Heuristic: drop any leading run of `<think>\n\n` blocks.
            while text.startswith("<think>"):
                text = text[len("<think>"):].lstrip("\n")
    return text


def cmd_probes(tok, model, device, temperature: float):
    bar = "─" * 72
    for tag, q in PROBES:
        t0 = time.time()
        ans = generate(tok, model, q, temperature=temperature, device=device)
        dt = time.time() - t0
        print(bar)
        print(f"[{tag}]  ({dt:.1f}s)")
        print(f"  Q: {q}")
        print(f"  A: {ans}")
    print(bar)


def cmd_repl(tok, model, device, temperature: float):
    print("REPL mode. Ctrl-D / 'exit' / 'quit' to leave. T=", temperature)
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            print()
            break
        if not q:
            continue
        if q in {"exit", "quit"}:
            break
        t0 = time.time()
        ans = generate(tok, model, q, temperature=temperature, device=device)
        print(f"({time.time()-t0:.1f}s) {ans}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "data/checkpoints/sft_full",
                    help="Base model dir (default sft_full).")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="Optional PEFT adapter dir to layer on the base "
                         "(e.g. data/checkpoints/dpo_curated for Stage 05).")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--temperature", type=float, default=0.0)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probes", action="store_true")
    mode.add_argument("--repl", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load(args.ckpt, args.dtype, device, args.adapter)

    if args.probes:
        cmd_probes(tok, model, device, args.temperature)
    elif args.repl:
        cmd_repl(tok, model, device, args.temperature)


if __name__ == "__main__":
    main()
