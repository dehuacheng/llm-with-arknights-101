"""Verifiable reward function for Stage 06 RLVR / GRPO.

Pure function, no torch dependency, no side effects. Importable for unit
tests and re-runnable post-hoc on logged response strings to A/B different
reward weights without re-training.

See `README.md` §3 for the design; `test_reward.py` for the 5 canonical
test cases that document the expected semantics.
"""
from __future__ import annotations

import unicodedata
from typing import Any


# Default refusal phrases the reward function looks for on refusal items
# (`key_facts == []`). Override at call time via `refusal_phrases` kwarg.
# Stage 06's RL set is factoid-only so this code path isn't exercised in
# the baseline run — but the agent-enrichment ticket may add refusal items,
# and the test suite hits this branch.
DEFAULT_REFUSAL_PHRASES: tuple[str, ...] = (
    "无法回答",
    "档案中没有",
    "未明确记载",
    "没有相关记录",
    "我不知道",
    "i cannot",
    "i don't have",
    "no record",
    "not in the archive",
)


def _normalise(s: str) -> str:
    """NFKC + lower + strip. CJK characters are unchanged by .lower();
    English/punctuation get folded. Substring matching happens after this."""
    return unicodedata.normalize("NFKC", s).lower().strip()


def _split_fact(fact: str) -> list[str]:
    """Split a bilingual fact like '菲林 / Feline' on '/' and return both
    sides normalised (whitespace-stripped, non-empty)."""
    parts = [p.strip() for p in fact.split("/")]
    return [_normalise(p) for p in parts if p.strip()]


def _fact_matched(fact: str, normalised_response: str) -> bool:
    """A fact matches if ANY of its '/'-separated sub-strings appears in
    the normalised response. Substring match — same as AGENT_BRIEF §6
    promises ('fact in answer')."""
    for sub in _split_fact(fact):
        if sub and sub in normalised_response:
            return True
    return False


def _refusal_phrase_present(normalised_response: str,
                            refusal_phrases: tuple[str, ...]) -> bool:
    return any(_normalise(p) in normalised_response for p in refusal_phrases)


def _gold_char_estimate(key_facts: list[str]) -> int:
    """Rough estimate of how long a 'just answers the question' response
    should be, in characters. = longest sub-fact length + 12-char
    'carrier sentence' allowance. Used to set the length-penalty
    threshold. Falls back to 8 for refusal items."""
    if not key_facts:
        return 8 + 12
    longest_sub = max(
        max((len(s.strip()) for s in f.split("/") if s.strip()), default=0)
        for f in key_facts
    )
    return longest_sub + 12


def reward(response: str, item: dict[str, Any], *,
           length_penalty_threshold: float = 2.0,
           length_penalty_rate: float = 0.1,
           trap_weight: float = 0.5,
           refusal_phrases: tuple[str, ...] = DEFAULT_REFUSAL_PHRASES,
           ) -> tuple[float, dict]:
    """Compute the verifiable reward for one (response, RL prompt item) pair.

    Returns (reward in [-1.0, +1.0], debug dict for logging).

    Args:
      response: the model's generated text (raw, post-detokenisation).
      item: an RL prompt row with at least `key_facts` and `must_not_contain`.
      length_penalty_threshold: penalty activates past this multiple of
        the estimated gold-answer length.
      length_penalty_rate: linear penalty per unit of overshoot past the
        threshold.
      trap_weight: per-trap penalty, normalised by len(must_not_contain).
      refusal_phrases: phrases the reward function reads as "I don't know"
        for the refusal-item branch.
    """
    key_facts: list[str] = list(item.get("key_facts") or [])
    traps: list[str] = list(item.get("must_not_contain") or [])

    norm_response = _normalise(response)

    facts_matched_idx = [i for i, f in enumerate(key_facts)
                         if _fact_matched(f, norm_response)]
    traps_matched_idx = [i for i, t in enumerate(traps)
                         if _fact_matched(t, norm_response)]

    if key_facts:
        base = len(facts_matched_idx) / len(key_facts)
        is_refusal_item = False
    else:
        # Refusal items: reward 1.0 iff response acknowledges uncertainty
        # AND triggers no traps; 0.0 otherwise. The trap penalty below
        # still applies, so a fabricated answer (no refusal phrase,
        # triggers traps) goes net-negative.
        base = 1.0 if _refusal_phrase_present(norm_response, refusal_phrases) else 0.0
        is_refusal_item = True

    if traps:
        trap = trap_weight * (len(traps_matched_idx) / len(traps))
    else:
        trap = 0.0

    gold_chars = _gold_char_estimate(key_facts)
    overshoot = len(response) / (length_penalty_threshold * gold_chars)
    length_pen = length_penalty_rate * max(0.0, overshoot - 1.0)

    raw = base - trap - length_pen
    clamped = max(-1.0, min(1.0, raw))

    debug = {
        "base": base,
        "trap": trap,
        "length_pen": length_pen,
        "raw": raw,
        "facts_matched": [key_facts[i] for i in facts_matched_idx],
        "traps_matched": [traps[i] for i in traps_matched_idx],
        "n_key_facts": len(key_facts),
        "n_traps": len(traps),
        "response_chars": len(response),
        "gold_chars_est": gold_chars,
        "is_refusal_item": is_refusal_item,
    }
    return clamped, debug
