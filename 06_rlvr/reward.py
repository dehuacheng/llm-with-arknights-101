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


def _on_topic_char_fraction(response: str) -> float:
    """Fraction of response characters that are either CJK or Latin-ASCII
    alphabetic. The intent is "what fraction looks like Chinese-or-English
    prose" — used as a fluency proxy. Excludes spaces, punctuation,
    digits, and any non-CJK-non-Latin scripts (Cyrillic, Japanese kana,
    etc — the noise that mode-collapsed under grpo_baseline). Empty
    response returns 1.0 (no penalty)."""
    stripped = response.strip()
    if not stripped:
        return 1.0
    n_cjk = sum(1 for c in stripped if "一" <= c <= "鿿")
    # ASCII letters only — digits, punctuation, whitespace excluded from
    # numerator AND denominator below; only "graphical content" counts.
    n_latin = sum(1 for c in stripped if c.isascii() and c.isalpha())
    n_graphical = sum(1 for c in stripped if c.isalnum() or _is_cjk(c))
    if n_graphical == 0:
        return 1.0
    return (n_cjk + n_latin) / n_graphical


def _is_cjk(c: str) -> bool:
    return "一" <= c <= "鿿"


def reward(response: str, item: dict[str, Any], *,
           length_penalty_threshold: float = 2.0,
           length_penalty_rate: float = 0.1,
           trap_weight: float = 0.5,
           fluency_threshold: float = 0.7,
           fluency_penalty_cap: float = 0.0,  # off by default; grpo_v2 turns it on
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
      fluency_threshold: penalty fires when the fraction of "on-topic"
        (CJK or Latin alphabetic) characters drops below this. Set to
        0.0 to disable the fluency penalty entirely (recovers the
        Stage-06-baseline reward shape).
      fluency_penalty_cap: maximum fluency penalty when on-topic fraction
        is 0. Scales linearly between (threshold, 0).
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

    # Fluency penalty — added after grpo_baseline's mode-collapse failure.
    # The baseline reward function was indifferent to whether the response
    # was fluent: "凯尔希博士的种族是菲林 垾垾垾" scored +1.0 because the
    # key_fact substring was present. The collapse spot-check log
    # (data/rl_logs/grpo_baseline_responses.jsonl) showed Russian / kana /
    # garbage tokens leaking in by step 50; the gradient had no reason to
    # avoid them. The penalty is OFF by default (cap=0) unless the
    # config sets fluency_penalty_cap > 0 — keeps the unit-test contract
    # for the original baseline shape.
    on_topic_frac = _on_topic_char_fraction(response)
    if fluency_penalty_cap > 0 and on_topic_frac < fluency_threshold:
        deficit = (fluency_threshold - on_topic_frac) / fluency_threshold
        fluency_pen = fluency_penalty_cap * deficit
    else:
        fluency_pen = 0.0

    raw = base - trap - length_pen - fluency_pen
    clamped = max(-1.0, min(1.0, raw))

    debug = {
        "base": base,
        "trap": trap,
        "length_pen": length_pen,
        "fluency_pen": fluency_pen,
        "on_topic_frac": on_topic_frac,
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
