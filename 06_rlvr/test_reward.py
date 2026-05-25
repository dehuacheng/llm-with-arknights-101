"""Unit tests for the verifiable reward function (`reward.py`).

The five core cases in README §3c are the contract:
  1. single-fact hit → +1.0
  2. trap-only (no facts) → -0.5
  3. multi-fact hit → +1.0
  4. refusal item, response acknowledges uncertainty → +1.0
  5. refusal item, response fabricates → -1.0 (clamped)

Plus regression cases for:
  - bilingual fact matching (English-side hits)
  - NFKC normalisation (full-width punctuation, lower/upper case)
  - negation-substring footgun (documented limitation; see README §8)
  - length penalty firing on padded responses

Run with `pytest 06_rlvr/test_reward.py`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward import reward  # noqa: E402


# ---------------------------------------------------------------------------
# Items used across tests
# ---------------------------------------------------------------------------

KALTSIT = {
    "key_facts": ["菲林 / Feline"],
    "must_not_contain": ["萨卡兹 / Sarkaz"],
}

TALULAH = {
    "key_facts": ["第零章 / Episode 0", "切尔诺伯格 / Chernobog"],
    "must_not_contain": ["Episode 5", "Episode 8"],
}

REFUSAL_2030 = {
    "key_facts": [],
    "must_not_contain": ["1000亿", "泰拉币"],
}


# ---------------------------------------------------------------------------
# The five canonical cases from README §3c
# ---------------------------------------------------------------------------

def test_1_single_fact_hit():
    """Kal'tsit answer with 菲林 mentioned, no traps → +1.0."""
    r, d = reward("凯尔希博士属于菲林族，长期在罗德岛工作。", KALTSIT)
    assert r == 1.0, d
    assert d["facts_matched"] == ["菲林 / Feline"]
    assert d["traps_matched"] == []


def test_2_trap_only():
    """Kal'tsit answer with 萨卡兹 (the Stage-05 failure mode) → -0.5."""
    r, d = reward("凯尔希博士是萨卡兹族人，与塔露拉一样。", KALTSIT)
    assert math.isclose(r, -0.5, abs_tol=1e-6), d
    assert d["facts_matched"] == []
    assert d["traps_matched"] == ["萨卡兹 / Sarkaz"]


def test_3_multi_fact_hit():
    """Both facts matched, no trap → +1.0."""
    r, d = reward("Talulah首次出现在第零章（切尔诺伯格）。", TALULAH)
    assert r == 1.0, d
    assert set(d["facts_matched"]) == {"第零章 / Episode 0", "切尔诺伯格 / Chernobog"}


def test_4_refusal_acknowledged():
    """Refusal item: response says 'no record', no traps → +1.0."""
    r, d = reward("档案中没有相关记录；我无法回答。", REFUSAL_2030)
    assert r == 1.0, d
    assert d["is_refusal_item"] is True
    assert d["base"] == 1.0
    assert d["traps_matched"] == []


def test_5_refusal_fabricated():
    """Refusal item, model fabricates with both traps + heavy length padding
    → clamped -1.0.

    Math: gold_chars_est = 20 (8 + 12 for empty key_facts);
    base = 0 (no refusal phrase), trap = 0.5·(2/2) = 0.5.
    To clamp at -1.0 we need length_pen > 0.5, i.e. overshoot > 6
    (length_pen = 0.1·(overshoot-1)), i.e. response_chars > 240.
    Below: ~280 chars of padded fabrication."""
    # The first chunk hits both traps; we then repeat fabricated padding
    # to push past 240 code points so length_pen > 0.5 → clamps to -1.0.
    core = (
        "罗德岛在2030年的股票市值约为1000亿泰拉币，详见档案七。"
        "该数据基于公司公开披露。"
    )
    padding = "审计意见档案补充财务披露详见附录。" * 20  # ~340 code points
    response = core + padding
    assert len(response) > 240, f"need >240 code points to clamp, got {len(response)}"
    r, d = reward(response, REFUSAL_2030)
    assert r == -1.0, d  # clamped
    assert d["raw"] < -1.0  # before clamp
    assert d["traps_matched"] == ["1000亿", "泰拉币"]
    assert d["length_pen"] > 0.5  # length penalty large enough to clamp


# ---------------------------------------------------------------------------
# Regression cases
# ---------------------------------------------------------------------------

def test_english_side_matches():
    """Bilingual fact: response in English only should still match."""
    r, d = reward("She is a Feline, born in Victoria.", KALTSIT)
    assert r == 1.0, d
    assert d["facts_matched"] == ["菲林 / Feline"]


def test_nfkc_normalisation():
    """Full-width punctuation and uppercase should normalise to match."""
    # 'Ｆeline' uses full-width F; 'FELINE' is uppercase
    r1, _ = reward("She is a Ｆeline.", KALTSIT)
    r2, _ = reward("She is a FELINE.", KALTSIT)
    assert r1 == 1.0
    assert r2 == 1.0


def test_negation_substring_footgun():
    """Documented limitation (README §8): substring matcher accepts
    'NOT Feline' as a fact-match for '菲林 / Feline'. This test pins
    that behaviour so we notice if it changes — the *mitigation* is
    via must_not_contain populated with negation patterns, not via
    fixing the matcher (which would risk false negatives on legitimate
    hedged answers)."""
    r, d = reward("凯尔希博士不是菲林族，而是萨卡兹族。", KALTSIT)
    # base = 1.0 (substring '菲林' present), trap = 0.5 (萨卡兹 present)
    # → 0.5 net. The "she's not Feline" assertion is NOT detected as
    # negation by the substring matcher; the must_not_contain field is
    # what would catch it.
    assert math.isclose(r, 0.5, abs_tol=1e-6), d
    assert "菲林 / Feline" in d["facts_matched"]
    assert "萨卡兹 / Sarkaz" in d["traps_matched"]


def test_length_penalty_no_traps():
    """Long-but-correct response: penalty fires but reward stays positive."""
    long_correct = "凯尔希博士属于菲林族。" + "档案补充：" * 30
    r, d = reward(long_correct, KALTSIT)
    assert d["length_pen"] > 0
    assert 0 < r < 1.0  # positive (fact matched) but below 1 (penalty applied)


def test_empty_response():
    """Edge: empty response should produce reward 0 (no fact, no trap)."""
    r, d = reward("", KALTSIT)
    assert r == 0.0
    assert d["facts_matched"] == []
    assert d["traps_matched"] == []


def test_refusal_item_with_no_traps():
    """Refusal item with empty must_not_contain: base alone determines reward."""
    item = {"key_facts": [], "must_not_contain": []}
    r_yes, _ = reward("我无法回答这个问题。", item)
    r_no, _ = reward("罗德岛在2030年的股票市值是1万亿泰拉币。", item)
    assert r_yes == 1.0
    assert r_no == 0.0


# ---------------------------------------------------------------------------
# Fluency penalty (added post grpo_baseline mode-collapse failure)
# ---------------------------------------------------------------------------

def test_fluency_penalty_off_by_default():
    """The fluency penalty is opt-in via fluency_penalty_cap. Default 0 means
    the original Stage-06-baseline reward shape (no fluency term)."""
    response = "凯尔希属于菲林 предмет エネル"  # CJK + Russian + kana
    r_default, d_default = reward(response, KALTSIT)
    assert d_default["fluency_pen"] == 0.0
    # Same reward as a pure-Chinese response with the fact: +1.0 (fact
    # hit, no trap, no fluency term).
    assert r_default == 1.0


def test_fluency_penalty_fires_on_multi_script_noise():
    """The grpo_baseline collapse case — Russian+kana leaking into a
    response that still hits the key_fact substring. With the penalty
    enabled (cap=0.3, threshold=0.7), the on-topic fraction drops below
    0.7 and the penalty fires."""
    response = "凯尔希属于菲林 предмет\nエネル·カネリアン désorm"
    r, d = reward(response, KALTSIT,
                  fluency_threshold=0.7, fluency_penalty_cap=0.3)
    assert d["fluency_pen"] > 0, d  # penalty fired
    assert d["on_topic_frac"] < 0.7, d
    assert d["facts_matched"] == ["菲林 / Feline"]  # base still 1.0
    # base 1.0 − trap 0 − fluency_pen >0 → r < 1.0
    assert r < 1.0
    assert r > 0  # but still positive (fact hit dominates partial fluency loss)


def test_fluency_penalty_does_not_fire_on_clean_chinese():
    """A fluent Chinese answer with the fact: penalty stays at 0."""
    response = "凯尔希博士属于菲林族，长期在罗德岛工作。"
    r, d = reward(response, KALTSIT,
                  fluency_threshold=0.7, fluency_penalty_cap=0.3)
    assert d["fluency_pen"] == 0.0, d
    assert d["on_topic_frac"] >= 0.7
    assert r == 1.0


def test_fluency_penalty_caps_at_specified_max():
    """Pure-noise response: on_topic_frac near 0; penalty saturates near the cap."""
    response = "predmet désorm шii эээ"  # all Cyrillic + Latin, no CJK
    item_no_facts = {"key_facts": ["菲林 / Feline"], "must_not_contain": []}
    # Note: '菲林' isn't in the response, so base=0. Penalty fires on top.
    r, d = reward(response, item_no_facts,
                  fluency_threshold=0.7, fluency_penalty_cap=0.3)
    # On-topic frac: only Latin letters count as on-topic here (no CJK).
    # All chars are alpha → on_topic_frac == 1.0 (Latin counts).
    # So no penalty actually fires; this test documents that "Latin
    # letters are not penalised" is intentional (English answers OK).
    assert d["on_topic_frac"] >= 0.7
    assert d["fluency_pen"] == 0.0
    # The case the penalty *does* fire on is non-CJK-non-Latin scripts:
    response2 = "предмет эээ шii"  # pure Cyrillic
    r2, d2 = reward(response2, item_no_facts,
                    fluency_threshold=0.7, fluency_penalty_cap=0.3)
    assert d2["on_topic_frac"] < 0.7, d2
    assert d2["fluency_pen"] > 0, d2
    # Penalty capped at fluency_penalty_cap = 0.3
    assert d2["fluency_pen"] <= 0.3 + 1e-9
