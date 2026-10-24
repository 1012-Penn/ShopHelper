"""L1 指标扩展:precision、MRR、宏平均与安全除法。"""
from __future__ import annotations

from scripts.eval_metrics import (
    evidence_precision,
    group_recall_at,
    hard_negative_hit_rate,
    macro_average,
    mrr_at,
    safe_div,
)


def test_safe_div_and_macro_average():
    assert safe_div(1, 2) == 0.5
    assert safe_div(1, 0) is None
    assert macro_average({"a": 1.0, "b": 0.0, "c": None}) == 0.5
    assert macro_average({"a": None}) is None


def test_evidence_precision_denominator():
    assert evidence_precision([1, 2, 3], {1, 3}, cap=10) == 2 / 3
    assert evidence_precision([], {1}) is None
    assert evidence_precision([4, 5], {1, 3}) == 0.0


def test_mrr_and_group_recall_use_coverage_semantics():
    gold = [frozenset({1}), frozenset({2})]
    assert mrr_at([9, 1, 2], gold) == 1 / 3
    assert mrr_at([9, 1], gold) is None
    assert group_recall_at([9, 1], gold) == 0.5
    assert group_recall_at([9, 1, 2], gold) == 1.0


def test_hard_negative_hit_rate_is_binary_per_case():
    assert hard_negative_hit_rate([4, 5]) == 1.0
    assert hard_negative_hit_rate([]) == 0.0
