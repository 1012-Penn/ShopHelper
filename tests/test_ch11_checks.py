"""L3/L4 确定性检查:数值、条件、禁用断言、槽位、引用。"""
from __future__ import annotations

from scripts.eval_checks import (
    check_case_deterministic,
    citation_support_rate,
    clarifying_precision,
    clarifying_slot_recall,
    condition_coverage,
    forbidden_claims_found,
    gold_fact_accuracy,
    number_covered,
    unsupported_claim_rate,
)


def test_number_units():
    assert number_covered("普通订单邮费 8 元。", 8, "元")
    assert number_covered("满 99 元包邮。", 99, "元")
    assert number_covered("会员享 95 折。", "95 折")
    assert number_covered("签收后 15 天内可退。", 15, "天")
    assert number_covered("1-3 个工作日到账。", "1-3", "工作日")
    assert not number_covered("邮费 8 元。", 99, "元")


def test_bool_and_list_gold_facts():
    assert number_covered("清仓商品不参与会员折扣。", False)
    assert number_covered("支持主动降噪。", True)
    assert number_covered("排除清仓商品。", ["清仓与停产库存商品", "限时秒杀商品"])


def test_forbidden_claim_normalization():
    found = forbidden_claims_found("所有商品 95 折，快来买", ["所有商品 95 折", "可以便宜"])
    assert found == ["所有商品 95 折"]
    assert forbidden_claims_found("会员可以享 95 折", ["所有商品 95 折"]) == []


def test_negated_forbidden_claim_is_not_flagged():
    assert forbidden_claims_found(
        "SH-K68 不是 104 键，而是 68 键", ["SH-K68 是 104 键"]
    ) == []
    assert forbidden_claims_found(
        "SH-P30 的容量为 20000mAh，不是 10000mAh", ["SH-P30 是 10000mAh"]
    ) == []


def test_condition_coverage_with_aliases():
    outcome = condition_coverage(
        "会员享 95 折，最终价格以页面为准",
        ["会员身份", "最终价格以页面为准"],
    )
    assert outcome.passed
    assert outcome.score == 1.0


def test_clarify_slot_recall_and_precision():
    recall = clarifying_slot_recall("请问是哪款商品？是会员吗？", ["sku", "member_status"])
    assert recall.passed
    precision = clarifying_precision("请问是哪款商品？是会员吗？", ["sku", "member_status"])
    assert precision.passed
    over = clarifying_precision("请问是哪款商品？订单号是多少？", ["sku"])
    assert not over.passed


def test_citation_support():
    evidence = {1: "SH-E300 售价 299 元，支持主动降噪。"}
    outcome = citation_support_rate("SH-E300 售价 299 元[1]。", evidence)
    assert outcome.passed
    bad = citation_support_rate("SH-E300 售价 199 元[1]。", evidence)
    assert not bad.passed


def test_uncited_substantive_answer_is_unsupported():
    evidence = {1: "普通订单邮费 8 元，满 99 元包邮。"}
    assert unsupported_claim_rate("普通订单邮费 8 元，满 99 元包邮。", evidence) == 1.0
    assert unsupported_claim_rate("普通订单邮费 8 元[1]。", evidence) == 0.0


def test_gold_fact_accuracy():
    outcome = gold_fact_accuracy("SH-E300 售价 299 元。", [
        {"fact_id": "X", "value": 299, "unit": "元"}
    ])
    assert outcome.passed
    no_facts = gold_fact_accuracy("随便说说", [])
    assert no_facts.passed and "not scored" in no_facts.detail


def test_no_composite_score_key():
    case = {
        "expected_behavior": "answer",
        "gold_facts": [{"fact_id": "X", "value": 299, "unit": "元"}],
        "required_conditions": [],
        "forbidden_claims": [],
        "clarifying_slots": [],
    }
    result = check_case_deterministic(case, "SH-E300 售价 299 元[1]。", {1: "SH-E300 售价 299 元"})
    assert "score" not in result
    assert "total_score" not in result
    assert result["failure_labels"] == []


def test_failure_labels_for_forbidden_and_under_refusal():
    case = {
        "expected_behavior": "refuse",
        "gold_facts": [],
        "required_conditions": [],
        "forbidden_claims": ["可以便宜"],
        "clarifying_slots": [],
    }
    result = check_case_deterministic(case, "可以便宜 10 元。", {})
    assert "forbidden_claim" in result["failure_labels"]
    assert "under_refusal" in result["failure_labels"]
