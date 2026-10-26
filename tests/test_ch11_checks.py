"""L3/L4 确定性检查:数值、条件、禁用断言、槽位、引用。"""
from __future__ import annotations

import pytest

from scripts.eval_checks import (
    check_case_deterministic,
    citation_support_rate,
    classify_behavior,
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
    assert bad.status == "unsupported"


def test_citation_after_terminal_punctuation_belongs_to_previous_claim():
    evidence = {1: "SH-K104 为 104 键机械键盘。"}
    outcome = citation_support_rate("SH-K104 是 104 键机械键盘。[1]", evidence)
    assert outcome.passed
    assert outcome.status == "supported"


def test_unmatched_paraphrase_is_uncertain_not_supported_by_character_overlap():
    evidence = {1: "商品支持在签收后发起无理由退货。"}
    outcome = citation_support_rate("该商品可以无理由退回。[1]", evidence)
    assert not outcome.passed
    assert outcome.status == "uncertain"
    assert "uncertain" in outcome.detail


def test_digits_inside_sku_are_not_treated_as_numeric_claims():
    evidence = {2: "质量问题退货，运费由商家承担；非质量问题由买家承担。"}
    outcome = citation_support_rate("SH-K104 质量问题退货运费由商家承担[2]。", evidence)
    assert outcome.status == "uncertain"
    assert "unsupported" not in outcome.detail


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


def test_scoring_overlay_can_exclude_background_fact_without_mutating_case():
    case = {
        "id": "Q1",
        "expected_behavior": "answer_with_conditions",
        "gold_facts": [
            {"fact_id": "PRICE", "value": 109, "unit": "元"},
            {"fact_id": "FEE", "value": 10, "unit": "元"},
        ],
        "required_conditions": ["部分地区", "商品原价"],
        "forbidden_claims": [],
        "clarifying_slots": [],
    }
    overlay = {
        "version": "rag-quality-v3",
        "cases": {
            "Q1": {
                "background_gold_fact_ids": ["PRICE"],
                "background_conditions": ["商品原价"],
                "basis": "问题只问加急费",
            }
        },
        "condition_aliases": {"配件类": ["数据线"]},
    }

    result = check_case_deterministic(
        case, "部分地区支持加急，加急费 10 元[1]。", {1: "部分地区支持加急，加急费 10 元。"},
        scoring_overlay=overlay,
    )

    assert result["gold_facts_checked"] == 1
    assert result["failure_labels"] == []
    assert result["scoring_overlay"]["background_gold_fact_ids"] == ["PRICE"]
    assert case["gold_facts"][0]["value"] == 109


def test_global_condition_alias_is_versioned_by_overlay():
    case = {
        "id": "Q2",
        "expected_behavior": "answer_with_conditions",
        "gold_facts": [{"fact_id": "W", "value": 6, "unit": "月"}],
        "required_conditions": ["配件类"],
        "forbidden_claims": [],
        "clarifying_slots": [],
    }
    overlay = {
        "version": "rag-quality-v3",
        "condition_aliases": {"配件类": ["数据线"]},
    }
    result = check_case_deterministic(
        case, "SH-C60 数据线质保半年[1]。", {1: "SH-C60 编织数据线半年质保。"},
        scoring_overlay=overlay,
    )
    assert "answer_missing_condition" not in result["failure_labels"]


def test_overlay_fact_alias_records_equivalence_for_background_status():
    case = {
        "id": "Q3",
        "expected_behavior": "answer",
        "gold_facts": [{"fact_id": "STATUS", "key": "status", "value": "退货中"}],
        "required_conditions": [],
        "forbidden_claims": [],
        "clarifying_slots": [],
    }
    overlay = {
        "version": "rag-quality-v3",
        "cases": {
            "Q3": {
                "fact_aliases": {"STATUS": ["仓库验货中", "退货申请已通过"]},
                "basis": "仓库验货是退货中状态的具体阶段。",
            }
        },
    }
    result = check_case_deterministic(
        case, "退货申请已通过，现为仓库验货中[1]。",
        {1: "退货申请已通过，仓库验货中。"}, scoring_overlay=overlay,
    )
    assert result["gold_fact_accuracy"]["passed"]


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


def test_bool_negation_does_not_match_positive():
    assert not number_covered("不支持主动降噪。", True)
    assert number_covered("不支持主动降噪。", False)


def test_forbidden_claim_with_negation_prefix_is_not_flagged():
    assert forbidden_claims_found("不是所有商品 95 折", ["所有商品 95 折"]) == []
    assert forbidden_claims_found("所有商品 95 折", ["所有商品 95 折"]) == ["所有商品 95 折"]


def test_bool_scopes_negation_to_the_predicate():
    answer = "SH-B7 支持血氧监测；活动商品标注不参与折扣。"
    assert number_covered(answer, True)
    assert not number_covered("SH-B7 不支持血氧监测。", True)


def test_date_condition_accepts_chinese_alias():
    outcome = condition_coverage(
        "6 月 1 日之后签收适用 v2", ["签收日期为 2026-06-01 或之后"]
    )
    assert outcome.passed


def test_duration_equivalences_and_window_calculation():
    assert number_covered("SH-C60 质保半年。", 6, "月")
    assert number_covered("整机一年质保。", 12, "月")
    assert condition_coverage(
        "签收第 12 天申请，仍在 15 天无理由窗口内。",
        ["15 天内"],
    ).passed
    assert not condition_coverage(
        "签收第 20 天申请，已超过 15 天无理由窗口。",
        ["15 天内"],
    ).passed


def test_iso_date_condition_is_compared_semantically():
    after = condition_coverage(
        "ORD-1004 于 2026-06-08 签收，适用 v2。",
        ["签收日期为 2026-06-01 或之后"],
    )
    before = condition_coverage(
        "ORD-1003 于 2026-05-20 签收，适用 v1。",
        ["签收日期早于 2026-06-01"],
    )
    assert after.passed and before.passed
    assert not condition_coverage(
        "ORD-1003 于 2026-05-20 签收。",
        ["签收日期为 2026-06-01 或之后"],
    ).passed
    assert not condition_coverage(
        "ORD-1003 于 2026-05-20 签收，早于 2026-06-01。",
        ["签收日期为 2026-06-01 或之后"],
    ).passed


def test_transfer_to_human_after_verification_is_not_refusal():
    answer = "若仍无法确认签收时间，需转人工复核 [1]。"
    assert classify_behavior(answer) != "refuse"


def test_marker_prefixed_slot_request_is_clarification():
    answer = "【无法回答】 请提供订单号，以便查询您的包裹预计发货时间。"
    assert classify_behavior(answer) == "clarify"


def test_natural_evidence_refusal_is_classified_as_refusal():
    answer = (
        "现有证据未提供 SH-W20 屏幕更换价格，无法给出报价。"
        "请转人工售后进一步核验。"
    )
    assert classify_behavior(answer) == "refuse"


@pytest.mark.parametrize(
    "answer",
    [
        "证据仅说明表带可更换，未提供单独购买价格，无法确定金额；建议转人工核实。",
        "现有证据仅列出血氧监测，未提及微信安装方式。该问题需转人工核实。",
        "无法确认 SH-E300 的充电功率；现有资料未包含该参数。",
    ],
)
def test_short_natural_refusal_prefixes_are_classified_as_refusal(answer):
    assert classify_behavior(answer) == "refuse"
