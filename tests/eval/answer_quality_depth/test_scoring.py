from __future__ import annotations

from scripts.answer_quality_depth.scoring import (
    apply_semantic_status,
    score_response,
    summarize_scores,
)


def _case(**updates):
    payload = {
        "id": "DEV-SINGLE-01",
        "split": "dev",
        "bucket": "single_hop",
        "query": "SH-E300 多少钱？",
        "answerable": True,
        "evidence_state": "sufficient",
        "expected_behavior": "answer",
        "gold_facts": [
            {
                "fact_id": "CATALOG-SH-E300-PRICE",
                "value": 299,
                "unit": "元",
            }
        ],
        "required_conditions": [],
        "condition_aliases": {},
        "forbidden_claims": [],
        "clarifying_slots": [],
        "slot_aliases": {},
        "verification_markers": [],
    }
    payload.update(updates)
    return payload


def _evidence(text: str = "SH-E300 售价 299 元。") -> list[dict]:
    return [{"chunk_id": 167, "text": text, "score": 0.9}]


def test_citation_after_period_belongs_to_the_previous_claim() -> None:
    result = score_response(
        _case(),
        "SH-E300 售价 299 元。[1]",
        _evidence(),
    )

    assert result["status"] == "correct"
    assert result["citation_status"] == "supported"
    assert result["facts_passed"] is True


def test_wrong_number_and_missing_unit_are_not_accepted() -> None:
    wrong = score_response(_case(), "SH-E300 售价 399 元。[1]", _evidence())
    no_unit = score_response(_case(), "SH-E300 售价 299。[1]", _evidence())

    assert wrong["status"] == "incorrect"
    assert wrong["facts_passed"] is False
    assert no_unit["status"] == "incorrect"


def test_half_year_and_six_months_are_equivalent() -> None:
    case = _case(
        gold_facts=[
            {
                "fact_id": "CATALOG-SH-C60-WARRANTY",
                "value": 0.5,
                "unit": "个月",
            }
        ]
    )
    result = score_response(
        case,
        "SH-C60 数据线的质保期为半年。[1]",
        _evidence("SH-C60 编织数据线，半年质保。"),
    )

    assert result["facts_passed"] is True


def test_background_gold_fact_is_not_required_when_overlay_says_so() -> None:
    case = _case(
        id="DEV-MULTI-02",
        gold_facts=[
            {
                "fact_id": "CATALOG-SH-E300-PRICE",
                "value": 299,
                "unit": "元",
            },
            {
                "fact_id": "PROMO-FREE-SHIPPING-THRESHOLD",
                "value": 99,
                "unit": "元",
            },
        ],
    )
    result = score_response(
        case,
        "订单商品金额 299 元已达到满 99 元包邮门槛。[1]",
        _evidence("单笔订单商品金额满 99 元包邮。"),
    )

    assert result["facts_checked"] == 1
    assert result["facts_passed"] is True


def test_refusal_and_clarification_are_separate_behavior_denominators() -> None:
    refusal = _case(
        id="DEV-HARD-01",
        bucket="hard_negative",
        answerable=False,
        evidence_state="insufficient",
        expected_behavior="refuse",
        gold_facts=[],
    )
    clarify = _case(
        id="DEV-CLARIFY-01",
        bucket="clarify",
        answerable=True,
        evidence_state="insufficient",
        expected_behavior="clarify",
        gold_facts=[],
        clarifying_slots=["sku"],
        slot_aliases={"sku": ["型号", "哪款"]},
    )

    refused = score_response(
        refusal,
        "现有资料没有说明 SH-E300 是否支持无线充电，无法确认。",
        _evidence(),
    )
    clarified = score_response(
        clarify,
        "请问您问的是哪款商品？",
        _evidence(),
    )

    assert refused["status"] == "correct"
    assert refused["denominator"] == "refusal"
    assert clarified["status"] == "correct"
    assert clarified["denominator"] == "clarify"


def test_behavior_v2_accepts_natural_clarification_and_refusal_wording() -> None:
    clarify = _case(
        id="DEV-CLARIFY-03",
        bucket="clarify",
        answerable=True,
        evidence_state="insufficient",
        expected_behavior="clarify",
        gold_facts=[],
        clarifying_slots=["sku"],
        slot_aliases={"sku": ["型号", "哪款", "商品"]},
    )
    clarify_answer = "一般整机保修一年。请先提供具体型号或订单号，我再确认对应期限。"
    assert score_response(clarify, clarify_answer, _evidence())["status"] == "correct"

    refusal = _case(
        id="DEV-HARD-01",
        bucket="hard_negative",
        answerable=False,
        evidence_state="insufficient",
        expected_behavior="refuse",
        gold_facts=[],
    )
    refusal_answer = "现有资料只写了蓝牙和续航，无法直接给出无线充电结论。"
    assert score_response(refusal, refusal_answer, _evidence())["status"] == "correct"

    refusal_with_explanation = (
        "现有资料没有列出礼包内容，因此无法可靠回答，不能直接猜测。"
    )
    assert (
        score_response(refusal, refusal_with_explanation, _evidence())["status"]
        == "correct"
    )


def test_uncertain_citation_is_retained_as_uncertain_not_correct() -> None:
    case = _case(
        gold_facts=[
            {
                "fact_id": "CATALOG-SH-E300-BATTERY",
                "value": 30,
                "unit": "小时",
            }
        ]
    )
    result = score_response(
        case,
        "SH-E300 的续航可以达到三十小时。[1]",
        _evidence("SH-E300 整机续航 30 小时。"),
    )

    assert result["status"] == "uncertain"
    assert result["citation_status"] == "uncertain"
    assert result["facts_passed"] is True


def test_markdown_between_number_and_condition_does_not_hide_a_correct_answer() -> None:
    case = _case(
        gold_facts=[{"fact_id": "WINDOW", "value": 15, "unit": "天"}],
        required_conditions=["15 天内"],
        expected_behavior="answer_with_conditions",
    )
    result = score_response(
        case,
        "签收后 **15 天**内可以无理由退货。[1]",
        _evidence("签收后 15 天内可以无理由退货。"),
    )

    assert result["facts_passed"] is True
    assert result["conditions_passed"] is True


def test_case_fact_override_is_applied_only_to_scoring() -> None:
    case = _case(
        id="CORRECTED",
        gold_facts=[{"fact_id": "WRONG_TARGET", "value": None}],
    )
    overlay = {
        "version": "test",
        "fact_overrides": {
            "CORRECTED": {
                "gold_facts": [{"fact_id": "HEART_RATE", "value": True, "unit": ""}]
            }
        },
        "cases": {},
    }
    result = score_response(
        case,
        "该型号支持心率监测。[1]",
        _evidence("该型号支持心率监测。"),
        overlay=overlay,
    )

    assert result["facts_passed"] is True
    assert case["gold_facts"][0]["fact_id"] == "WRONG_TARGET"


def test_case_semantic_override_marks_ambiguous_label_uncertain() -> None:
    case = _case(
        id="AMBIGUOUS",
        expected_behavior="clarify",
        gold_facts=[],
        clarifying_slots=["member_status"],
    )
    overlay = {
        "version": "test",
        "cases": {},
        "semantic_overrides": {
            "AMBIGUOUS": {
                "final_status": "uncertain",
                "basis": "Generic all-tier answer may fully answer the question.",
            }
        },
    }
    result = score_response(
        case,
        "所有会员等级都享 95 折，各等级积分倍率不同。",
        _evidence(),
        overlay=overlay,
    )
    result = apply_semantic_status(result, None)

    assert result["final_status"] == "uncertain"
    assert result["semantic_override"]["final_status"] == "uncertain"


def test_summary_does_not_merge_clarify_refusal_or_uncertain_into_answer_accuracy() -> (
    None
):
    rows = [
        {
            "split": "dev",
            "case_id": "A",
            "arm": "baseline",
            "status": "correct",
            "denominator": "evidence_answer",
            "bucket": "multi_hop",
            "technical_failure": False,
            "cost_usd": 0.1,
            "latency_ms": 10,
        },
        {
            "split": "dev",
            "case_id": "A",
            "arm": "slate",
            "status": "uncertain",
            "denominator": "evidence_answer",
            "bucket": "multi_hop",
            "technical_failure": False,
            "cost_usd": 0.2,
            "latency_ms": 20,
        },
        {
            "split": "dev",
            "case_id": "R",
            "arm": "baseline",
            "status": "correct",
            "denominator": "refusal",
            "bucket": "hard_negative",
            "technical_failure": False,
            "cost_usd": 0.1,
            "latency_ms": 10,
        },
        {
            "split": "dev",
            "case_id": "R",
            "arm": "slate",
            "status": "correct",
            "denominator": "refusal",
            "bucket": "hard_negative",
            "technical_failure": False,
            "cost_usd": 0.2,
            "latency_ms": 20,
        },
    ]
    summary = summarize_scores(rows)
    evidence = summary["splits"]["dev"]["by_denominator"]["evidence_answer"]

    assert evidence["slate"]["correct"] == 0
    assert evidence["slate"]["uncertain"] == 1
    assert evidence["slate"]["strict_accuracy"] == 0.0
    assert evidence["slate"]["uncertainty_rate"] == 1.0
    assert (
        summary["splits"]["dev"]["by_denominator"]["refusal"]["slate"]["correct"] == 1
    )
    assert summary["splits"]["dev"]["pairwise"]["slate_vs_baseline"] == {
        "wins": 0,
        "losses": 1,
        "ties": 1,
    }


def test_semantic_judge_resolves_only_citation_uncertainty_not_hard_fact_errors() -> (
    None
):
    clean = score_response(_case(), "SH-E300 售价 299 元 [1]。", _evidence())
    clean["citation_status"] = "uncertain"
    clean["status"] = "uncertain"
    judged = apply_semantic_status(
        clean,
        {
            "disposition": "answer",
            "answer_status": "correct",
            "grounding": "supported",
            "citation_support": "supported",
            "completeness": 100,
            "unsupported_claims": [],
            "missing_requirements": [],
            "rationale": "证据直接支持。",
        },
    )
    assert judged["final_status"] == "correct"
    assert judged["grounded_correct"] is True

    wrong = score_response(_case(), "SH-E300 售价 399 元 [1]。", _evidence())
    prevented = apply_semantic_status(
        wrong,
        {
            "disposition": "answer",
            "answer_status": "correct",
            "grounding": "supported",
            "citation_support": "supported",
            "completeness": 100,
            "unsupported_claims": [],
            "missing_requirements": [],
            "rationale": "judge误判",
        },
    )
    assert prevented["final_status"] == "incorrect"

    optional_followup = apply_semantic_status(
        {
            **clean,
            "citation_status": "uncertain",
            "status": "uncertain",
        },
        {
            "disposition": "clarify",
            "answer_status": "correct",
            "grounding": "supported",
            "citation_support": "supported",
            "completeness": 100,
            "unsupported_claims": [],
            "missing_requirements": [],
            "rationale": "答案完整，末尾只是可选的下一步追问。",
        },
    )
    assert optional_followup["final_status"] == "correct"

    incomplete = apply_semantic_status(
        {
            **clean,
            "citation_status": "uncertain",
            "status": "uncertain",
        },
        {
            "disposition": "answer",
            "answer_status": "correct",
            "grounding": "supported",
            "citation_support": "supported",
            "completeness": 70,
            "unsupported_claims": [],
            "missing_requirements": ["签收日期"],
            "rationale": "缺少关键条件。",
        },
    )
    assert incomplete["final_status"] == "uncertain"

    partially_grounded = apply_semantic_status(
        {
            **clean,
            "citation_status": "supported",
            "status": "correct",
        },
        {
            "disposition": "answer",
            "answer_status": "partial",
            "grounding": "partly_supported",
            "citation_support": "partial",
            "completeness": 70,
            "unsupported_claims": [],
            "missing_requirements": ["适用条件"],
            "rationale": "只回答了一半。",
        },
    )
    assert partially_grounded["final_status"] == "incorrect"


def test_complete_answer_with_optional_followup_is_not_overclassified_as_clarify() -> (
    None
):
    case = _case(
        gold_facts=[
            {"fact_id": "STATUS", "value": "已发货", "unit": ""},
            {"fact_id": "CANCEL", "value": "不能直接取消", "unit": ""},
        ]
    )
    answer = (
        "ORD-1003 已发货 [1]，不能直接取消 [1][2]，可以选择拒收，"
        "或者在签收后申请退货 [2][3]。请问您更希望按哪种方式处理？"
    )
    result = score_response(
        case,
        answer,
        _evidence("ORD-1003 已发货；已发货不能直接取消，可拒收或签收后走退货。"),
    )

    assert result["behavior_observed"] == "clarify"
    assert result["behavior_passed"] is True
