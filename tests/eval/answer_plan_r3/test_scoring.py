from __future__ import annotations

from scripts.answer_plan_r3.scoring import score_case


def _case(**updates):
    payload = {
        "id": "D1",
        "expected_behavior": "answer",
        "facts": [{
            "fact_id": "F1",
            "entity": "SH-K68",
            "predicate": "售价",
            "value": 149,
            "unit": "元",
            "require_entity": True,
        }],
        "conditions": [],
        "condition_aliases": {},
        "forbidden_claims": [],
        "requires_citation": True,
        "clarifying_slots": [],
        "evidence_ids": ["EV-K68"],
    }
    payload.update(updates)
    return payload


def _evidence(**updates):
    row = {
        "evidence_id": "EV-K68",
        "text": "SH-K68 售价 149 元。",
        "source_id": "catalog#k68",
        "source_title": "商品目录",
        "entities": ["SH-K68"],
    }
    row.update(updates)
    return row


def _outcome(answer, items=None):
    return {
        "answer": answer,
        "citation_items": items or {1: "EV-K68"},
    }


def test_strict_success_requires_fact_behavior_conditions_and_supported_citation():
    result = score_case(_case(), _outcome("SH-K68 售价 149 元 [1]。"), [_evidence()])

    assert result["facts"]["passed"] is True
    assert result["citations"]["status"] == "supported"
    assert result["strict_success"] is True


def test_wrong_entity_is_unsupported_not_uncertain():
    result = score_case(
        _case(), _outcome("SH-K104 售价 149 元 [1]。"), [_evidence()]
    )

    assert result["facts"]["passed"] is False
    assert result["citations"]["status"] == "unsupported"
    assert result["strict_success"] is False


def test_citation_without_direct_support_is_uncertain_and_never_strict_success():
    result = score_case(
        _case(),
        _outcome("SH-K68 很可能就是 149 元 [1]。"),
        [_evidence(text="SH-K68 是一款机械键盘。", entities=["SH-K68"])],
    )

    assert result["citations"]["status"] == "uncertain"
    assert result["strict_success"] is False


def test_missing_citation_is_unsupported_for_knowledge_answer():
    result = score_case(_case(), _outcome("SH-K68 售价 149 元。", {}), [_evidence()])

    assert result["citations"]["status"] == "unsupported"
    assert result["strict_success"] is False


def test_persisted_json_string_citation_keys_are_normalized():
    result = score_case(
        _case(),
        {
            "answer": "SH-K68 售价 149 元。 [1]",
            "citation_items": {"1": "EV-K68"},
        },
        [_evidence()],
    )

    assert result["citations"]["status"] == "supported"
    assert result["strict_success"] is True


def test_condition_alias_is_accepted_and_omission_fails():
    case = _case(
        facts=[],
        conditions=["签收当日"],
        condition_aliases={"签收当日": ["当天"]},
    )
    evidence = _evidence(
        text="生鲜仅限签收当日发起的质量问题退货。",
        entities=[],
    )

    accepted = score_case(
        case, _outcome("生鲜有质量问题时仅限当天申请 [1]。"), [evidence]
    )
    omitted = score_case(
        case, _outcome("生鲜有质量问题可以退货 [1]。"), [evidence]
    )

    assert accepted["conditions"]["passed"] is True
    assert accepted["strict_success"] is True
    assert omitted["conditions"]["passed"] is False


def test_clarify_and_refuse_are_scored_as_behaviors_not_answers():
    clarify = score_case(
        _case(
            expected_behavior="clarify",
            facts=[],
            requires_citation=False,
            evidence_ids=[],
            clarifying_slots=["sku"],
        ),
        _outcome("请提供具体商品型号或 SKU。", {}),
        [],
    )
    refuse = score_case(
        _case(
            expected_behavior="refuse",
            facts=[],
            requires_citation=False,
            evidence_ids=[],
        ),
        _outcome("抱歉，现有资料无法确认礼包内具体商品。", {}),
        [],
    )

    assert clarify["behavior"]["passed"] is True
    assert clarify["strict_success"] is True
    assert refuse["behavior"]["passed"] is True
    assert refuse["strict_success"] is True


def test_cross_entity_claims_must_bind_each_amount_to_its_own_order():
    case = _case(
        facts=[
            {"fact_id": "F4", "entity": "ORD-1004", "predicate": "商品金额",
             "value": 399, "unit": "元", "require_entity": True},
            {"fact_id": "F8", "entity": "ORD-1008", "predicate": "商品金额",
             "value": 129, "unit": "元", "require_entity": True},
        ],
        evidence_ids=["EV-1004", "EV-1008"],
    )
    evidence = [
        _evidence(
            evidence_id="EV-1004",
            text="ORD-1004 商品金额 399 元。",
            entities=["ORD-1004"],
        ),
        _evidence(
            evidence_id="EV-1008",
            text="ORD-1008 商品金额 129 元。",
            entities=["ORD-1008"],
        ),
    ]
    result = score_case(
        case,
        _outcome(
            "ORD-1004 商品金额为 399 元 [1]。ORD-1008 商品金额为 129 元 [2]。",
            {1: "EV-1004", 2: "EV-1008"},
        ),
        evidence,
    )

    assert result["facts"]["passed"] is True
    assert result["citations"]["status"] == "supported"
    assert result["strict_success"] is True


def test_citation_attaches_across_semicolon_and_after_terminal_punctuation():
    case = _case(
        facts=[{
            "fact_id": "F1",
            "predicate": "退货窗口",
            "value": 7,
            "unit": "天",
            "require_entity": False,
        }],
    )
    evidence = [
        _evidence(
            text="签收日期早于 2026-06-01 的订单适用 v1；7 天内支持无理由退货。",
            entities=[],
        )
    ]
    result = score_case(
        case,
        _outcome(
            "签收日期早于 2026-06-01 适用 v1；7 天内支持无理由退货。[1]",
            {1: "EV-K68"},
        ),
        evidence,
    )

    assert result["citations"]["status"] == "supported"
