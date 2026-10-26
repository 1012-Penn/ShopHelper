from __future__ import annotations

from scripts.answer_plan_r3.scoring_v2 import score_case_v2


def _case(**updates):
    payload = {
        "id": "H1",
        "expected_behavior": "answer_with_conditions",
        "facts": [],
        "conditions": [],
        "condition_aliases": {},
        "forbidden_claims": [],
        "requires_citation": True,
        "clarifying_slots": [],
    }
    payload.update(updates)
    return payload


def test_v2_checks_all_predicate_occurrences_for_negation():
    case = _case(
        facts=[{
            "fact_id": "F1",
            "predicate": "95折",
            "value": False,
            "require_entity": False,
        }],
    )
    evidence = [{
        "evidence_id": "EV1",
        "text": "在 95 折活动中，清仓商品不享 95 折。",
        "source_id": "policy",
        "entities": [],
    }]
    result = score_case_v2(
        case,
        {
            "answer": "在 95 折活动中，清仓商品不享 95 折 [1]。",
            "citation_items": {1: "EV1"},
        },
        evidence,
    )

    assert result["facts"]["passed"] is True
    assert result["strict_success"] is True


def test_v2_does_not_cover_unmentioned_fact_by_shared_entity_alone():
    case = _case(
        expected_behavior="answer",
        facts=[
            {
                "fact_id": "WATER",
                "entity": "SH-S10",
                "predicate": "防水",
                "value": "IPX7",
                "require_entity": True,
            },
            {
                "fact_id": "BATTERY",
                "entity": "SH-S10",
                "predicate": "续航",
                "value": 12,
                "unit": "小时",
                "require_entity": True,
            },
        ],
    )
    evidence = [{
        "evidence_id": "EV1",
        "text": "SH-S10 支持 IPX7 防水，续航 12 小时。",
        "source_id": "catalog",
        "entities": ["SH-S10"],
    }]
    result = score_case_v2(
        case,
        {
            "answer": "SH-S10 的续航为 12 小时 [1]。",
            "citation_items": {1: "EV1"},
        },
        evidence,
    )

    assert result["citations"]["status"] == "uncertain"
    assert result["citations"]["covered_fact_ids"] == ["BATTERY"]


def test_v2_links_answer_and_evidence_condition_aliases():
    case = _case(
        expected_behavior="answer",
        facts=[{
            "fact_id": "DISCOUNT",
            "predicate": "折扣",
            "value": "95折",
            "accepted_phrases": ["95 折"],
            "require_entity": False,
        }],
        conditions=["正价在售商品"],
        condition_aliases={"正价在售商品": ["在售的正价商品"]},
    )
    evidence = [{
        "evidence_id": "EV1",
        "text": "所有会员均享 95 折；仅适用于在售的正价商品。",
        "source_id": "membership",
        "entities": [],
    }]
    result = score_case_v2(
        case,
        {
            "answer": "所有会员购买正价在售商品均享 95 折 [1]。",
            "citation_items": {1: "EV1"},
        },
        evidence,
    )

    assert result["conditions"]["passed"] is True
    assert result["citations"]["status"] == "supported"
    assert result["strict_success"] is True
