from __future__ import annotations

import pytest

from scripts.e2e_quality_r2.scoring import (
    InputBoundaryError,
    assert_model_input_safe,
    classify_observed_behavior,
    resolve_citation_evidence,
    score_case,
)


def _case(**updates):
    payload = {
        "id": "D1",
        "split": "dev",
        "expected_behavior": "answer",
        "facts": [
            {
                "fact_id": "F-PRICE",
                "entity": "SH-E300",
                "predicate": "售价",
                "value": 299,
                "unit": "元",
                "require_entity": True,
            }
        ],
        "conditions": [],
        "forbidden_claims": [],
        "required_evidence_terms": [["SH-E300", "299"]],
        "clarifying_slots": [],
    }
    payload.update(updates)
    return payload


def _outcome(answer: str, evidence_text: str = "SH-E300 售价 299 元。"):
    return {
        "answer": answer,
        "behavior": "answer",
        "evidence": {
            1: {
                "n": 1,
                "chunk_id": 167,
                "category": "商品目录",
                "section_path": "商品目录.md > 耳机",
                "text": evidence_text,
            }
        },
        "tool_calls": [],
        "technical_failures": [],
        "route_nodes": ["intent", "retrieve", "gate", "agent"],
        "cost_usd": 0.001,
        "latency_ms": 100.0,
    }


@pytest.mark.parametrize(
    "projection",
    [
        {"expected_behavior": "answer"},
        {"payload": {"gold_facts": [{"fact_id": "F-PRICE"}]}},
        {"required_conditions": ["商品完好"]},
        {"scoring": {"threshold": 1}},
        {"answer_key": "SH-E300 售价 299 元"},
    ],
)
def test_model_projection_rejects_hidden_label_fields(projection):
    with pytest.raises(InputBoundaryError):
        assert_model_input_safe(projection, _case())


def test_model_projection_allows_business_evidence_and_user_turns():
    assert_model_input_safe(
        {
            "messages": [
                {"role": "user", "content": "SH-E300 多少钱？"},
                {"role": "tool", "content": "SH-E300 售价 299 元。"},
            ],
            "tools": [{"name": "query_order", "schema": {"order_id": "string"}}],
        },
        _case(),
    )


def test_citation_mapping_offsets_agent_tool_evidence():
    state = {"evidence": [{"n": 1, "chunk_id": 10, "answer": "政策一"}]}
    tools = [
        {
            "name": "query_faq",
            "result": {
                "items": [
                    {"n": 1, "chunk_id": 20, "answer": "政策二"},
                    {"n": 2, "chunk_id": 21, "answer": "政策三"},
                ]
            },
        }
    ]

    mapped = resolve_citation_evidence(state, tools)

    assert mapped[1]["chunk_id"] == 10
    assert mapped[2]["chunk_id"] == 20
    assert mapped[3]["chunk_id"] == 21


def test_scorer_requires_entity_number_and_unit():
    result = score_case(_case(), _outcome("SH-E300 售价 299 元 [1]。"))
    assert result["facts"]["passed"] is True
    assert result["task_success"] is True

    wrong_entity = score_case(_case(), _outcome("SH-E120 售价 299 元 [1]。"))
    assert wrong_entity["entities"]["passed"] is False
    assert wrong_entity["task_success"] is False

    missing_unit = score_case(
        _case(), _outcome("SH-E300 售价 299 [1]。", "SH-E300 售价 299 元。")
    )
    assert missing_unit["facts"]["passed"] is False


def test_uncertain_citation_is_review_not_success_or_hallucination():
    case = _case(
        facts=[],
        required_evidence_terms=[["主动降噪"]],
        expected_behavior="answer",
    )
    outcome = _outcome(
        "SH-E300 支持主动降噪 [1]。",
        "SH-E300 具备主动降噪功能。",
    )
    result = score_case(case, outcome)

    assert result["citations"]["status"] == "uncertain"
    assert result["review_required"] is True
    assert result["task_success"] is False
    assert "hallucination" not in result["failure_labels"]


def test_missing_and_invalid_citations_fail_without_claiming_hallucination():
    case = _case()
    missing = score_case(case, _outcome("SH-E300 售价 299 元。"))
    invalid = score_case(case, _outcome("SH-E300 售价 299 元 [9]。"))

    assert missing["citations"]["passed"] is False
    assert invalid["citations"]["invalid_labels"] == [9]
    assert invalid["failure_labels"] == ["citation_invalid"]


def test_required_condition_and_tool_completion_are_layers():
    case = _case(
        facts=[],
        conditions=["商品完好"],
        required_evidence_terms=[["商品完好"]],
        expected_tools=[{"name": "query_order", "args_contains": {"order_id": "1002"}}],
    )
    outcome = _outcome("商品完好时可退 [1]。", "商品完好时可退。")
    outcome["tool_calls"] = [
        {
            "name": "query_order",
            "args": {"order_id": "1002"},
            "is_error": False,
        }
    ]
    result = score_case(case, outcome)

    assert result["conditions"]["passed"] is True
    assert result["tools"]["passed"] is True
    assert result["task_success"] is True


def test_technical_failure_is_not_scored_as_model_wrong():
    outcome = _outcome("SH-E300 售价 299 元 [1]。")
    outcome["technical_failures"] = ["429 after bounded retries"]
    result = score_case(_case(), outcome)

    assert result["technical_failure"] is True
    assert result["task_success"] is False
    assert result["failure_labels"] == ["technical_failure"]


def test_complete_answer_with_optional_followup_is_not_misclassified_as_clarify():
    answer = "SH-E300 售价 299 元 [1]。需要我帮您再介绍其他型号吗？"
    assert classify_observed_behavior(answer) == "answer"
    assert classify_observed_behavior("请问您具体想看哪个型号？") == "clarify"


def test_tool_only_mode_does_not_emit_citation_missing_label():
    case = _case(
        facts=[],
        required_evidence_terms=[],
        requires_citation=False,
        expected_tools=[{"name": "query_order", "args_contains": {"order_id": "1002"}}],
    )
    outcome = _outcome("订单 1002 金额 399 元。", "")
    outcome["tool_calls"] = [
        {"name": "query_order", "args": {"order_id": "1002"}, "is_error": False}
    ]

    result = score_case(case, outcome)

    assert result["citations"]["status"] == "tool_only"
    assert "citation_missing" not in result["failure_labels"]
    assert result["task_success"] is True
