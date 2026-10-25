import asyncio
from pathlib import Path

from harness import load_scenarios, run_case, score_case


def scenario(scenario_id):
    return next(row for row in load_scenarios() if row["id"] == scenario_id)


def test_refusal_is_scored_against_case_gold_not_as_a_blanket_success():
    row = scenario("CH15-HOF-001")
    trace = {
        "reply": "【无法回答】我不能提供诊断或用药建议。",
        "tool_actions": [],
        "tool_calls": [],
        "suggested_actions": ["transfer_human", "create_ticket"],
        "ticket_count": 0,
        "audit_statuses": [],
        "forbidden_actions": [],
    }

    scored = score_case(row, "autonomous", trace)

    assert scored["refusal_correct"] is True
    assert scored["handoff_correct"] is True
    assert scored["task_complete"] is True


def test_real_graph_executes_order_tool_and_records_parameter_trace():
    row = scenario("CH15-ORD-001")

    result = asyncio.run(run_case(row, "autonomous"))

    assert result["tool_actions"] == ["query_order"]
    assert result["tool_calls"] == [
        {"name": "query_order", "arguments": {"order_id": "1001"}}
    ]
    assert result["graph_trace"]
    assert result["ticket_count"] == 0
    assert result["score"]["task_complete"] is True


def test_duplicate_confirmation_uses_real_endpoint_and_is_idempotent():
    row = scenario("CH15-FLT-001")

    result = asyncio.run(run_case(row, "autonomous"))

    assert result["ticket_count"] == 1
    assert len(result["fault_trace"]["confirmation_requests"]) == 2
    assert result["fault_trace"]["confirmation_responses"][1]["body"]["deduplicated"] is True
    assert result["score"]["database_correct"] is True


def test_successful_write_with_lost_response_replays_same_confirmation_key():
    row = scenario("CH15-FLT-002")

    result = asyncio.run(run_case(row, "autonomous"))

    assert result["ticket_count"] == 1
    assert result["fault_trace"]["response_dropped_after_commit"] is True
    assert result["fault_trace"]["confirmation_responses"][0]["client_visible"] is False
    assert result["fault_trace"]["confirmation_responses"][1]["body"]["deduplicated"] is True


def test_unconfirmed_write_is_blocked_and_tool_timeout_is_audited():
    blocked = asyncio.run(run_case(scenario("CH15-FLT-004"), "autonomous"))
    timed_out = asyncio.run(run_case(scenario("CH15-FLT-003"), "autonomous"))

    assert blocked["ticket_count"] == 0
    assert "权限拒绝" in blocked["audit_statuses"]
    assert timed_out["ticket_count"] == 0
    assert "超时" in timed_out["audit_statuses"]
    assert timed_out["audit_rows"][0]["retry_count"] == 1


def test_multi_turn_result_uses_last_turn_not_a_prior_order_selector():
    row = scenario("CH15-REV-009")

    result = asyncio.run(run_case(row, "fixed"))

    assert result["turn_frames"][0][0]["type"] == "order_selector"
    assert result["score"]["outcome"] == "answer"
    assert result["score"]["task_complete"] is True


def test_scenario_manifest_is_available_as_reviewable_jsonl():
    assert Path(__file__).with_name("scenarios.jsonl").is_file()
    assert len(load_scenarios()) == 72
