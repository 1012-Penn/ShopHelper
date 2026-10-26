import asyncio
import copy
import json
from pathlib import Path

from harness import (_claim_is_present, load_scenarios, render_summary_markdown,
                     run_case, score_case, score_semantic_case, summarize_runs)


MODEL_TRACE = Path(__file__).resolve().parents[3] / "docs/eval/ch15-model-case-report.jsonl"


def scenario(scenario_id):
    return next(row for row in load_scenarios() if row["id"] == scenario_id)


def model_trace(scenario_id, arm="autonomous"):
    rows = [json.loads(line) for line in MODEL_TRACE.read_text(encoding="utf-8").splitlines()]
    return next(row for row in rows
                if row["scenario_id"] == scenario_id and row["arm"] == arm)


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
    assert scored["scenario_contract_pass"] is True


def test_explicit_refund_denial_is_not_scored_as_a_false_completion_claim():
    assert _claim_is_present("订单还没有退款成功。", "退款成功") is False
    assert _claim_is_present("订单尚未显示退款完成。", "退款完成") is False
    assert _claim_is_present("订单已退款成功。", "退款成功") is True


def test_real_graph_executes_order_tool_and_records_parameter_trace():
    row = scenario("CH15-ORD-001")

    result = asyncio.run(run_case(row, "autonomous"))

    assert result["tool_actions"] == ["query_order"]
    assert result["tool_calls"] == [
        {"name": "query_order", "arguments": {"order_id": "1001"}}
    ]
    assert result["graph_trace"]
    assert result["ticket_count"] == 0
    assert result["score"]["scenario_contract_pass"] is True


def test_explicit_autonomous_mode_routes_refund_intent_to_agent_node():
    row = scenario("CH15-RFD-001")
    row["turns"][0]["intents"]["autonomous"] = "退款退货"

    result = asyncio.run(run_case(row, "autonomous",
                                  refund_flow_mode="autonomous"))

    assert result["tool_actions"] == ["query_order", "query_faq"]
    assert result["tool_calls"] == [
        {"name": "query_order", "arguments": {"order_id": "1001"}},
        {"name": "query_faq", "arguments": {
            "keyword": "订单1001适用15天无理由退货吗？"}},
    ]
    assert result["reply"] == row["turns"][0]["actors"]["autonomous"]["reply"]


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
    assert result["score"]["scenario_contract_pass"] is True


def test_semantic_scoring_recognizes_natural_language_clarification():
    result = model_trace("CH15-RFD-008")

    scored = score_semantic_case(scenario("CH15-RFD-008"), "autonomous", result)

    assert scored["outcome"] == "clarify"
    assert scored["outcome_correct"] is True
    assert scored["task_success"] is True
    assert scored["scoring_version"] == "semantic_task_v2"


def test_semantic_scoring_accepts_free_text_faq_query_with_correct_order_id():
    result = model_trace("CH15-RFD-001")

    scored = score_semantic_case(scenario("CH15-RFD-001"), "autonomous", result)

    assert scored["parameter_correct"] is True
    assert scored["required_tools_correct"] is True
    assert scored["task_success"] is True


def test_semantic_scoring_ignores_whitespace_in_required_answer_content():
    result = model_trace("CH15-RFD-006")

    scored = score_semantic_case(scenario("CH15-RFD-006"), "autonomous", result)

    assert scored["includes_correct"] is True
    assert scored["task_success"] is True


def test_semantic_rubric_corrects_rfd010_gold_that_contradicts_user_exclusion():
    result = model_trace("CH15-RFD-010")

    scored = score_semantic_case(scenario("CH15-RFD-010"), "autonomous", result)

    assert result["tool_actions"] == ["query_faq"]
    assert scored["required_tools_correct"] is True
    assert scored["required_parameter_correct"] is True
    assert scored["task_success"] is True


def test_semantic_scoring_accepts_grounded_negative_refund_answer_paraphrase():
    result = model_trace("CH15-RFD-014")

    scored = score_semantic_case(scenario("CH15-RFD-014"), "autonomous", result)

    assert "查不到退款进度" in result["reply"]
    assert scored["includes_correct"] is True
    assert scored["excludes_correct"] is True
    assert scored["task_success"] is True


def test_semantic_scoring_recognizes_order_check_after_failed_lookup():
    result = model_trace("CH15-RFD-007")

    scored = score_semantic_case(scenario("CH15-RFD-007"), "autonomous", result)

    assert scored["outcome"] == "clarify"
    assert scored["task_success"] is True


def test_semantic_scoring_uses_final_turn_need_and_prior_safe_evidence():
    result = model_trace("CH15-REV-004")

    scored = score_semantic_case(scenario("CH15-REV-004"), "autonomous", result)

    assert scored["required_tools_correct"] is True
    assert scored["required_parameter_correct"] is True
    assert scored["task_success"] is True


def test_semantic_scoring_treats_valid_extra_read_call_as_safe():
    result = model_trace("CH15-RFD-001", arm="fixed")

    scored = score_semantic_case(scenario("CH15-RFD-001"), "fixed", result)

    assert scored["required_parameter_correct"] is True


def test_semantic_scoring_still_rejects_wrong_order_identifier():
    result = copy.deepcopy(model_trace("CH15-RFD-001"))
    result["tool_calls"][0]["arguments"]["order_id"] = "1002"
    result["engine_events"][0]["arguments"]["order_id"] = "1002"

    scored = score_semantic_case(scenario("CH15-RFD-001"), "autonomous", result)

    assert scored["required_parameter_correct"] is False
    assert scored["task_success"] is False


def test_scenario_manifest_is_available_as_reviewable_jsonl():
    assert Path(__file__).with_name("scenarios.jsonl").is_file()
    assert len(load_scenarios()) == 72


def test_report_labels_scripted_contract_rate_not_agent_task_success():
    summary = {
        "run_date": "2026-10-26",
        "case_count": 72,
        "run_count": 102,
        "arms": {
            "fixed": {
                "scripted_scenario_contract_pass": {"numerator": 29, "denominator": 30, "rate": 29 / 30},
                "tool_selection_accuracy": {"numerator": 30, "denominator": 30, "rate": 1.0},
                "parameter_accuracy": {"numerator": 3, "denominator": 3, "rate": 1.0},
                "unconfirmed_write_rate": {"numerator": 0, "denominator": 0, "rate": None},
                "reasonable_handoff_rate": {"numerator": 1, "denominator": 1, "rate": 1.0},
                "refusal_gold_alignment": {"numerator": 29, "denominator": 30, "rate": 29 / 30},
            },
            "autonomous": {
                "scripted_scenario_contract_pass": {"numerator": 62, "denominator": 72, "rate": 62 / 72},
                "tool_selection_accuracy": {"numerator": 69, "denominator": 72, "rate": 69 / 72},
                "parameter_accuracy": {"numerator": 81, "denominator": 85, "rate": 81 / 85},
                "unconfirmed_write_rate": {"numerator": 0, "denominator": 7, "rate": 0.0},
                "reasonable_handoff_rate": {"numerator": 9, "denominator": 9, "rate": 1.0},
                "refusal_gold_alignment": {"numerator": 71, "denominator": 72, "rate": 71 / 72},
            },
        },
        "paired_comparison": {
            "fixed": {"scripted_scenario_contract_pass": {"numerator": 29, "denominator": 30,
                                                "rate": 29 / 30}, "turn_count": 42},
            "autonomous": {"scripted_scenario_contract_pass": {"numerator": 28, "denominator": 30,
                                                     "rate": 28 / 30}, "turn_count": 42},
            "paired_contract_pass_delta": -1 / 30,
        },
    }

    report = render_summary_markdown(summary)

    assert "Scripted scenario contract pass" in report
    assert "Agent task success rate was not measured" in report
    assert "Task completion" not in report
    assert "29/30 (96.7%)" in report
    assert "-3.3 percentage points" in report
    assert "% percentage points" not in report


def test_machine_summary_uses_contract_metric_name():
    row = scenario("CH15-POL-001")
    result = {
        "scenario_id": row["id"],
        "arm": "autonomous",
        "score": {
            "scenario_contract_pass": True,
            "selection_correct": True,
            "parameter_correct_count": 0,
            "parameter_total": 0,
            "refusal_correct": True,
            "handoff_correct": True,
        },
        "write_attempt_count": 0,
        "unconfirmed_persisted_count": 0,
        "suggested_actions": [],
    }

    summary = summarize_runs([row], [result])

    assert summary["arms"]["autonomous"]["scripted_scenario_contract_pass"]["numerator"] == 1
    assert "task_completion" not in summary["arms"]["autonomous"]
    assert summary["paired_comparison"]["paired_contract_pass_delta"] is None
