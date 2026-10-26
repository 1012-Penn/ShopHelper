import json
from pathlib import Path

import pytest
from langchain.tools import tool
from langchain_core.messages import AIMessage

from scripts.eval_multi_agent import (
    _AccurateGuardedChatModel,
    _read_jsonl,
    _write_jsonl,
    ConcurrentPaidRunBudget,
    load_scenarios,
    model_visible_turn,
    scenario_manifest_sha256,
    score_case,
)
from scripts.ch15_paid_guard import ArmLimit, paid_call_context
from scripts.eval_cost import PricingTable


SCENARIO_PATH = Path("tests/eval/multi_agent/scenarios.jsonl")


def test_scenario_set_is_frozen_small_and_gold_is_not_in_model_projection():
    scenarios = load_scenarios(SCENARIO_PATH)
    assert len(scenarios) == 30
    assert sum(row["split"] == "dev" for row in scenarios) == 10
    assert sum(row["split"] == "frozen" for row in scenarios) == 20
    assert len({row["id"] for row in scenarios}) == 30
    assert len(scenario_manifest_sha256(SCENARIO_PATH)) == 64

    row = next(item for item in scenarios if item["id"] == "MA-DEV-003")
    projection = model_visible_turn(row, 0)
    assert projection == {"message": "订单1001现在是什么状态？"}
    assert "must_include" not in json.dumps(projection)


def test_scoring_accepts_complete_observable_result_without_using_gold_in_projection():
    scenario = next(row for row in load_scenarios(SCENARIO_PATH)
                    if row["id"] == "MA-DEV-005")
    result = {
        "reply": "订单1001已签收；订单1002已发货。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1001"}},
            {"name": "query_order", "arguments": {"order_id": "1002"}},
        ],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["task_success"] is True
    assert score["entity_error_count"] == 0
    assert score["parameter_error_count"] == 0
    assert score["condition_error_count"] == 0


def test_scoring_keeps_wrong_entity_and_missing_required_condition_as_failures():
    scenario = next(row for row in load_scenarios(SCENARIO_PATH)
                    if row["id"] == "MA-DEV-003")
    result = {
        "reply": "订单1002已签收。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1002"}},
        ],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["task_success"] is False
    assert score["entity_error_count"] >= 2
    assert score["condition_error_count"] >= 1


def test_write_scoring_requires_preview_and_exactly_one_confirmed_write():
    scenario = next(row for row in load_scenarios(SCENARIO_PATH)
                    if row["id"] == "MA-DEV-010")
    good = {
        "reply": "订单1001的工单预览已生成，请确认；当前没有创建工单。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1001"}},
            {"name": "create_ticket", "arguments": {
                "conversation_id": 1, "description": "订单1001耳机售后",
                "ticket_type": "售后",
            }},
        ],
        "write_previews": [{"tool_name": "create_ticket"}],
        "confirmed_write_count": 1,
        "unconfirmed_write_count": 0,
    }
    bad = {**good, "confirmed_write_count": 2}
    assert score_case(scenario, good)["task_success"] is True
    assert score_case(scenario, bad)["task_success"] is False


def test_optional_offer_to_ask_for_order_does_not_turn_factual_answer_into_clarify():
    scenario = next(row for row in load_scenarios(SCENARIO_PATH)
                    if row["id"] == "MA-DEV-001")
    result = {
        "reply": "签收后15天内可无理由退货，需保留吊牌。如需要，可提供订单号再核对。",
        "tool_calls": [{"name": "query_faq", "arguments": {"keyword": "退货窗口"}}],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    assert score_case(scenario, result)["actual_outcome"] == "answer"


def test_dates_and_evidence_chunk_ids_are_not_counted_as_order_entities():
    scenario = next(row for row in load_scenarios(SCENARIO_PATH)
                    if row["id"] == "MA-DEV-003")
    result = {
        "reply": "订单1001已签收，下单日期2026-08-30，依据证据15012。",
        "tool_calls": [{"name": "query_order", "arguments": {"order_id": "1001"}}],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["entity_error_count"] == 0
    assert score["task_success"] is True


def test_semantic_scoring_overlay_accepts_equivalent_safe_wording():
    by_id = {row["id"]: row for row in load_scenarios(SCENARIO_PATH)}
    deny = {
        "reply": "订单1002已发货，因此无法直接取消，可拒收或签收后退货。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1002"}},
            {"name": "query_faq", "arguments": {"keyword": "取消政策"}},
        ],
        "write_previews": [], "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    qualified = {
        "reply": "非质量问题由买家承担；作为对比，质量问题由商家承担。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1002"}},
            {"name": "query_faq", "arguments": {"keyword": "运费"}},
        ],
        "write_previews": [], "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    cancelled = {
        "reply": "好的，不办理退货，也不创建工单。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1001"}},
            {"name": "query_faq", "arguments": {"keyword": "退货政策"}},
        ],
        "write_previews": [], "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    assert score_case(by_id["MA-FRZ-018"], deny)["task_success"] is True
    assert score_case(by_id["MA-FRZ-023"], qualified)["task_success"] is True
    assert score_case(by_id["MA-FRZ-024"], cancelled)["task_success"] is True


@pytest.mark.asyncio
async def test_accurate_guard_measures_payload_not_message_class_schema():
    class Inner:
        def bind_tools(self, _tools, **_kwargs):
            return self

        async def ainvoke(self, _messages, **_kwargs):
            return AIMessage(content="ok", usage_metadata={
                "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
            })

    class Budget:
        max_output_tokens = 500

        def __init__(self):
            self.request_bytes = None

        def begin_generation(self, _arm, **kwargs):
            self.request_bytes = kwargs["request_bytes"]
            return "reservation"

        def complete_generation(self, *_args, **_kwargs):
            return None

    @tool
    def query_order(order_id: str) -> str:
        """Query one order."""
        return order_id

    budget = Budget()
    model = _AccurateGuardedChatModel(Inner(), budget, "single_agent", "agent")
    result = await model.bind_tools([query_order]).ainvoke("查询1001")

    assert result.content == "ok"
    assert budget.request_bytes is not None
    assert budget.request_bytes < 2500


def test_concurrent_paid_budget_allows_overlapping_reservations(tmp_path):
    pricing = PricingTable({
        "verified": True,
        "models": {"deepseek-flash": {
            "input_per_million": 0.30,
            "input_cache_hit_per_million": 0.006,
            "output_per_million": 1.20,
        }},
    })
    budget = ConcurrentPaidRunBudget(
        pricing, path=tmp_path / "ledger.sqlite3",
        experiment_id="parallel", config_hash="sha256:test",
        model="deepseek-flash",
        arm_limits={"multi_agent": ArmLimit(10, 0.1)},
        run_cap_usd=0.1, max_input_tokens=1000, max_output_tokens=100,
        allow_paid=True,
    )
    with paid_call_context(scenario_id="case", arm="multi_agent", turn=1, run_id="run"):
        first = budget.begin_generation(
            "multi_agent", request_bytes=100, scenario_id="case", turn=1,
            call_kind="retrieval-a", request={"messages": ["a"]},
        )
        second = budget.begin_generation(
            "multi_agent", request_bytes=100, scenario_id="case", turn=1,
            call_kind="retrieval-b", request={"messages": ["b"]},
        )
        assert budget.snapshot()["reserved_usd"] > 0
    budget.complete_generation(first, usage={
        "input_tokens": 10, "output_tokens": 2, "cache_hit_tokens": 0,
        "cache_miss_tokens": 10,
    })
    budget.complete_generation(second, usage={
        "input_tokens": 10, "output_tokens": 2, "cache_hit_tokens": 0,
        "cache_miss_tokens": 10,
    })
    assert budget.snapshot()["reserved_usd"] == 0
    assert budget.snapshot()["arms"]["multi_agent"]["generations"] == 2


def test_gzip_trace_roundtrip(tmp_path):
    path = tmp_path / "traces.jsonl.gz"
    rows = [{"scenario_id": "MA-1", "arm": "multi_agent", "content": "中文"}]
    _write_jsonl(path, rows)
    assert _read_jsonl(path) == rows
