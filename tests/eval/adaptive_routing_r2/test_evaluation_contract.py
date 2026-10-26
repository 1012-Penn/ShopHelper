"""Contracts that keep the adaptive-routing comparison auditable."""

from __future__ import annotations

import json
import gzip
from pathlib import Path

import pytest

from scripts.eval_adaptive_routing_r2 import (
    ADAPTIVE_BUDGET_ARMS,
    ARM_NAMES,
    ARM_LIMITS,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MODEL,
    MANIFEST_PATH,
    RUN_CAP_USD,
    AdaptivePaidRunBudget,
    _pricing,
    build_manifest,
    load_scenarios,
    calls_for_attempt,
    paired_route_diagnostics,
    routing_input_for,
    score_case,
    validate_frozen_dataset,
    verify_manifest,
)
from scripts.ch15_paid_guard import paid_call_context


def test_dataset_is_frozen_as_10_dev_and_24_heldout_without_group_leakage() -> None:
    scenarios = load_scenarios()
    audit = validate_frozen_dataset(scenarios)
    assert audit["splits"] == {"dev": 10, "held-out": 24}
    assert audit["source_groups_cross_splits"] == []
    assert audit["expected_behavior_fields"] == 0
    assert audit["unique_ids"] == 34


def test_manifest_pins_every_policy_and_infrastructure_input() -> None:
    manifest = build_manifest()
    frozen = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert verify_manifest(frozen) == manifest
    assert frozen["status"] == "frozen-before-heldout"
    assert frozen["model"] == "deepseek-flash"
    assert frozen["run_cap_usd"] == RUN_CAP_USD == 2.0
    assert frozen["splits"] == {"dev": 10, "held-out": 24}
    assert len(frozen["scenario_sha256"]) == 64
    assert len(frozen["rubric_sha256"]) == 64
    assert len(frozen["router_policy_sha256"]) == 64
    assert len(frozen["harness_sha256"]) == 64
    assert len(frozen["executor_config_sha256"]) == 64
    assert frozen["fixtures"]["policies_sha256"]
    assert frozen["fixtures"]["orders_sha256"]
    assert frozen["stop_rule"]["heldout_run_count"] == 1
    assert frozen["stop_rule"]["heldout_retuning"] is False


def test_router_input_projection_never_contains_gold_or_scenario_identity() -> None:
    scenario = next(row for row in load_scenarios() if row["id"] == "AR2-HLD-017")
    projected = routing_input_for(scenario, turn_index=1)
    payload = projected.model_dump()
    assert set(payload) == {"question", "history", "visible_budget"}
    assert payload["question"] == "更正，改问订单1003的退货条件。"
    serialized = json.dumps(payload, ensure_ascii=False)
    assert scenario["id"] not in serialized
    assert "must_include" not in serialized
    assert "expected_behavior" not in serialized
    assert "吊牌" not in serialized
    assert len(payload["history"]) == 2
    assert all("订单1001" in item["content"] or item["role"] == "assistant"
               for item in payload["history"])


def _row(scenario_id: str, arm: str, success: bool, *, routes: list[str] | None = None):
    return {
        "scenario_id": scenario_id,
        "arm": arm,
        "success": success,
        "route_selected": (routes or ["single"])[-1],
        "route_targets": routes or ["single"],
        "duration_ms": 10,
        "model_calls": 1,
        "input_tokens": 10,
        "output_tokens": 2,
        "cost_usd": 0.001,
        "routing_ms": 1 if arm == "adaptive" else 0,
        "routing_cost_usd": 0.0,
    }


def test_route_diagnostics_identify_false_and_missed_upgrades() -> None:
    rows = [
        _row("s1", "fixed_single", True),
        _row("s1", "fixed_multi", False),
        _row("s1", "adaptive", True, routes=["single"]),
        _row("s2", "fixed_single", True),
        _row("s2", "fixed_multi", False),
        _row("s2", "adaptive", False, routes=["multi"]),
        _row("s3", "fixed_single", False),
        _row("s3", "fixed_multi", True),
        _row("s3", "adaptive", False, routes=["single"]),
    ]
    diagnostics = paired_route_diagnostics(rows)
    assert diagnostics["false_upgrade_ids"] == ["s2"]
    assert diagnostics["missed_upgrade_ids"] == ["s3"]
    assert diagnostics["adaptive_vs_single"] == {"wins": 0, "losses": 1, "ties": 2}
    assert diagnostics["adaptive_vs_multi"] == {"wins": 1, "losses": 1, "ties": 1}


def test_scorer_checks_required_tools_entities_and_preview_confirmation() -> None:
    scenario = next(row for row in load_scenarios() if row["id"] == "AR2-HLD-023")
    result = {
        "reply": "订单1002的工单预览已生成，请在卡片上确认。",
        "disposition": "",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1002"}},
            {"name": "create_ticket", "arguments": {
                "conversation_id": 1, "description": "售后", "ticket_type": "售后"
            }},
        ],
        "write_previews": [{"type": "ticket_preview"}],
        "confirmed_write_count": 1,
        "unconfirmed_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["task_success"] is True
    assert score["write_correct"] is True

    result["confirmed_write_count"] = 0
    score = score_case(scenario, result)
    assert score["task_success"] is False
    assert score["write_correct"] is False


def test_clarification_is_not_misread_as_answer_when_reply_mentions_status() -> None:
    scenario = next(row for row in load_scenarios() if row["id"] == "AR2-HLD-020")
    result = {
        "reply": "我没有看到这单目前的状态，方便提供一下订单号吗？",
        "disposition": "",
        "tool_calls": [],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["actual_outcome"] == "clarify"
    assert score["task_success"] is True


def test_policy_answer_with_optional_order_offer_remains_an_answer() -> None:
    scenario = next(row for row in load_scenarios() if row["id"] == "AR2-HLD-002")
    result = {
        "reply": "质量问题退货的运费由商家承担。您方便提供订单号吗？",
        "disposition": "",
        "tool_calls": [{"name": "query_faq", "arguments": {"keyword": "质量问题运费"}}],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["actual_outcome"] == "answer"
    assert score["task_success"] is True


def test_substantive_order_facts_win_over_refusal_prefix() -> None:
    scenario = next(row for row in load_scenarios() if row["id"] == "AR2-HLD-013")
    result = {
        "reply": (
            "【无法回答】订单1001已签收，订单1002已发货；"
            "还需要确认您想处理退货还是物流。"
        ),
        "disposition": "",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1001"}},
            {"name": "query_order", "arguments": {"order_id": "1002"}},
        ],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["actual_outcome"] == "answer"
    assert score["task_success"] is True


def test_markdown_does_not_defeat_negative_context_exception() -> None:
    scenario = next(row for row in load_scenarios() if row["id"] == "AR2-HLD-018")
    result = {
        "reply": (
            "非质量问题由买家承担。对比参考："
            "**质量问题退货的运费由商家承担**。"
        ),
        "disposition": "",
        "tool_calls": [{"name": "query_order"}, {"name": "query_faq"}],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["forbidden_conditions"] == []


def test_cancel_revision_synonyms_are_scored_without_changing_original_gold() -> None:
    scenario = next(row for row in load_scenarios() if row["id"] == "AR2-HLD-019")
    result = {
        "reply": "好的，那就不退货、也不建工单啦。",
        "disposition": "",
        "tool_calls": [{"name": "query_order"}, {"name": "query_faq"}],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    score = score_case(scenario, result)
    assert "不会" not in score["missing_conditions"]


def test_final_artifact_contains_all_102_scored_branches() -> None:
    path = (
        Path(__file__).parent / "runs" / "final-rescore-v3"
        / "traces.jsonl.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    assert len(rows) == 102
    assert {row["arm"] for row in rows} == set(ARM_NAMES)
    assert {row["score"]["scoring_version"] for row in rows} == {
        "adaptive_routing_r2_scorer_v3"
    }


def test_saved_model_interactions_do_not_contain_evaluator_gold() -> None:
    path = (
        Path(__file__).parent / "runs" / "final-rescore-v3"
        / "traces.jsonl.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    requests = json.dumps(
        [
            interaction["request"]
            for row in rows for interaction in row.get("model_interactions", [])
        ],
        ensure_ascii=False,
    )
    for forbidden in (
        "expected_behavior", "must_include", "must_not_include",
        "required_tools", "write_mode", "gold",
    ):
        assert forbidden not in requests


def test_frozen_manifest_exists_and_uses_three_named_arms() -> None:
    assert MANIFEST_PATH.exists()
    assert ARM_NAMES == ("fixed_single", "fixed_multi", "adaptive")


def test_isolated_budget_supports_multiple_pending_reservations(tmp_path) -> None:
    budget = AdaptivePaidRunBudget(
        _pricing(),
        path=tmp_path / "adaptive-ledger.sqlite3",
        experiment_id="adaptive-routing-r2-test",
        config_hash="f" * 64,
        model=MODEL,
        arm_limits=ARM_LIMITS,
        run_cap_usd=RUN_CAP_USD,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        allow_paid=True,
    )
    with paid_call_context(
        scenario_id="offline-case", arm="adaptive_single", turn=1, run_id="run"
    ):
        first = budget.begin_generation(
            "adaptive_single",
            request_bytes=100,
            scenario_id="offline-case",
            turn=1,
            call_kind="first",
            request={"message": "one"},
        )
        second = budget.begin_generation(
            "adaptive_single",
            request_bytes=100,
            scenario_id="offline-case",
            turn=1,
            call_kind="second",
            request={"message": "two"},
        )
    snapshot = budget.snapshot()
    assert snapshot["run_cap_usd"] == 2.0
    assert snapshot["arms"]["adaptive_single"]["pending_calls"] == 2
    assert snapshot["reserved_usd"] > 0
    budget.complete_generation(
        first, usage={"input_tokens": 20, "output_tokens": 2}
    )
    budget.complete_generation(
        second, usage={"input_tokens": 20, "output_tokens": 2}
    )
    assert budget.snapshot()["reserved_usd"] == 0


def test_attempt_metrics_exclude_previous_runs_of_the_same_case() -> None:
    class _Ledger:
        def calls_for_case(self, _scenario_id: str, _arm: str):
            return [
                {"run_id": "old:AR2-DEV-001:turn-1", "charged_usd": 0.1},
                {"run_id": "new:AR2-DEV-001:turn-1", "charged_usd": 0.2},
                {"run_id": "new:AR2-DEV-001:turn-2", "charged_usd": 0.3},
            ]

    calls = calls_for_attempt(
        _Ledger(), "AR2-DEV-001", ("fixed_single",), "new:AR2-DEV-001:"
    )
    assert [call["charged_usd"] for call in calls] == [0.2, 0.3]
