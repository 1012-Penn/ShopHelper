"""Frozen-data and scoring tests for the action-contract harness."""

from __future__ import annotations

import json

from tests.eval.action_contract_r3.harness import (
    config_sha256,
    load_scenarios,
    manifest_sha256,
    model_projection,
    score_case,
)


def test_splits_groups_hashes_and_gold_projection_are_frozen():
    scenarios = load_scenarios()
    assert len(scenarios) == 28
    assert sum(row["split"] == "dev" for row in scenarios) == 8
    assert sum(row["split"] == "held-out" for row in scenarios) == 20
    assert len(manifest_sha256()) == 64
    assert len(config_sha256()) == 64
    projection = model_projection(scenarios[0], 0, [])
    assert projection == {
        "message": scenarios[0]["workflow"][0]["message"],
        "history": [],
    }
    assert "gold" not in json.dumps(projection)


def test_score_requires_structured_preview_and_records_type_error():
    scenario = next(row for row in load_scenarios() if row["id"] == "AC-DEV-003")
    result = {
        "reply": "订单1001工单预览已生成。",
        "disposition": "",
        "tool_calls": [{
            "name": "create_ticket",
            "arguments": {
                "conversation_id": 1,
                "description": "订单1001售后",
                "ticket_type": "退款",
            },
        }],
        "write_previews": [{
            "ticket_type": "退款",
            "arguments": {
                "conversation_id": 1,
                "description": "订单1001售后",
                "ticket_type": "退款",
            },
        }],
        "confirmed_write_count": 1,
        "unconfirmed_write_count": 0,
        "duplicate_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["preview_correct"] is True
    assert score["parameter_error_count"] >= 1
    assert score["ticket_type_correct"] is False
    assert score["task_success"] is False


def test_claim_without_preview_is_not_scored_as_success():
    scenario = next(row for row in load_scenarios() if row["id"] == "AC-DEV-003")
    result = {
        "reply": "订单1001的工单预览已经生成，请确认。",
        "disposition": "",
        "tool_calls": [],
        "write_previews": [],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
        "duplicate_write_count": 0,
    }
    assert score_case(scenario, result)["task_success"] is False


def test_extra_preview_on_feasibility_question_is_failure():
    scenario = next(row for row in load_scenarios() if row["id"] == "AC-DEV-002")
    result = {
        "reply": "订单1002已发货。",
        "disposition": "",
        "tool_calls": [],
        "write_previews": [{"ticket_type": "售后"}],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
        "duplicate_write_count": 0,
    }
    score = score_case(scenario, result)
    assert score["extra_action_count"] == 1
    assert score["task_success"] is False


def test_same_key_confirmation_replay_is_not_counted_as_duplicate_write():
    scenario = next(row for row in load_scenarios() if row["id"] == "AC-DEV-003")
    result = {
        "reply": "订单1001工单预览已生成，等待确认。",
        "disposition": "",
        "tool_calls": [{
            "name": "create_ticket",
            "arguments": {
                "conversation_id": 1,
                "description": "订单1001售后",
                "ticket_type": "售后",
            },
        }],
        "write_previews": [{
            "ticket_type": "售后",
            "arguments": {
                "conversation_id": 1,
                "description": "订单1001售后",
                "ticket_type": "售后",
            },
        }],
        "confirmed_write_count": 1,
        "unconfirmed_write_count": 0,
        "duplicate_write_count": 0,
        "confirmation_replay_count": 1,
    }
    score = score_case(scenario, result)
    assert score["duplicate_write_count"] == 0
    assert score["task_success"] is True
