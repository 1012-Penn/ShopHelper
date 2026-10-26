import json
from pathlib import Path

import pytest

from tests.eval.typed_handoff_r2.scoring import score_case


ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ROOT / "tests" / "eval" / "typed_handoff_r2" / "scenarios.jsonl"


def _rows():
    return [
        json.loads(line) for line in SCENARIOS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_dataset_is_frozen_at_ten_dev_twenty_heldout_and_groups_do_not_cross_splits():
    rows = _rows()
    assert len(rows) == 30
    assert len({row["id"] for row in rows}) == 30
    assert sum(row["split"] == "dev" for row in rows) == 10
    assert sum(row["split"] == "held-out" for row in rows) == 20
    group_splits = {}
    for row in rows:
        group_splits.setdefault(row["group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_model_projection_contains_message_without_gold_fields():
    from scripts.typed_handoff_r2_eval import model_visible_turn

    row = _rows()[0]
    assert model_visible_turn(row, 0) == {"message": row["turns"][0]}
    assert "gold" not in json.dumps(model_visible_turn(row, 0))


def test_scorer_accepts_complete_answer_and_rejects_wrong_entity_and_condition():
    row = next(item for item in _rows() if item["id"] == "TH-HLD-023")
    good = {
        "reply": "订单1007跑步鞋待发货，订单1008显示器已签收。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1007"}},
            {"name": "query_order", "arguments": {"order_id": "1008"}},
        ],
        "write_previews": [], "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    bad = {
        **good,
        "reply": "订单1007显示器已签收，订单1008跑步鞋待发货。",
    }
    assert score_case(row, good)["task_success"] is True
    score = score_case(row, bad)
    assert score["task_success"] is False
    assert score["entity_error_count"] > 0


def test_scorer_accepts_correct_markdown_table_entity_relations():
    row = next(item for item in _rows() if item["id"] == "TH-HLD-024")
    result = {
        "reply": (
            "| 订单号 | 商品 | 状态 |\n"
            "| 1002 | 机械键盘 | 已发货 |\n"
            "| 1005 | 双肩包 | 已发货 |\n"
        ),
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1002"}},
            {"name": "query_order", "arguments": {"order_id": "1005"}},
        ],
        "write_previews": [], "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    score = score_case(row, result)
    assert score["task_success"] is True
    assert score["entity_error_count"] == 0


def test_scorer_treats_expired_old_policy_and_opposite_case_as_background():
    return_window = next(item for item in _rows() if item["id"] == "TH-DEV-004")
    shipping = next(item for item in _rows() if item["id"] == "TH-DEV-005")
    answers = [
        (return_window, "当前 v2 是7天。旧版 v1 的15天已于2026-10-15过期，不再适用。"),
        (shipping, "非质量问题由买家承担。质量问题由商家承担。"),
    ]
    for scenario, reply in answers:
        result = {
            "reply": reply,
            "tool_calls": [{"name": "query_faq", "arguments": {"keyword": "政策"}}],
            "write_previews": [], "confirmed_write_count": 0,
            "unconfirmed_write_count": 0,
        }
        score = score_case(scenario, result)
        assert score["task_success"] is True, (scenario["id"], score)


def test_scorer_does_not_treat_negated_forbidden_order_as_entity_error():
    row = next(item for item in _rows() if item["id"] == "TH-HLD-025")
    result = {
        "reply": "订单1003的工单预览已生成，待确认；本草稿未带入订单1004。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1003"}},
            {
                "name": "create_ticket",
                "arguments": {
                    "conversation_id": 1,
                    "description": "订单1003售后",
                    "ticket_type": "售后",
                },
            },
        ],
        "write_previews": [{"tool_name": "create_ticket"}],
        "confirmed_write_count": 1,
        "unconfirmed_write_count": 0,
        "action_visible_claims": [
            {"predicate": "product"}, {"predicate": "status"},
        ],
    }
    score = score_case(row, result)
    assert score["task_success"] is True, score
    assert score["entity_error_count"] == 0


def test_scorer_accepts_semantically_equivalent_no_submit_wording():
    row = next(item for item in _rows() if item["id"] == "TH-HLD-027")
    result = {
        "reply": "订单1007售后工单预览已生成，目前未提交，等待您确认。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1007"}},
            {
                "name": "create_ticket",
                "arguments": {
                    "conversation_id": 1,
                    "description": "订单1007售后",
                    "ticket_type": "售后",
                },
            },
        ],
        "write_previews": [{"tool_name": "create_ticket"}],
        "confirmed_write_count": 0,
        "unconfirmed_write_count": 0,
    }
    assert score_case(row, result)["task_success"] is True


def test_scorer_requires_confirmation_idempotency_contract():
    row = next(item for item in _rows() if item["id"] == "TH-DEV-010")
    result = {
        "reply": "订单1001的售后工单预览已生成，请确认；当前没有创建。",
        "tool_calls": [
            {"name": "query_order", "arguments": {"order_id": "1001"}},
            {"name": "create_ticket", "arguments": {
                "conversation_id": 1, "description": "订单1001无线耳机售后",
                "ticket_type": "售后",
            }},
        ],
        "write_previews": [{"tool_name": "create_ticket"}],
        "confirmed_write_count": 1,
        "unconfirmed_write_count": 0,
        "action_visible_claims": [
            {"predicate": "product"}, {"predicate": "status"},
        ],
    }
    assert score_case(row, result)["task_success"] is True
    assert score_case(row, {**result, "confirmed_write_count": 2})["task_success"] is False


@pytest.mark.asyncio
async def test_budgeted_model_retries_429_twice_and_keeps_every_attempt():
    from scripts.typed_handoff_r2_eval import BudgetedChatModel

    class Inner:
        calls = 0

        async def ainvoke(self, _value):
            self.calls += 1
            raise RuntimeError("HTTP 429 rate limit")

    class Budget:
        max_input_tokens = 16000
        max_output_tokens = 500
        attempts = []

        def begin_generation(self, arm, **kwargs):
            call_id = f"call-{len(self.attempts)}"
            self.attempts.append({"arm": arm, "call_id": call_id, **kwargs})
            return call_id

        def complete_generation(self, call_id, **kwargs):
            next(row for row in self.attempts if row["call_id"] == call_id).update(kwargs)

    inner = Inner()
    budget = Budget()
    model = BudgetedChatModel(
        inner, budget, "typed", "planner", retry_sleep=lambda _attempt: 0,
    )
    with pytest.raises(RuntimeError, match="429"):
        await model.ainvoke("prompt")
    assert inner.calls == 3
    assert len(budget.attempts) == 3
    assert all(row["error_type"] for row in budget.attempts)
