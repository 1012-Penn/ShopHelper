"""TDD contract for the cost-aware routing boundary."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.adaptive_routing import (
    AdaptiveRouter,
    HistoryTurn,
    JudgeSignal,
    RouteDispatcher,
    RoutingInput,
    VisibleRunBudget,
    extract_routing_features,
)
from app.adaptive_routing.api import (
    SessionAwareAdaptiveService,
    create_adaptive_routing_app,
)
from scripts.eval_adaptive_routing_r2 import _RetryingGuardedModel


def _input(question: str, history: list[dict] | None = None) -> RoutingInput:
    return RoutingInput(
        question=question,
        history=[HistoryTurn.model_validate(item) for item in (history or [])],
        visible_budget=VisibleRunBudget(
            remaining_seconds=30.0, remaining_cost_usd=0.05, max_steps=12
        ),
    )


def test_routing_input_rejects_hidden_gold_fields() -> None:
    with pytest.raises(ValidationError):
        RoutingInput.model_validate(
            {
                "question": "订单1001能退吗？",
                "history": [],
                "visible_budget": {},
                "gold": {"outcome": "answer"},
            }
        )


def test_simple_policy_turn_routes_to_single_agent() -> None:
    decision = AdaptiveRouter().route(_input("签收后无理由退货窗口是多少天？"))
    assert decision.target == "single"
    assert "simple_policy" in decision.reason_codes
    assert decision.features.entity_count == 0
    assert decision.features.policy_order_dependency is False


def test_cross_order_policy_turn_routes_to_multi_agent() -> None:
    decision = AdaptiveRouter().route(
        _input("订单1002和订单1003哪个能直接取消？")
    )
    assert decision.target == "multi"
    assert "cross_order_comparison" in decision.reason_codes
    assert decision.features.entity_count == 2
    assert decision.features.cross_order_comparison is True
    assert decision.features.policy_order_dependency is True


def test_revision_across_orders_routes_to_multi_agent() -> None:
    decision = AdaptiveRouter().route(
        _input(
            "更正，改问订单1003能否取消。",
            [
                {"role": "user", "content": "订单1001能取消吗？"},
                {"role": "assistant", "content": "订单1001已签收，不能直接取消。"},
            ],
        )
    )
    assert decision.features.entity_count == 2
    assert decision.features.correction is True
    assert decision.target == "multi"
    assert "revision_with_order_dependency" in decision.reason_codes


def test_missing_slot_clarification_stays_single_agent() -> None:
    decision = AdaptiveRouter().route(_input("这个能退吗？"))
    assert decision.features.missing_slot is True
    assert decision.target == "single"
    assert "missing_slot_clarify" in decision.reason_codes


def test_write_risk_is_visible_but_does_not_upgrade_permissions() -> None:
    decision = AdaptiveRouter().route(
        _input("请为订单1001创建售后工单，先给我预览，不要提交。")
    )
    assert decision.features.write_confirmation_risk is True
    assert decision.target == "single"
    assert "write_requires_confirmation" in decision.reason_codes
    assert all("permission" not in reason for reason in decision.reason_codes)


def test_feature_extraction_uses_only_question_and_visible_history() -> None:
    features = extract_routing_features(
        "那订单1003呢？",
        [HistoryTurn(role="user", content="先看订单1002能否取消")],
    )
    assert features.visible_order_ids == ("1002", "1003")
    assert features.entity_count == 2


class _RaisingSingle:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, _routing_input: RoutingInput):
        self.calls += 1
        raise RuntimeError("single failed")


class _ForbiddenMulti:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, _routing_input: RoutingInput):
        self.calls += 1
        return {"final_reply": "should not run"}


def test_dispatcher_calls_exactly_one_arm_and_never_falls_back() -> None:
    single = _RaisingSingle()
    multi = _ForbiddenMulti()
    result = asyncio.run(
        RouteDispatcher().dispatch(
            _input("签收后退货窗口是多少天？"), single_executor=single, multi_executor=multi
        )
    )
    assert result.selected_arm == "single"
    assert result.selected_executor_calls == 1
    assert result.other_executor_calls == 0
    assert single.calls == 1
    assert multi.calls == 0
    assert "single failed" in result.error
    assert result.response.reply == ""


class _SuccessfulSingle:
    async def __call__(self, _routing_input: RoutingInput):
        await asyncio.sleep(0.01)
        return {
            "final_reply": "普通运费为 8 元。",
            "conversation_id": 17,
            "write_previews": [],
        }


class _UnexpectedMulti:
    async def __call__(self, _routing_input: RoutingInput):
        raise AssertionError("multi executor must not run")


def test_dispatcher_normalizes_existing_graph_shape_and_counts_route_time() -> None:
    result = asyncio.run(
        RouteDispatcher().dispatch(
            _input("普通运费是多少？"),
            single_executor=_SuccessfulSingle(),
            multi_executor=_UnexpectedMulti(),
        )
    )
    assert result.response.reply == "普通运费为 8 元。"
    assert result.response.conversation_id == 17
    assert result.raw_response is not None
    assert result.total_ms >= result.routing_ms
    assert result.total_ms >= result.execution_ms
    assert result.routing_cost_usd == 0.0


class _LowCostJudge:
    def __init__(self) -> None:
        self.calls = 0

    async def assess(self, routing_input: RoutingInput, features):
        self.calls += 1
        assert routing_input.question == "订单1001和订单1002有售后问题，接下来怎么办？"
        assert features.entity_count == 2
        return JudgeSignal(
            target="multi",
            confidence=0.91,
            reason="cross_entity_dependency",
            input_tokens=40,
            output_tokens=4,
            cost_usd=0.00002,
        )


def test_optional_judge_runs_once_only_in_uncertain_band_and_costs_are_visible() -> None:
    judge = _LowCostJudge()
    router = AdaptiveRouter(judge=judge)
    decision = asyncio.run(
        router.route_async(
            _input(
                "订单1001和订单1002有售后问题，接下来怎么办？",
                [{"role": "assistant", "content": "可以继续。"}],
            )
        )
    )
    assert judge.calls == 1
    assert decision.judge_used is True
    assert decision.judge_confidence == pytest.approx(0.91)
    assert decision.judge_input_tokens == 40
    assert decision.judge_output_tokens == 4
    assert decision.routing_cost_usd == pytest.approx(0.00002)


def test_optional_judge_is_skipped_when_rule_is_decisive() -> None:
    judge = _LowCostJudge()
    router = AdaptiveRouter(judge=judge)
    decision = asyncio.run(router.route_async(_input("普通运费是多少？")))
    assert judge.calls == 0
    assert decision.judge_used is False
    assert decision.routing_cost_usd == 0.0


def test_policy_signature_is_stable_and_contains_no_threshold_specific_gold() -> None:
    first = AdaptiveRouter().policy_signature()
    second = AdaptiveRouter().policy_signature()
    assert first == second
    assert len(first) == 64
    assert "MA-" not in first
    assert "gold" not in first.casefold()


def test_routing_is_cheap_local_work_without_hidden_model_calls() -> None:
    started = time.perf_counter()
    decision = AdaptiveRouter().route(
        _input("请比较订单1001和订单1003的状态以及下一步处理方式。")
    )
    elapsed = time.perf_counter() - started
    assert decision.target == "multi"
    assert decision.routing_cost_usd == 0.0
    assert elapsed < 0.25


class _SessionService:
    def __init__(self) -> None:
        self.calls: list[tuple[RoutingInput, int | None]] = []

    async def respond(
        self, routing_input: RoutingInput, *, session_id: int | None = None
    ):
        self.calls.append((routing_input, session_id))
        execution = await RouteDispatcher().dispatch(
            routing_input,
            single_executor=_SuccessfulSingle(),
            multi_executor=_UnexpectedMulti(),
        )
        return execution


def test_optional_app_passes_session_to_injected_existing_service() -> None:
    service = _SessionService()
    app = create_adaptive_routing_app(service)
    response = TestClient(app).post(
        "/api/adaptive-routing/chat",
        json={
            "message": "普通运费是多少？",
            "session_id": 23,
            "history": [{"role": "user", "content": "你好"}],
            "visible_budget": {"remaining_seconds": 20, "remaining_cost_usd": 0.02},
        },
    )
    assert response.status_code == 200
    assert response.json()["selected_arm"] == "single"
    assert response.json()["response"]["reply"] == "普通运费为 8 元。"
    assert len(service.calls) == 1
    assert service.calls[0][0].question == "普通运费是多少？"
    assert service.calls[0][1] == 23


def test_optional_api_rejects_gold_before_the_service_runs() -> None:
    service = _SessionService()
    app = create_adaptive_routing_app(service)
    response = TestClient(app).post(
        "/api/adaptive-routing/chat",
        json={
            "message": "普通运费是多少？",
            "session_id": 23,
            "history": [],
            "visible_budget": {},
            "expected_behavior": "answer",
        },
    )
    assert response.status_code == 422
    assert service.calls == []


class _RateLimitedOnce:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 rate limit")
        return "ok"

    async def astream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 rate limit")
        yield "ok"


class _ClearRecorder:
    def __init__(self) -> None:
        self.clears = 0

    def clear_rate_limit_stop(self) -> None:
        self.clears += 1


def test_rate_limit_retry_is_bounded_and_clears_only_current_stop(monkeypatch) -> None:
    inner = _RateLimitedOnce()
    budget = _ClearRecorder()
    wrapper = _RetryingGuardedModel(inner, budget)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    assert asyncio.run(wrapper.ainvoke([])) == "ok"
    assert inner.calls == 2
    assert budget.clears == 1


def test_streaming_entrypoint_exists_and_retries_before_first_chunk(monkeypatch) -> None:
    inner = _RateLimitedOnce()
    budget = _ClearRecorder()
    wrapper = _RetryingGuardedModel(inner, budget)

    async def no_sleep(_seconds):
        return None

    async def collect():
        return [item async for item in wrapper.astream([])]

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    assert asyncio.run(collect()) == ["ok"]
    assert inner.calls == 2
    assert budget.clears == 1
