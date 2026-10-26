from __future__ import annotations

from app.prompts import ResolvedQuestion
from langchain_core.messages import AIMessage
from scripts.e2e_quality_r2.budget import ExactCallCache, R2Budget
from scripts.e2e_quality_r2.harness import (
    BudgetedChat,
    BudgetedJsonStructured,
    TraceCapture,
    infer_recorded_call_kind,
    json_structured_instruction,
)


class _FakeChat:
    def __init__(self):
        self.requests = []

    def invoke(self, request, config=None):
        self.requests.append(request)
        return AIMessage(
            content='```json\n{"resolved":"SH-E300 多少钱","changed":true}\n```',
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
            },
        )


class _ZeroPricing:
    def cost(self, *_args, **_kwargs):
        return 0.0


def test_json_structured_instruction_contains_schema_not_gold():
    instruction = json_structured_instruction({
        "title": "ResolvedQuestion",
        "type": "object",
    })
    assert "JSON" in instruction
    assert "ResolvedQuestion" in instruction
    assert "gold_facts" not in instruction


def test_json_structured_wrapper_parses_and_records_full_exchange(tmp_path):
    inner = _FakeChat()
    capture = TraceCapture()
    capture.start_case(case={"id": "D1", "facts": []}, split="dev", turn=0)
    wrapper = BudgetedJsonStructured(
        inner=inner,
        model="deepseek-flash",
        pricing=_ZeroPricing(),
        budget=R2Budget(tmp_path / "budget.json", limit_usd=2.0),
        cache=ExactCallCache(tmp_path / "cache.json"),
        capture=capture,
        namespace="test",
        kind="resolve",
        schema=ResolvedQuestion.model_json_schema(),
        schema_class=ResolvedQuestion,
    )

    result = wrapper.invoke("最新消息: SH-E300 多少钱")

    assert result == ResolvedQuestion(resolved="SH-E300 多少钱", changed=True)
    assert "ResolvedQuestion" in inner.requests[0]
    calls = capture.case_model_calls("D1")
    assert calls[0]["outcome"] == "succeeded"
    assert "SH-E300 多少钱" in calls[0]["output"]["content"]


def test_json_fallback_uses_schema_name_for_cost_kind(tmp_path):
    wrapper = BudgetedChat(
        inner=_FakeChat(),
        model="deepseek-flash",
        pricing=_ZeroPricing(),
        budget=R2Budget(tmp_path / "budget.json", limit_usd=2.0),
        cache=ExactCallCache(tmp_path / "cache.json"),
        capture=TraceCapture(),
        namespace="test",
        kind="llm",
        structured_mode="json-fallback",
    )

    structured = wrapper.with_structured_output(ResolvedQuestion)

    assert structured.kind == "resolve"


def test_old_structured_json_calls_are_reclassified_from_captured_prompt():
    assert infer_recorded_call_kind({
        "kind": "structured_json",
        "input": {"request": {"input": "你是证据审核员。证据是否回答？"}},
    }) == "evidence_judge"
    assert infer_recorded_call_kind({
        "kind": "structured_json",
        "input": {"schema": {"title": "ResolvedQuestion"}},
    }) == "resolve"
    assert infer_recorded_call_kind({
        "kind": "structured_json",
        "input": {"input": {"input": "把客服原话改写成标准 FAQ 问题"}},
    }) == "flywheel_normalize"


def test_chat_retries_429_at_most_twice_then_records_only_final_failure_state(tmp_path):
    class RateLimited:
        def __init__(self):
            self.calls = 0

        def invoke(self, request, config=None):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("429 rate limit")
            return AIMessage(content="ok", usage_metadata={
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2
            })

    inner = RateLimited()
    capture = TraceCapture()
    capture.start_case(case={"id": "D1", "facts": []}, split="dev", turn=0)
    wrapper = BudgetedChat(
        inner=inner,
        model="deepseek-flash",
        pricing=_ZeroPricing(),
        budget=R2Budget(tmp_path / "budget.json", limit_usd=2.0),
        cache=ExactCallCache(tmp_path / "cache.json"),
        capture=capture,
        namespace="test",
        kind="llm",
    )

    response = wrapper.invoke("request")

    assert response.content == "ok"
    assert inner.calls == 3
    assert capture.case_failures("D1") == []
