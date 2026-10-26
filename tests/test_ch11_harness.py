"""分层 harness:L1 分母、闸误判、回答检查、缓存与预算。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import scripts.ch11_harness as ch11
from scripts.ch11_harness import (
    HarnessConfig,
    HarnessContext,
    dump_traces,
    run_answer_case,
    run_case,
    run_split,
)
from scripts.eval_ch11 import _summarize
from scripts.eval_cost import BudgetExceeded


class FakeService:
    def __init__(self, items=None, *, low_confidence=False, reason="", entailment=None):
        self.items = items or []
        self.low_confidence = low_confidence
        self.reason = reason
        self.entailment = entailment
        self.calls = 0

    def retrieve(self, query, strategy="hybrid_rerank", prepared=None):
        self.calls += 1
        return SimpleNamespace(
            items=self.items,
            low_confidence=self.low_confidence,
            reason=self.reason,
            filter_fallback=False,
            degraded=False,
            evidence_entailment=self.entailment,
        )


class FakeChat:
    def __init__(self, answer="普通订单邮费 8 元[1]。", usage=None):
        self.answer = answer
        self.usage = usage or {"input_tokens": 10, "output_tokens": 5}
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=self.answer, usage_metadata=self.usage)


class FakeCost:
    def __init__(self, *, raise_guard=False):
        self.raise_guard = raise_guard
        self.guarded = 0
        self.added = 0
        self.guard_kinds = []

    def guard(self, tokens, model, *, kind="api", request_text=""):
        self.guarded += 1
        self.guard_kinds.append(kind)
        if self.raise_guard:
            raise BudgetExceeded("over")

    def add(self, model, input_tokens, output_tokens, kind="answer"):
        self.added += 1
        return 0.0


class FakeCache:
    def __init__(self):
        self.data = {}

    def response_key(self, **kwargs):
        return json.dumps(kwargs, sort_keys=True)

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value


def _row(chunk_id=1):
    return SimpleNamespace(
        id=chunk_id,
        category="售后政策",
        questions="邮费",
        answer="普通订单邮费 8 元。",
        section_path="退货政策.md > 邮费与运费",
    )


def _case(**overrides):
    case = {
        "id": "T01",
        "split": "dev",
        "bucket": "single_hop",
        "query": "邮费多少",
        "expected_behavior": "answer",
        "answerable": True,
        "evidence_state": "sufficient",
        "gold_facts": [{"fact_id": "X", "value": 8, "unit": "元"}],
        "required_evidence": [{
            "doc": "退货政策.md",
            "section": "邮费与运费",
            "quotes": ["普通订单邮费 8 元。"],
        }],
        "required_conditions": [],
        "forbidden_claims": [],
        "clarifying_slots": [],
    }
    case.update(overrides)
    return case


def _ctx(service, chat=None, cost=None, cache=None):
    return HarnessContext(
        service=service,
        all_rows={1: _row()},
        gold_by_case={"T01": [frozenset({1})]},
        prepared={},
        chat=chat,
        judge=None,
        cache=cache,
        cost=cost,
    )


def _config(mode="retrieval", **overrides):
    payload = dict(split="dev", mode=mode, contract_sha="c", model="m")
    payload.update(overrides)
    return HarnessConfig(**payload)


def test_retrieval_trace_has_l1_fields_and_denominator_excludes_empty_evidence():
    service = FakeService(items=[SimpleNamespace(chunk_id=1, score=0.9)])
    ctx = _ctx(service)
    positive = run_case(_case(), ctx, _config())
    assert positive["l1"]["n"] == 1
    assert positive["l1"]["coverage_rank"] == 1
    negative = run_case(
        _case(id="T02", bucket="hard_negative", expected_behavior="refuse",
              required_evidence=[], gold_facts=[]),
        ctx,
        _config(),
    )
    assert negative["l1"]["n"] == 0
    assert negative["l1"]["coverage_rank"] is None


def test_evidence_gate_false_accept_and_false_block():
    accept = run_case(
        _case(id="H1", bucket="hard_negative", expected_behavior="refuse",
              required_evidence=[], gold_facts=[]),
        _ctx(FakeService(items=[SimpleNamespace(chunk_id=1, score=0.9)])),
        _config(),
    )
    block = run_case(
        _case(id="P1"),
        _ctx(FakeService(items=[], low_confidence=True, reason="低置信")),
        _config(),
    )
    summary = _summarize([accept, block])
    assert summary["l2"]["gate_false_accept"] == 1
    assert summary["l2"]["gate_false_block"] == 1


def test_forbidden_claim_and_under_refusal():
    chat = FakeChat(answer="可以便宜 10 元。")
    case = _case(expected_behavior="refuse", answerable=False,
                 evidence_state="insufficient", gold_facts=[],
                 required_evidence=[], forbidden_claims=["可以便宜"],
                 expected_refusal_reason="无依据")
    trace = run_case(case, _ctx(FakeService(items=[SimpleNamespace(chunk_id=1, score=0.9)]),
                                chat=chat), _config("full"))
    assert "forbidden_claim" in trace["failure_labels"]
    assert "under_refusal" in trace["failure_labels"]


def test_clarify_slot_trace():
    chat = FakeChat(answer="请问是哪款商品？")
    case = _case(expected_behavior="clarify", gold_facts=[], required_evidence=[],
                 clarifying_slots=["sku"], evidence_state="insufficient")
    trace = run_case(case, _ctx(FakeService(), chat=chat), _config("full"))
    assert trace["deterministic_checks"]["clarify_slot_recall"]["score"] == 1.0
    assert trace["deterministic_checks"]["clarify_precision"]["score"] == 1.0


def test_verify_case_compliance():
    chat = FakeChat(answer="这一定是质量问题。")
    case = _case(expected_behavior="verify_then_answer",
                 required_conditions=["是否属质量问题以检测结果为准"],
                 forbidden_claims=[], verification_markers=["检测"])
    trace = run_case(case, _ctx(FakeService(items=[SimpleNamespace(chunk_id=1, score=0.9)]),
                                chat=chat), _config("full"))
    assert "verification_overclaim" in trace["failure_labels"]


def test_full_mode_response_cache_avoids_second_paid_call():
    chat = FakeChat()
    cache = FakeCache()
    ctx = _ctx(FakeService(items=[SimpleNamespace(chunk_id=1, score=0.9)]),
               chat=chat, cache=cache, cost=FakeCost())
    config = _config("full")
    first = run_case(_case(), ctx, config)
    second = run_case(_case(), ctx, config)
    assert chat.calls == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert ctx.cost.guard_kinds == ["answer"]


def test_budget_exceeded_stops_before_llm_call():
    chat = FakeChat()
    ctx = _ctx(FakeService(items=[SimpleNamespace(chunk_id=1, score=0.9)]),
               chat=chat, cost=FakeCost(raise_guard=True))
    with pytest.raises(BudgetExceeded):
        run_case(_case(), ctx, _config("full"))
    assert chat.calls == 0


def test_rescore_preserves_answer_and_updates_deterministic_behavior():
    case = _case(
        expected_behavior="refuse",
        answerable=False,
        evidence_state="insufficient",
        gold_facts=[],
        required_evidence=[],
    )
    trace = {
        "case_id": "T01",
        "expected_behavior": "refuse",
        "answer": "现有证据未提供屏幕价格，无法给出报价，请转人工核验。",
        "retrieved": [{"rank": 1, "chunk_id": 1, "score": 0.9}],
        "rag_stop_reason": "",
    }

    rescored = ch11.rescore_answer_trace(case, trace, {1: _row()})

    assert rescored["answer"] == trace["answer"]
    assert rescored["behavior_pred"] == "refuse"
    assert rescored["deterministic_checks"]["refused"] is True
    assert rescored["failure_labels"] == []


def test_trace_jsonl_one_line_per_case(tmp_path):
    service = FakeService(items=[SimpleNamespace(chunk_id=1, score=0.9)])
    ctx = _ctx(service)
    traces = run_split([_case(), _case(id="T02")], ctx, _config())
    path = tmp_path / "traces.jsonl"
    dump_traces(traces, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_offline_mode_does_not_construct_paid_clients(monkeypatch, tmp_path):
    from scripts import eval_ch11

    def _boom(*args, **kwargs):
        raise AssertionError("offline 不应构造 Settings")

    monkeypatch.setattr("app.config.Settings", _boom)
    code = eval_ch11.main([
        "--mode", "offline", "--split", "dev", "--out", str(tmp_path)
    ])
    assert code == 0
    assert (tmp_path / "dev-offline" / "offline_report.json").exists()
