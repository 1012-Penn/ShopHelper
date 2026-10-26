"""ch12 评测三组模式测试:检索段分派、rag 轨迹字段、GuardedGapAnalyzer 缓存/守卫、
三组契约差异仅 rag./harness. 块、BudgetExceeded 穿透。全部替身,不触网。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.evidence import EvidenceVerdict
from app.prompts import GAP_ANALYZER_PROMPT, GapReport
from app.rag_loop import STOP_FIRST_PASS, STOP_ROUNDS
from scripts.ch11_harness import HarnessConfig, HarnessContext, run_case
from scripts.eval_contract import build_contract, contract_diff
from scripts.eval_cost import BudgetExceeded, GuardedGapAnalyzer


class FakeRagService:
    """按查询脚本返回结果的检索替身;暴露 loop 消费的 _evidence_judge/_evidence_calibration。"""

    def __init__(self, script=None, default=None):
        from app.retrieval import EvidenceCalibration

        self._script = script or {}
        self._default = default
        self._evidence_calibration = EvidenceCalibration()
        self._evidence_judge = None
        self.calls: list[str] = []

    def retrieve(self, query, strategy="hybrid_rerank", category=None,
                 use_rewrite=True, prepared=None):
        self.calls.append(query)
        action = self._script.get(query, self._default)
        if isinstance(action, Exception):
            raise action
        if action is None:
            action = SimpleNamespace(items=[], low_confidence=True,
                                     reason="知识库无相关内容", filter_fallback=False,
                                     degraded=False, evidence_entailment=None)
        return action


class FakeJudgeObj:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def judge(self, query, documents):
        self.calls += 1
        return EvidenceVerdict(answered=self.answers.pop(0) if self.answers else False,
                               supported_by=[], reason="")


class FakeAnalyzerObj:
    def __init__(self, reports):
        self.reports = list(reports)
        self.calls = 0

    def analyze(self, query, evidence):
        self.calls += 1
        report = self.reports.pop(0)
        if isinstance(report, Exception):
            raise report
        return report


class FakeChat:
    def __init__(self, answer="普通订单邮费 8 元[1]。"):
        self.answer = answer
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=self.answer,
                               usage_metadata={"input_tokens": 10, "output_tokens": 5})


class FakeCost:
    def __init__(self, *, raise_guard=False):
        self.raise_guard = raise_guard
        self.guarded = 0
        self.added = []
        self.guard_kinds = []

    def guard(self, tokens, model, *, kind="api", request_text=""):
        self.guarded += 1
        self.guard_kinds.append(kind)
        if self.raise_guard:
            raise BudgetExceeded("over")

    def add(self, model, input_tokens, output_tokens, kind="answer"):
        self.added.append(kind)
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
        id=chunk_id, category="售后政策", questions="邮费",
        answer=f"chunk{chunk_id} 的答案。", section_path="退货政策.md > 邮费与运费")


def _case(**overrides):
    case = {
        "id": "T01", "split": "frozen-v1", "bucket": "single_hop",
        "query": "邮费多少", "expected_behavior": "answer", "answerable": True,
        "evidence_state": "sufficient",
    }
    case.update(overrides)
    return case


def _result(items, *, low=False, entailment=None):
    return SimpleNamespace(items=items, low_confidence=low, reason="" if not low else "低置信",
                           filter_fallback=False, degraded=False,
                           evidence_entailment=entailment, evidence_confidence=None)


def _ctx(service, rows, *, chat=None, cost=None, cache=None, analyzer=None, gold=None):
    return HarnessContext(
        service=service, all_rows=rows, gold_by_case=gold or {},
        prepared={}, chat=chat, judge=None, cache=cache, cost=cost, gap_analyzer=analyzer)


def _config(mode="full", rag_mode="baseline", **kw):
    return HarnessConfig(split="frozen-v1", mode=mode, rag_mode=rag_mode, **kw)


ROWS = {1: _row(1), 2: _row(2), 3: _row(3)}
HIGH = _result([SimpleNamespace(chunk_id=1, score=0.9)], low=False, entailment=True)
LOW = _result([SimpleNamespace(chunk_id=1, score=0.2)], low=True, entailment=False)
SUB_HIGH = _result([SimpleNamespace(chunk_id=2, score=0.9)], low=False, entailment=True)


def test_baseline_mode_single_retrieve_no_rag_fields():
    service = FakeRagService({"邮费多少": HIGH})
    trace = run_case(_case(), _ctx(service, ROWS), _config())
    assert service.calls == ["邮费多少"]
    assert "rag_stop_reason" not in trace
    assert trace["gate"]["passed"] is True


def test_rag_mode_first_pass_sufficient():
    service = FakeRagService({"邮费多少": HIGH})
    analyzer = FakeAnalyzerObj([])
    trace = run_case(_case(), _ctx(service, ROWS, analyzer=analyzer),
                     _config(rag_mode="controlled_loop"))
    assert trace["rag_stop_reason"] == STOP_FIRST_PASS
    assert trace["rag_rounds"] == 0 and trace["rag_analyzer_calls"] == 0
    assert service.calls == ["邮费多少"]  # 首轮即充分:与基线同一调用,零额外检索
    assert trace["gate"]["passed"] is True


def test_rag_mode_decompose_once_full_flow():
    service = FakeRagService({"邮费多少": LOW, "包邮门槛是多少": SUB_HIGH})
    judge = FakeJudgeObj([False, False])  # 轮首与终判都说没回答
    service._evidence_judge = judge
    analyzer = FakeAnalyzerObj([GapReport(action="retrieve_more", sub_queries=["包邮门槛是多少"])])
    trace = run_case(_case(), _ctx(service, ROWS, analyzer=analyzer),
                     _config(rag_mode="decompose_once"))
    assert trace["rag_stop_reason"] == STOP_ROUNDS
    assert trace["rag_rounds"] == 1
    assert service.calls == ["邮费多少", "包邮门槛是多少"]
    assert [s["round_index"] for s in trace["rag_steps"]] == [0, 1, 1]
    assert [s["phase"] for s in trace["rag_steps"]] == ["retrieve", "retrieve", "final"]
    assert trace["rag_steps"][1]["sub_queries"] == ["包邮门槛是多少"]
    assert trace["gate"]["passed"] is False  # 终判仍未过 → 低置信拒答口径
    assert trace["rag_analyzer_calls"] == 1


def test_rag_mode_stop_reason_time_budget_labels_timeout():
    from app.rag_loop import RagLoopLimits  # noqa: F401  (确认可导入)

    service = FakeRagService({"邮费多少": LOW})
    analyzer = FakeAnalyzerObj([GapReport(action="retrieve_more", sub_queries=["x"])])
    config = _config(rag_mode="controlled_loop", rag_time_budget_s=0.0)
    trace = run_case(_case(), _ctx(service, ROWS, analyzer=analyzer), config)
    assert trace["rag_stop_reason"] == "time_budget"
    assert "timeout" in trace["failure_labels"]  # 既有标签位补记


def test_budget_exceeded_from_analyzer_propagates():
    service = FakeRagService({"邮费多少": LOW})
    analyzer = FakeAnalyzerObj([BudgetExceeded("累计预算超限")])
    with pytest.raises(BudgetExceeded):
        run_case(_case(), _ctx(service, ROWS, analyzer=analyzer),
                 _config(rag_mode="controlled_loop"))


class _Inner:
    def __init__(self):
        self.calls = 0

    def analyze(self, query, evidence):
        self.calls += 1
        return GapReport(action="retrieve_more", sub_queries=["q1"], reason="r")


def test_guarded_gap_analyzer_cache_hit_skips_guard_and_cost():
    cost, cache, inner = FakeCost(), FakeCache(), _Inner()
    wrapped = GuardedGapAnalyzer(inner, cost, cache, "m", "sha")
    cache.data[json.dumps({"model": "m", "prompt_sha256": "sha", "temperature": 0.0,
                           "query": "q\0e", "evidence_ids": []}, sort_keys=True)] = {
        "sufficient": True, "action": "answer", "missing_aspects": [], "conflicts": [],
        "sub_queries": [], "clarify_question": "", "reason": "hit"}
    report = wrapped.analyze("q", "e")
    assert report.sufficient is True and report.action == "answer"
    assert wrapped.hits == 1 and cost.guarded == 0 and inner.calls == 0  # 命中不进守卫不计费


def test_guarded_gap_analyzer_miss_guards_then_records():
    cost, cache, inner = FakeCost(), FakeCache(), _Inner()
    wrapped = GuardedGapAnalyzer(inner, cost, cache, "m", "sha")
    report = wrapped.analyze("q", "e")
    assert report.action == "retrieve_more"
    assert cost.guarded == 1 and inner.calls == 1 and cost.added == ["gap_analyzer"]
    assert cost.guard_kinds == ["gap_analyzer"]
    assert wrapped.calls == 1 and len(cache.data) == 1  # 结果已写缓存,复跑零费用


def test_guarded_gap_analyzer_guard_blocks_before_call():
    cost, cache, inner = FakeCost(raise_guard=True), FakeCache(), _Inner()
    wrapped = GuardedGapAnalyzer(inner, cost, cache, "m", "sha")
    with pytest.raises(BudgetExceeded):
        wrapped.analyze("q", "e")
    assert inner.calls == 0  # 守卫先于调用,不产生付费调用


def _contract_for(rag_mode: str) -> dict:
    same = {
        "dataset": {"split": "frozen-v1", "sha256": "d" * 8, "n": 90},
        "corpus": {"docs_sha256": "c" * 8, "facts_sha256": "f" * 8, "doc_count": 8, "chunk_count": 130},
        "index": {"backend": "milvus-lite", "build_id": "b" * 8, "embedding_model": "bge", "dim": 1024},
        "retrieval": {"strategy": "hybrid_rerank", "candidates": 50, "top_k": 10},
        "rewrite": {"enabled": True, "model": "deepseek-flash", "mode": "frozen", "cache_sha256": "r" * 8},
        "llm": {"model": "deepseek-flash", "temperature": 0.0, "prompt_sha256": "p" * 8},
        "judge": {"model": "deepseek-flash", "temperature": 0, "prompt_sha256": "j" * 8,
                  "enabled": False, "evidence_model": "deepseek-flash", "evidence_temperature": 0,
                  "evidence_prompt_sha256": "e" * 8, "evidence_enabled": True},
        "pricing": {"version": "2026-10-25", "currency": "USD", "verified": True, "sha256": "$" * 8},
        "run": {"git_commit": "x", "seed": 1, "python": "3.10.12", "host": ""},
    }
    return build_contract(
        **same,
        harness={"version": "ch11-harness-1" if rag_mode == "baseline" else "ch11-harness-1-rag",
                 "mode": "full", "allow_paid": True, "use_judge": False, "rag_mode": rag_mode},
        rag={"mode": rag_mode,
             "max_rounds": 1 if rag_mode == "decompose_once" else 3,
             "token_budget": 3000, "time_budget_s": 45.0, "max_sub_queries": 3,
             "analyzer_enabled": rag_mode != "baseline",
             "analyzer_model": "deepseek-flash" if rag_mode != "baseline" else "",
             "analyzer_temperature": 0,
             "analyzer_prompt_sha256": "g" * 8})


def test_contract_diff_across_three_modes_limited_to_rag_and_harness():
    base = _contract_for("baseline")
    for mode in ("decompose_once", "controlled_loop"):
        changed = [key for key in contract_diff(base, _contract_for(mode))]
        assert changed, mode
        assert all(key.startswith("rag.") or key.startswith("harness.")
                   for key in changed), (mode, changed)
    # 同组重算契约(除易变 run 字段外)diff 为空 → 指纹稳定
    assert contract_diff(_contract_for("baseline"), _contract_for("baseline")) == []
