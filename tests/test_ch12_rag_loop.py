"""ch12 受控 Agentic 检索循环核心单测:界限、去重、故障收敛、停止原因与证据闸不放宽。

全部离线替身(retrieve/judge/analyzer 注入),不触网。对应设计稿 3.1/3.2 与验收标准 3。
"""
import pytest

from app.evidence import EvidenceVerdict
from app.prompts import GapReport
from app.rag_loop import (
    CONVERGE_STOPS,
    ControlledRagLoop,
    RagLoopLimits,
    STOP_ANALYZER_ERROR,
    STOP_CLARIFY,
    STOP_FIRST_PASS,
    STOP_NO_NEW,
    STOP_REFUSE,
    STOP_ROUNDS,
    STOP_SUFFICIENT,
    STOP_TIME,
    STOP_TOKENS,
    STOP_RETRIEVAL_ERROR,
    normalize_query,
)
from app.retrieval import EvidenceCalibration, RetrievalResult, Retrieved

QUERY = "会员买A商品能便宜多少"

TEXTS = {1: "会员9折", 2: "A售价100元", 3: "B证据", 4: "C证据", 5: "D证据", 6: "E证据"}


class BudgetExceeded(RuntimeError):
    """与 scripts/eval_cost.BudgetExceeded 同名:循环按类名识别并必须穿透。"""


class FakeJudge:
    def __init__(self, answers: list[bool]):
        self.answers = list(answers)
        self.calls = 0

    def judge(self, query, documents):
        self.calls += 1
        answered = self.answers.pop(0) if self.answers else False
        return EvidenceVerdict(answered=answered, supported_by=[1], reason="")


class FakeAnalyzer:
    def __init__(self, reports: list):
        self.reports = list(reports)
        self.calls = 0

    def analyze(self, query, evidence):
        self.calls += 1
        report = self.reports.pop(0)
        if isinstance(report, Exception):
            raise report
        return report


def result(items, low=False, entailment=None, reason="", degraded=False):
    return RetrievalResult(
        items=[Retrieved(cid, score) for cid, score in items],
        low_confidence=low, reason=reason, evidence_entailment=entailment,
        degraded=degraded)


def make_retrieve(script: dict, default=None):
    calls: list[tuple[str, bool]] = []

    def retrieve(query, *, use_rewrite, prepared=None):
        calls.append((query, use_rewrite))
        action = script.get(query, default)
        if isinstance(action, Exception):
            raise action
        if action is None:
            action = result([], low=True, reason="知识库无相关内容")
        return action

    return retrieve, calls


def make_loop(retrieve, *, judge=None, analyzer=None, texts=None, **limits):
    texts = texts or TEXTS
    return ControlledRagLoop(
        retrieve,
        lambda ids: {i: texts[i] for i in ids if i in texts},
        judge, analyzer, EvidenceCalibration(), RagLoopLimits(**limits))


def test_normalize_query_folds_whitespace_and_case():
    assert normalize_query("  A  b C\n") == "abc"
    assert normalize_query("会员 价格") == normalize_query("会员价格")


def test_first_pass_sufficient_no_extra_llm_calls():
    retrieve, calls = make_retrieve({
        QUERY: result([(7, 0.9)], low=False, entailment=True)})
    judge, analyzer = FakeJudge([]), FakeAnalyzer([])
    loop = make_loop(retrieve, judge=judge, analyzer=analyzer)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_FIRST_PASS
    assert out.gate_passed is True
    assert [(i.chunk_id, i.score) for i in out.items] == [(7, 0.9)]
    assert out.rounds == 0 and analyzer.calls == 0 and judge.calls == 0
    assert len(out.steps) == 1 and out.steps[0]["stop_reason"] == STOP_FIRST_PASS
    assert calls == [(QUERY, True)]  # 首轮照常改写


def test_sufficient_after_merge_decompose_once():
    retrieve, calls = make_retrieve({
        QUERY: result([(1, 0.2)], low=True, entailment=False, reason="证据未直接回答问题"),
        "A商品多少钱": result([(2, 0.9)], low=False, entailment=True),
    })
    judge = FakeJudge([True])
    analyzer = FakeAnalyzer([GapReport(action="retrieve_more",
                                       sub_queries=["A商品多少钱"], reason="缺价格")])
    loop = make_loop(retrieve, judge=judge, analyzer=analyzer, max_rounds=1)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_SUFFICIENT and out.gate_passed is True
    assert [i.chunk_id for i in out.items] == [2]
    assert out.rounds == 1 and analyzer.calls == 1 and judge.calls == 1
    assert (QUERY, True) in calls and ("A商品多少钱", False) in calls  # 子查询不改写
    assert [s["phase"] for s in out.steps] == ["retrieve", "retrieve", "final"]
    assert out.steps[-1]["judge_answered"] is True


def test_pass_inside_loop_when_merge_satisfies():
    retrieve, _ = make_retrieve({
        QUERY: result([(1, 0.2)], low=True, entailment=False),
        "sub1": result([(3, 0.8)], low=False, entailment=True),
    })
    judge = FakeJudge([True])
    analyzer = FakeAnalyzer([GapReport(action="retrieve_more", sub_queries=["sub1"])])
    loop = make_loop(retrieve, judge=judge, analyzer=analyzer, max_rounds=2)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_SUFFICIENT and out.gate_passed is True
    assert out.steps[-1]["phase"] == "retrieve" and out.steps[-1]["round_index"] == 2
    assert judge.calls == 1  # r=2 轮首蕴含再判即放行


def test_rounds_exhausted_judge_false_blocks_high_scores():
    retrieve, calls = make_retrieve({
        QUERY: result([(1, 0.2)], low=True, entailment=False),
        "sub1": result([(3, 0.9)], low=False, entailment=True),
        "sub2": result([(4, 0.85)], low=False, entailment=True),
    })
    judge = FakeJudge([False, False])  # 高分但蕴含判断器说没回答 → 不放行(闸未放宽)
    analyzer = FakeAnalyzer([
        GapReport(action="retrieve_more", sub_queries=["sub1"]),
        GapReport(action="retrieve_more", sub_queries=["sub2"]),
    ])
    loop = make_loop(retrieve, judge=judge, analyzer=analyzer, max_rounds=2)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_ROUNDS and out.gate_passed is False
    assert out.evidence_entailment is False
    assert [i.chunk_id for i in out.items] == [3, 4]
    assert out.analyzer_calls == 2 and judge.calls == 2
    assert [c[0] for c in calls] == [QUERY, "sub1", "sub2"]


def test_dedup_skips_repeated_and_original_queries():
    retrieve, calls = make_retrieve({
        QUERY: result([(1, 0.2)], low=True, entailment=False)})
    analyzer = FakeAnalyzer([GapReport(
        action="retrieve_more",
        sub_queries=[f" {QUERY} ", QUERY.upper(), ""])])  # 全部与原查询同键或为空
    loop = make_loop(retrieve, analyzer=analyzer, max_rounds=3)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_NO_NEW
    assert out.reason == "缺口分析未给出新的可执行子查询"
    assert analyzer.calls == 1 and len(calls) == 1  # 只执行了首轮,子查询全被去重
    assert out.gate_passed is False


def test_no_new_evidence_round_stops():
    retrieve, _ = make_retrieve({
        QUERY: result([(1, 0.2)], low=True, entailment=False),
        "sub1": result([(3, 0.4)], low=True, entailment=False, reason="低置信"),
    })
    analyzer = FakeAnalyzer([GapReport(action="retrieve_more", sub_queries=["sub1"])])
    loop = make_loop(retrieve, analyzer=analyzer, max_rounds=3)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_NO_NEW
    assert out.reason == "本轮补检索未产生新证据"
    assert out.gate_passed is False and out.items == []


def test_clarify_action_sets_question():
    retrieve, _ = make_retrieve({
        QUERY: result([(1, 0.2)], low=True, entailment=False)})
    analyzer = FakeAnalyzer([GapReport(action="clarify",
                                       clarify_question="请问是哪款商品?")])
    loop = make_loop(retrieve, analyzer=analyzer, max_rounds=3)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_CLARIFY
    assert out.clarify_question == "请问是哪款商品?" and out.gate_passed is False


def test_refuse_and_escalate_actions():
    retrieve, _ = make_retrieve({QUERY: result([(1, 0.2)], low=True, entailment=False)})
    loop = make_loop(retrieve, analyzer=FakeAnalyzer(
        [GapReport(action="refuse", reason="超出知识范围")]), max_rounds=3)
    assert loop.run(QUERY).stop_reason == STOP_REFUSE
    loop = make_loop(retrieve, analyzer=FakeAnalyzer(
        [GapReport(action="escalate", conflicts=["新旧政策口径不同"])]), max_rounds=3)
    out = loop.run(QUERY)
    assert out.stop_reason == "escalate_conflict" and out.conflicts == ["新旧政策口径不同"]


def test_analyzer_error_converges_cautiously():
    retrieve, _ = make_retrieve({QUERY: result([(1, 0.2)], low=True, entailment=False)})
    analyzer = FakeAnalyzer([RuntimeError("upstream down")])
    loop = make_loop(retrieve, analyzer=analyzer, max_rounds=3)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_ANALYZER_ERROR and out.gate_passed is False


def test_round0_retrieval_error():
    retrieve, _ = make_retrieve({QUERY: ValueError("向量库不可用")})
    loop = make_loop(retrieve, max_rounds=3)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_RETRIEVAL_ERROR
    assert out.items == [] and len(out.steps) == 1


def test_time_budget_stop():
    retrieve, _ = make_retrieve({QUERY: result([(1, 0.2)], low=True, entailment=False)})
    analyzer = FakeAnalyzer([GapReport(action="retrieve_more", sub_queries=["sub1"])])
    loop = make_loop(retrieve, analyzer=analyzer, max_rounds=3, time_budget_s=0.0)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_TIME and out.gate_passed is False


def test_token_budget_stop_before_analyzer():
    retrieve, _ = make_retrieve({QUERY: result([(1, 0.2)], low=True, entailment=False)})
    analyzer = FakeAnalyzer([GapReport(action="retrieve_more", sub_queries=["sub1"])])
    loop = make_loop(retrieve, analyzer=analyzer, max_rounds=3, token_budget=1)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_TOKENS
    assert analyzer.calls == 0  # 预算先于调用检查,不产生付费调用


def test_budget_exceeded_propagates_from_retrieve_and_analyzer():
    retrieve, _ = make_retrieve({QUERY: BudgetExceeded("累计预算超限")})
    with pytest.raises(BudgetExceeded):
        make_loop(retrieve, max_rounds=1).run(QUERY)

    retrieve2, _ = make_retrieve({QUERY: result([(1, 0.2)], low=True, entailment=False)})
    analyzer = FakeAnalyzer([BudgetExceeded("累计预算超限")])
    with pytest.raises(BudgetExceeded):
        make_loop(retrieve2, analyzer=analyzer, max_rounds=1).run(QUERY)


def test_step_callback_fires_per_round_with_stop_on_last():
    retrieve, _ = make_retrieve({
        QUERY: result([(1, 0.2)], low=True, entailment=False),
        "sub1": result([(3, 0.9)], low=False, entailment=True),
    })
    judge = FakeJudge([False])
    analyzer = FakeAnalyzer([GapReport(action="retrieve_more", sub_queries=["sub1"])])
    seen: list[dict] = []
    loop = ControlledRagLoop(
        retrieve, lambda ids: {i: TEXTS[i] for i in ids if i in TEXTS},
        judge, analyzer, EvidenceCalibration(), RagLoopLimits(max_rounds=1),
        on_step=seen.append)
    out = loop.run(QUERY)
    assert len(seen) == len(out.steps) == 3  # round0 + round1 + final
    assert seen[-1]["stop_reason"] == STOP_ROUNDS
    assert all(s["round_index"] >= 0 for s in seen)
    assert {s["phase"] for s in seen} == {"retrieve", "final"}


def test_step_callback_failure_does_not_break_loop():
    retrieve, _ = make_retrieve({
        QUERY: result([(1, 0.2)], low=True, entailment=False),
        "sub1": result([(3, 0.9)], low=False, entailment=True),
    })
    analyzer = FakeAnalyzer([GapReport(action="retrieve_more", sub_queries=["sub1"])])

    def boom(step):
        raise RuntimeError("audit store down")

    loop = ControlledRagLoop(
        retrieve, lambda ids: {i: TEXTS[i] for i in ids if i in TEXTS},
        FakeJudge([True]), analyzer, EvidenceCalibration(), RagLoopLimits(max_rounds=1),
        on_step=boom)
    out = loop.run(QUERY)
    assert out.stop_reason == STOP_SUFFICIENT and out.gate_passed is True


def test_low_confidence_candidates_only_in_snapshots():
    retrieve, _ = make_retrieve({
        QUERY: result([(1, 0.2), (6, 0.1)], low=True, entailment=False),
        "sub1": result([(3, 0.9)], low=False, entailment=True),
    })
    analyzer = FakeAnalyzer([GapReport(action="retrieve_more", sub_queries=["sub1"])])
    loop = make_loop(retrieve, judge=FakeJudge([True]), analyzer=analyzer, max_rounds=1)
    out = loop.run(QUERY)
    assert out.gate_passed is True
    snapshot_ids = {s["chunk_id"] for s in out.snapshots}
    assert {1, 6, 3} <= snapshot_ids  # 低置信候选保留诊断快照
    assert [i.chunk_id for i in out.items] == [3]  # 但不进证据


def test_converge_stops_never_pass_gate():
    assert STOP_FIRST_PASS not in CONVERGE_STOPS
    assert STOP_SUFFICIENT not in CONVERGE_STOPS
    for stop in (STOP_ROUNDS, STOP_TIME, STOP_TOKENS, STOP_NO_NEW,
                 STOP_RETRIEVAL_ERROR, STOP_ANALYZER_ERROR):
        assert stop in CONVERGE_STOPS
