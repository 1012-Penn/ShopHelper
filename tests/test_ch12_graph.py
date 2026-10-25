"""ch12 图接入测试:retrieval_route 开关(默认零变化)、agentic_retrieve 节点行为、
审计逐轮落库与 initial_state 防残留。全部替身,不触网。"""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage

from app.chunking import Chunk
from app.config import Settings
from app.evidence import EvidenceVerdict
from app.graph.agentic import CLARIFY_PREFIX, make_agentic_retrieve_node
from app.graph.builder import build_graph, initial_state
from app.guard import REFUSAL_MARKER, is_refusal
from app.kb import KnowledgeBaseStore
from app.prompts import GapReport
from app.rag_steps import RagLoopStepStore
from app.retrieval import EvidenceCalibration, RetrievalResult, Retrieved
from app.rag_loop import STOP_CLARIFY, STOP_ESCALATE, STOP_ROUNDS
from tests.conftest import make_session_factory
from tests.helpers import (EchoResolver, GraphChatModel, StubExpand, builtin_engine)


class StubService:
    """只实现 agentic 节点消费的三个口:retrieve/_evidence_judge/_evidence_calibration。"""

    def __init__(self, script=None, default=None):
        self._script = script or {}
        self._default = default or RetrievalResult(
            [], True, "知识库无相关内容", "", "", "")
        self._evidence_calibration = EvidenceCalibration()
        self._evidence_judge = None
        self.calls: list[tuple[str, bool]] = []

    def retrieve(self, query, strategy="hybrid_rerank", category=None,
                 use_rewrite=True, prepared=None):
        self.calls.append((query, use_rewrite))
        action = self._script.get(query, self._default)
        if isinstance(action, Exception):
            raise action
        return action


class FakeGapAnalyzer:
    def __init__(self, reports):
        self.reports = list(reports)
        self.calls = 0

    def analyze(self, query, evidence):
        self.calls += 1
        return self.reports.pop(0)


class FrameSink:
    def __init__(self):
        self.frames = []

    def __call__(self, frame):
        self.frames.append(frame)

    def of(self, kind):
        return [f for f in self.frames if f["type"] == kind]


def seed_kb(factory) -> KnowledgeBaseStore:
    kb = KnowledgeBaseStore(factory)
    kb.replace_doc_chunks("测试.md", [
        Chunk(category="会员", questions="会员有什么优惠", answer="会员享9折",
              section_path="会员>优惠", content_type="policy", is_key_clause=False),
        Chunk(category="商品", questions="A商品多少钱", answer="A商品100元",
              section_path="商品>价格", content_type="faq", is_key_clause=False),
    ])
    return kb


def make_settings(**kw):
    base = dict(openai_api_key="sk-test", _env_file=None)
    base.update(kw)
    return Settings(**base)


QUERY = "会员买A商品多少钱"
HIGH = RetrievalResult([Retrieved(1, 0.9)], False, "", "", "", "",
                       evidence_entailment=True)
LOW = RetrievalResult([Retrieved(1, 0.2)], True, "证据未直接回答问题", "", "", "",
                      evidence_entailment=False)


async def run_node(node, state_over=None, sink=None):
    state = {"session_id": 1, "user_message": QUERY, "resolved_message": QUERY,
             "trace": [], "turn_user_msg_id": "utest1"}
    state.update(state_over or {})
    return await node(state, writer=sink, config=None)


@pytest.fixture
def factory():
    return make_session_factory()


def test_node_first_pass_sufficient_uses_agent_path(factory):
    from app.guard import LowConfidencePool

    kb = seed_kb(factory)
    service = StubService({QUERY: HIGH})
    node = make_agentic_retrieve_node(
        service, kb, LowConfidencePool(factory), make_settings(retrieval_route="controlled_loop"),
        max_rounds=3, analyzer=FakeGapAnalyzer([]))
    sink = FrameSink()
    out = asyncio.run(run_node(node, sink=sink))
    assert out["gate_passed"] is True and out["low_confidence"] is False
    assert out["evidence"][0]["n"] == 1 and out["evidence"][0]["answer"] == "会员享9折"
    assert out["rag_rounds"] == 0 and out["rag_stop_reason"] == "first_pass_sufficient"
    assert any("node=agentic_retrieve" in line for line in out["trace"])
    assert sink.of("citations") and sink.of("citations")[0]["items"][0]["n"] == 1


def test_node_insufficient_refuses_with_pool_and_actions(factory):
    from app.guard import LowConfidencePool
    from app.models import LowConfidenceQuestion

    kb = seed_kb(factory)
    pool = LowConfidencePool(factory)
    analyzer = FakeGapAnalyzer([GapReport(action="refuse", reason="超出知识范围")])
    node = make_agentic_retrieve_node(
        StubService({QUERY: LOW}), kb, pool,
        make_settings(retrieval_route="controlled_loop"), max_rounds=3, analyzer=analyzer)
    sink = FrameSink()
    out = asyncio.run(run_node(node, sink=sink))
    assert out["gate_passed"] is False and out["evidence"] == []
    assert is_refusal(out["final_reply"])
    assert out["suggested_actions"] == ["transfer_human", "create_ticket"]
    assert sink.of("token") and sink.of("token")[0]["content"] == out["final_reply"]
    with factory() as session:
        rows = session.query(LowConfidenceQuestion).all()
        assert len(rows) == 1 and rows[0].source == "retrieval_low_conf"
        assert rows[0].reason.startswith("agentic:refuse_no_evidence")


def test_node_clarify_reply_without_marker(factory):
    from app.guard import LowConfidencePool
    from app.models import LowConfidenceQuestion

    kb = seed_kb(factory)
    analyzer = FakeGapAnalyzer([GapReport(action="clarify", clarify_question="请问是哪款商品?")])
    node = make_agentic_retrieve_node(
        StubService({QUERY: LOW}), kb, LowConfidencePool(factory),
        make_settings(retrieval_route="decompose_once"), max_rounds=1, analyzer=analyzer)
    out = asyncio.run(run_node(node))
    assert out["rag_stop_reason"] == STOP_CLARIFY
    assert out["final_reply"].startswith(CLARIFY_PREFIX)
    assert not is_refusal(out["final_reply"])
    assert out["suggested_actions"] == []
    assert isinstance(out["messages"][0], AIMessage)
    with factory() as session:
        assert session.query(LowConfidenceQuestion).count() == 0  # 澄清不算检索失败


def test_node_escalate_mentions_conflict(factory):
    from app.guard import LowConfidencePool

    kb = seed_kb(factory)
    analyzer = FakeGapAnalyzer([GapReport(action="escalate", conflicts=["新旧政策口径不同"])])
    node = make_agentic_retrieve_node(
        StubService({QUERY: LOW}), kb, LowConfidencePool(factory),
        make_settings(retrieval_route="controlled_loop"), max_rounds=2, analyzer=analyzer)
    out = asyncio.run(run_node(node))
    assert out["rag_stop_reason"] == STOP_ESCALATE
    assert is_refusal(out["final_reply"]) and "冲突" in out["final_reply"]
    assert out["suggested_actions"] == ["transfer_human", "create_ticket"]


def test_steps_store_records_each_round_and_finalizes_stop(factory):
    from app.guard import LowConfidencePool
    from app.models import RagLoopStep

    kb = seed_kb(factory)
    store = RagLoopStepStore(factory)
    analyzer = FakeGapAnalyzer([GapReport(action="retrieve_more", sub_queries=["A商品多少钱"])])
    sub_high = RetrievalResult([Retrieved(2, 0.9)], False, "", "", "", "",
                               evidence_entailment=True)
    service = StubService({QUERY: LOW, "A商品多少钱": sub_high})
    # 蕴含判断器对合并证据仍说没回答 → 轮数耗尽拒答,但补检索轮已落审计
    judge = type("J", (), {"judge": lambda self, q, docs: EvidenceVerdict(
        answered=False, supported_by=[], reason="")})()
    service._evidence_judge = judge
    node = make_agentic_retrieve_node(
        service, kb, LowConfidencePool(factory),
        make_settings(retrieval_route="decompose_once"), max_rounds=1,
        analyzer=analyzer, steps_store=store)
    out = asyncio.run(run_node(node))
    assert out["rag_stop_reason"] == STOP_ROUNDS
    with factory() as session:
        rows = session.query(RagLoopStep).order_by(RagLoopStep.id).all()
        assert len(rows) == len(out["rag_steps"]) == 3  # round0 + round1 + final
        assert rows[-1].stop_reason == STOP_ROUNDS
        assert rows[1].sub_queries == ["A商品多少钱"]
        assert rows[0].turn_user_msg_id == "utest1"
        assert {r.phase for r in rows} == {"retrieve", "final"}


def test_builder_default_route_has_no_agentic_node(factory):
    from app.guard import LowConfidencePool
    from app.kb import KnowledgeBaseStore as KB

    kb = KB(factory)
    service = StubService()
    graph = build_graph(GraphChatModel(), builtin_engine(), make_settings(),
                        _StubStore(), LowConfidencePool(factory), service, kb,
                        resolver=EchoResolver({}), expander=StubExpand({}))
    assert "agentic_retrieve" not in graph.nodes


def test_builder_agentic_route_registers_node_and_redirect(factory):
    from app.guard import LowConfidencePool
    from app.kb import KnowledgeBaseStore as KB

    kb = KB(factory)
    service = StubService()
    settings = make_settings(retrieval_route="controlled_loop")
    graph = build_graph(GraphChatModel(), builtin_engine(), settings,
                        _StubStore(), LowConfidencePool(factory), service, kb,
                        resolver=EchoResolver({}), expander=StubExpand({}),
                        gap_analyzer=FakeGapAnalyzer([]))
    assert "agentic_retrieve" in graph.nodes


class _StubStore:
    async def append(self, session_id, rows):
        return None


def test_initial_state_resets_rag_fields_every_turn():
    first = initial_state(1, "第一问")
    assert first["rag_rounds"] == 0 and first["rag_stop_reason"] == "" and first["rag_steps"] == []
    # 第二回合即便 checkpointer 里残留上回合值,initial_state 全量覆盖(防泄漏纪律)
    second = initial_state(1, "第二问")
    assert second["rag_rounds"] == 0 and second["rag_stop_reason"] == ""


@pytest.mark.asyncio
async def test_full_graph_controlled_loop_end_to_end(factory):
    """商品咨询 → agentic_retrieve(首轮即充分) → agent → log,全图走线。"""
    from app.guard import LowConfidencePool
    from app.store import ConversationStore

    kb = seed_kb(factory)
    service = StubService({QUERY: HIGH})
    settings = make_settings(retrieval_route="controlled_loop")
    chat = GraphChatModel(intent_reply=json.dumps(
        {"intent": "商品咨询", "confidence": 0.99}))
    graph = build_graph(chat, builtin_engine(), settings,
                        ConversationStore(factory), LowConfidencePool(factory),
                        service, kb, resolver=EchoResolver({}), expander=StubExpand({}),
                        intent_model=chat, gap_analyzer=FakeGapAnalyzer([]))
    state = initial_state(1, QUERY)
    out = await graph.ainvoke(state, {"configurable": {"thread_id": "t1"}})
    assert out["gate_passed"] is True
    assert out["rag_stop_reason"] == "first_pass_sufficient"
    assert out["final_reply"] == "默认回答。"
    assert any("node=agentic_retrieve" in line for line in out["trace"])


@pytest.mark.asyncio
async def test_full_graph_agentic_refusal_short_circuits_to_log(factory):
    from app.guard import LowConfidencePool
    from app.store import ConversationStore

    kb = seed_kb(factory)
    service = StubService(default=LOW)
    settings = make_settings(retrieval_route="decompose_once")
    chat = GraphChatModel(intent_reply=json.dumps(
        {"intent": "商品咨询", "confidence": 0.99}))
    analyzer = FakeGapAnalyzer([GapReport(action="refuse", reason="超出知识范围")])
    graph = build_graph(chat, builtin_engine(), settings,
                        ConversationStore(factory), LowConfidencePool(factory),
                        service, kb, resolver=EchoResolver({}), expander=StubExpand({}),
                        intent_model=chat, gap_analyzer=analyzer)
    out = await graph.ainvoke(initial_state(1, QUERY), {"configurable": {"thread_id": "t2"}})
    assert out["gate_passed"] is False
    assert is_refusal(out["final_reply"])
    assert "node=agentic_retrieve" in out["trace"][-2] or any(
        "node=agentic_retrieve" in line for line in out["trace"])
    # 拒答收口在 agentic 节点内完成:agent 节点未跑(final_reply 不是模型默认回复)
    assert out["final_reply"] != "默认回答。"
