"""知识路径:强制检索 → 置信度闸 → 弱证据兜底不进 Agent。"""
from types import SimpleNamespace

from app.graph.knowledge import make_knowledge_nodes
from tests.helpers import graph_state as _state


def _service(items=None, low=False, reason=""):
    """RetrievalService 替身:retrieve 返回 dataclass 样对象。"""
    def retrieve(query, strategy="hybrid_rerank", category=None, use_rewrite=True):
        assert strategy == "hybrid_rerank"  # 知识路径强制 hybrid_rerank
        return SimpleNamespace(items=items or [], low_confidence=low, reason=reason,
                               filter_fallback=False, degraded=False)
    return SimpleNamespace(retrieve=retrieve)


def _kb(chunk_id=7):
    row = SimpleNamespace(questions="退货政策是什么\n怎么退", answer="七天无理由退货。",
                          category="售后政策", section_path="d.md > 退货政策")
    return SimpleNamespace(get_chunks=lambda ids: [row for _ in ids])


class FakePool:
    def __init__(self):
        self.rows = []

    def insert(self, source, conversation_id, question, reason="",
               retrieved_chunks=None, matched_review_id=None):
        self.rows.append((source, conversation_id, question, reason, retrieved_chunks))


async def test_retrieve_attaches_evidence_and_trace():
    retrieve, _, _ = make_knowledge_nodes(
        _service(items=[SimpleNamespace(chunk_id=7, score=0.9)]), _kb(), FakePool())
    out = await retrieve(_state(resolved_message="退货政策是什么"))
    assert out["evidence"][0]["n"] == 1 and out["evidence"][0]["chunk_id"] == 7
    assert out["evidence"][0]["answer"] == "七天无理由退货。"
    assert any("node=retrieve" in t and "top1=0.90" in t for t in out["trace"])


async def test_retrieve_empty_marks_low_confidence():
    retrieve, _, _ = make_knowledge_nodes(_service(low=True, reason="知识库无相关内容"), _kb(), FakePool())
    out = await retrieve(_state(resolved_message="量子力学怎么退货"))
    assert out["evidence"] == [] and out["low_confidence"] is True


async def test_gate_weak_evidence_falls_back_and_pools():
    pool = FakePool()
    _, gate, fallback = make_knowledge_nodes(_service(low=True, reason="证据置信度低"), _kb(), pool)
    g = await gate(_state(session_id=1, user_message="量子力学怎么退货",
                          low_confidence=True, low_reason="证据置信度低"))
    assert g["gate_passed"] is False
    assert pool.rows[0][0] == "retrieval_low_conf" and pool.rows[0][2] == "量子力学怎么退货"

    out = await fallback(_state(user_message="量子力学怎么退货", **g))
    assert "无法回答" in out["final_reply"]
    # ch07:节点只吐本轮新增 assistant 消息(user 由 initial_state 带入,log 节点一并落库)
    assert [type(m).__name__ for m in out["messages"]] == ["AIMessage"]  # I1 修复延续:兜底轮 user 照常落库
    assert out["messages"][-1].content == out["final_reply"]


async def test_gate_strong_evidence_passes_no_pooling():
    pool = FakePool()
    retrieve, gate, _ = make_knowledge_nodes(
        _service(items=[SimpleNamespace(chunk_id=7, score=0.9)]), _kb(), pool)
    st = _state(resolved_message="退货政策是什么")
    st = {**st, **(await retrieve(st))}
    out = await gate(st)
    assert out["gate_passed"] is True
    assert pool.rows == []
