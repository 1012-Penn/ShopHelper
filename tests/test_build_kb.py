from pathlib import Path

from app.kb import KnowledgeBaseStore
from scripts.build_kb import build
from tests.helpers import FakeEmbedding, FakeVectorStore

DOCS_DIR = Path(__file__).resolve().parent.parent / "knowledge"


def _make(db_session_factory):
    kb = KnowledgeBaseStore(db_session_factory)
    return kb, FakeVectorStore(), FakeEmbedding()


def test_build_full_and_rerun_idempotent(db_session_factory):
    kb, vectors, embedder = _make(db_session_factory)
    stats = build(DOCS_DIR, kb, vectors, embedder)
    assert stats["docs"] == 3
    assert stats["chunks"] > 10
    assert stats["vectorized"] == stats["chunks"]
    assert vectors.count() == stats["chunks"]
    total = stats["chunks"]
    stats2 = build(DOCS_DIR, kb, vectors, embedder)  # 幂等重建
    assert stats2["chunks"] == total
    assert vectors.count() == total  # 旧向量删干净,无重复


def test_build_interrupt_then_resume(db_session_factory):
    kb, vectors, embedder = _make(db_session_factory)
    stats = build(DOCS_DIR, kb, vectors, embedder, max_chunks=5)  # 模拟中断
    assert stats["vectorized"] == 5
    assert len(kb.pending_chunks()) == stats["chunks"] - 5
    stats2 = build(DOCS_DIR, kb, vectors, embedder, rebuild_docs=False)  # 续跑:只补 pending
    assert stats2["vectorized"] == stats["chunks"] - 5
    assert kb.pending_chunks() == []
    assert vectors.count() == stats["chunks"]


def test_build_max_chars_controls_granularity(db_session_factory):
    _, _, embedder = _make(db_session_factory)
    kb_small, v_small = KnowledgeBaseStore(db_session_factory), FakeVectorStore()
    kb_big, v_big = KnowledgeBaseStore(db_session_factory), FakeVectorStore()
    small = build(DOCS_DIR, kb_small, v_small, embedder, max_chars=100, overlap_chars=20)
    big = build(DOCS_DIR, kb_big, v_big, embedder, max_chars=500, overlap_chars=80)
    assert small["chunks"] > big["chunks"]  # 块越小切得越多
