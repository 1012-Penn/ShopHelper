"""检索策略模块:四策略同一真库(milvus-lite tmp),反漏斗纯函数,低置信与过滤回退。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import pytest

from app.chunking import Chunk
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.models import Base
from app.retrieval import RETRIEVAL_SCORE_FLOOR, RetrievalService, lost_in_middle_order
from tests.helpers import FakeEmbedding, FakeReranker, FakeRewriter, FakeVectorStore

DOCS = [
    # (category, question, answer)
    ("售后政策", "退货的规定是什么", "支持七天无理由退货,商品需保持完好。"),
    ("数码配件", "SH-E300 无线降噪耳机", "SH-E300 头戴式无线降噪耳机,售价 299 元,支持主动降噪。"),
    ("物流", "多久能发货", "现货商品 48 小时内发出。"),
]


@pytest.fixture
def service():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    kb = KnowledgeBaseStore(factory)
    kb.replace_doc_chunks("t.md", [
        Chunk(category=cat, questions=q, answer=a, section_path=f"t.md > {q}",
              content_type="faq", is_key_clause=False)
        for cat, q, a in DOCS
    ])
    vectors = FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    return RetrievalService(FakeEmbedding(), vectors, kb,
                            rewriter=FakeRewriter({"东西坏了想退": ("质量问题怎么退货", ["质量问题退货"])}),
                            reranker=FakeReranker())


def test_lost_in_middle_order():
    assert lost_in_middle_order(list(range(1, 11))) == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    assert lost_in_middle_order([]) == []
    assert lost_in_middle_order([7]) == [7]


def test_dense_hits_semantic_but_misses_model(service):
    r = service.retrieve("退货的规定是什么", strategy="dense")
    assert r.items[0].chunk_id == 1 and not r.low_confidence
    # 型号文本对 FakeEmbedding 是零词向量 → 相似度低于 floor 全被滤掉
    r2 = service.retrieve("SH-E300 怎么样", strategy="dense")
    assert all(it.chunk_id != 2 for it in r2.items)


def test_bm25_hits_model_number(service):
    r = service.retrieve("SH-E300", strategy="bm25")
    assert r.items[0].chunk_id == 2 and not r.low_confidence


def test_hybrid_fuses(service):
    r = service.retrieve("SH-E300 降噪", strategy="hybrid")
    assert r.items[0].chunk_id == 2


def test_hybrid_rerank_ranks_model_first(service):
    r = service.retrieve("SH-E300 降噪耳机", strategy="hybrid_rerank")
    assert r.items[0].chunk_id == 2
    assert len(r.items) <= 10


def test_rewrite_applies(service):
    r = service.retrieve("东西坏了想退", strategy="bm25", use_rewrite=True)
    assert r.rewritten == "质量问题怎么退货"
    assert "质量问题退货" in r.search_text


def test_low_confidence_on_no_evidence(service):
    r = service.retrieve("量子力学", strategy="hybrid_rerank")
    assert r.low_confidence and r.reason


def test_category_filter_fallback(service):
    r = service.retrieve("退货的规定是什么", strategy="hybrid_rerank", category="不存在的品类")
    assert r.filter_fallback is True and r.items  # 过滤空 → 回退无过滤


def test_category_filter_honored(service):
    r = service.retrieve("退货的规定是什么", strategy="hybrid_rerank", category="售后政策")
    assert r.filter_fallback is False and all(it.chunk_id == 1 for it in r.items)


def test_unknown_strategy_raises(service):
    with pytest.raises(ValueError):
        service.retrieve("q", strategy="magic")


def test_dense_floor_constant():
    assert RETRIEVAL_SCORE_FLOOR == 0.5


def test_rerank_failure_degrades_to_rrf():
    """rerank API 挂掉 → 降级 RRF 序(有结果、不低置信、不抛错),spec §4 降级语义。"""
    class ExplodingReranker:
        def rerank(self, query, documents, top_n=None):
            raise RuntimeError("rerank api down")

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    kb = KnowledgeBaseStore(factory)
    kb.replace_doc_chunks("t.md", [
        Chunk(category=cat, questions=q, answer=a, section_path=f"t.md > {q}",
              content_type="faq", is_key_clause=False)
        for cat, q, a in DOCS
    ])
    vectors = FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    svc = RetrievalService(FakeEmbedding(), vectors, kb, rewriter=None,
                           reranker=ExplodingReranker())
    r = svc.retrieve("SH-E300 降噪", strategy="hybrid_rerank")
    assert r.items and not r.low_confidence  # 降级仍有候选
