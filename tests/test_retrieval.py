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


class _SpyRewriter:
    def __init__(self):
        self.calls = 0

    def rewrite(self, query):
        from app.rewrite import QueryRewrite
        self.calls += 1
        return QueryRewrite(normalized="退货的规定是什么", synonyms=["退货"])


def test_prepared_query_shared_across_strategies(service):
    spy = _SpyRewriter()
    service._rewriter = spy
    prepared = service.prepare_query("东西坏了想退")
    for strategy in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        service.retrieve("东西坏了想退", strategy=strategy, prepared=prepared)
    assert spy.calls == 1


def test_prepare_query_no_rewrite_is_identity(service):
    spy = _SpyRewriter()
    service._rewriter = spy
    prepared = service.prepare_query("东西坏了想退", use_rewrite=False)
    assert prepared.rewritten == "东西坏了想退"
    assert prepared.search_text == "东西坏了想退"
    assert spy.calls == 0


def test_unknown_strategy_raises(service):
    with pytest.raises(ValueError):
        service.retrieve("q", strategy="magic")


def test_dense_floor_constant():
    assert RETRIEVAL_SCORE_FLOOR == 0.5


def test_rerank_failure_degrades_to_rrf():
    """rerank API 挂掉 → 降级 RRF 序,证据闸无法判定,fail closed(低置信)。"""
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
    assert r.items and r.low_confidence and r.degraded
    assert "证据闸无法判定" in r.reason


def test_category_injection_sanitized_and_falls_back(service):
    """评审修复:LLM 传来的 category 带引号/反斜杠 → 剥离;expr 仍炸则异常感知回退无过滤。"""
    r = service.retrieve("退货的规定是什么", strategy="hybrid_rerank",
                         category='x" or category != "x')
    assert r.items and r.filter_fallback is True  # 坏 expr 不炸穿,回退无过滤


def test_degraded_flag_on_missing_reranker():
    """评审修复:reranker 缺失降级 RRF 序时 degraded=True,证据质量降档可辨。"""
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
    svc = RetrievalService(FakeEmbedding(), vectors, kb, rewriter=None, reranker=None)
    r = svc.retrieve("SH-E300 降噪", strategy="hybrid_rerank")
    assert r.items and r.degraded is True and r.low_confidence is True


class _FakeJudge:
    def __init__(self, answered, reason="judge"):
        self.answered = answered
        self.reason = reason
        self.calls = []

    def judge(self, query, documents):
        from app.evidence import EvidenceVerdict
        self.calls.append((query, list(documents)))
        return EvidenceVerdict(answered=self.answered, reason=self.reason)


class _ExplodingJudge:
    def judge(self, query, documents):
        raise RuntimeError("judge down")


def _service_with_judge(service, judge):
    return RetrievalService(
        service._embedder, service._vectors, service._kb,
        rewriter=service._rewriter, reranker=service._reranker,
        evidence_judge=judge,
    )


def test_entailment_false_forces_low_confidence_even_if_score_passes(service):
    judge = _FakeJudge(False, "只写了蓝牙,没有充电方式")
    svc = _service_with_judge(service, judge)
    r = svc.retrieve("SH-E300 降噪", strategy="hybrid_rerank")

    assert r.low_confidence is True and r.evidence_entailment is False
    assert "证据未直接回答问题" in r.reason
    assert judge.calls and judge.calls[0][0] == "SH-E300 降噪"
    assert 1 <= len(judge.calls[0][1]) <= 3


def test_entailment_true_overrides_low_score(service):
    judge = _FakeJudge(True, "证据直接回答")
    svc = _service_with_judge(service, judge)
    r = svc.retrieve("量子力学", strategy="hybrid_rerank")

    assert r.evidence_entailment is True
    assert r.low_confidence is False


def test_entailment_failure_falls_back_to_score_gate(service):
    svc = _service_with_judge(service, _ExplodingJudge())
    r = svc.retrieve("SH-E300 降噪", strategy="hybrid_rerank")

    assert r.evidence_entailment is None
    assert r.low_confidence is False  # 该题分数闸可通过


def test_no_judge_leaves_entailment_none(service):
    r = service.retrieve("SH-E300 降噪", strategy="hybrid_rerank")
    assert r.evidence_entailment is None
