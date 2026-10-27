"""Static embedding/rerank cache: precision, bounds, concurrency, and gate parity."""
from __future__ import annotations

import threading
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.chunking import Chunk
from app.evidence import EvidenceVerdict
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.models import Base
from app.retrieval import RetrievalService
from app.retrieval_cache import CachedEmbedder, CachedReranker
from tests.helpers import FakeEmbedding, FakeReranker, FakeRewriter, FakeVectorStore


class _Embedder:
    def __init__(self, delay: float = 0.0):
        self.calls = 0
        self.items = 0
        self.delay = delay
        self.lock = threading.Lock()

    def embed(self, texts):
        with self.lock:
            self.calls += 1
            self.items += len(texts)
        if self.delay:
            time.sleep(self.delay)
        return [[float(len(text)), 1.0] for text in texts]


class _Reranker:
    def __init__(self):
        self.calls = 0
        self.items = 0

    def rerank(self, query, documents, top_n=None):
        self.calls += 1
        self.items += len(documents)
        rows = [(index, float(len(document))) for index, document in enumerate(documents)]
        rows.sort(key=lambda item: -item[1])
        return rows[:top_n] if top_n is not None else rows


def _cache(
    inner,
    *,
    model="embedding-model",
    corpus_version="corpus-v1",
    config_version="config-v1",
    max_entries=8,
    ttl_seconds=60.0,
    clock=lambda: 0.0,
):
    return CachedEmbedder(
        inner,
        model=model,
        corpus_version=corpus_version,
        config_version=config_version,
        max_entries=max_entries,
        ttl_seconds=ttl_seconds,
        clock=clock,
    )


def test_embedding_cache_reuses_exact_input_and_reports_metrics():
    inner = _Embedder()
    cache = _cache(inner)

    first = cache.embed(["a", "bb"])
    second = cache.embed(["a", "bb"])

    assert first == [[1.0, 1.0], [2.0, 1.0]]
    assert second == first
    assert inner.calls == 1
    assert cache.stats()["hits"] == 2
    assert cache.stats()["misses"] == 2
    assert cache.stats()["provider_calls"] == 1
    assert cache.stats()["hit_rate"] == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [("model", "other-model"), ("corpus_version", "corpus-v2"),
     ("config_version", "config-v2")],
)
def test_embedding_key_changes_when_model_or_version_changes(field, value):
    inner = _Embedder()
    first = _cache(inner)
    second = _cache(inner, **{field: value})

    assert first.embed(["same"]) == second.embed(["same"])
    assert inner.calls == 2


def test_embedding_expiry_is_ttl_based():
    now = [0.0]
    inner = _Embedder()
    cache = _cache(inner, ttl_seconds=10.0, clock=lambda: now[0])

    cache.embed(["a"])
    cache.embed(["a"])
    now[0] = 10.01
    cache.embed(["a"])

    assert inner.calls == 2
    assert cache.stats()["expired"] == 1
    assert cache.stats()["entries"] == 1


def test_embedding_cache_is_lru_bounded():
    inner = _Embedder()
    cache = _cache(inner, max_entries=2)

    cache.embed(["a", "b"])
    cache.embed(["a"])  # a is now more recent than b
    cache.embed(["c"])
    cache.embed(["a"])
    cache.embed(["b"])

    stats = cache.stats()
    assert stats["capacity"] == 2
    assert stats["entries"] == 2
    assert stats["evictions"] >= 1
    assert inner.calls == 3


def test_concurrent_embedding_misses_coalesce_to_one_provider_batch():
    inner = _Embedder(delay=0.15)
    cache = _cache(inner)
    barrier = threading.Barrier(8)
    results = []
    errors = []

    def worker():
        try:
            barrier.wait(timeout=2)
            results.append(cache.embed(["same"]))
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not errors
    assert len(results) == 8
    assert all(result == [[4.0, 1.0]] for result in results)
    assert inner.calls == 1
    assert cache.stats()["coalesced"] == 7


def test_failed_embedding_is_not_cached_and_can_retry():
    class Flaky:
        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("upstream unavailable")
            return [[1.0] for _ in texts]

    inner = Flaky()
    cache = _cache(inner)
    with pytest.raises(RuntimeError, match="unavailable"):
        cache.embed(["a"])

    assert cache.embed(["a"]) == [[1.0]]
    assert inner.calls == 2
    assert cache.stats()["entries"] == 1


def test_rerank_cache_uses_complete_request_and_returns_defensive_copies():
    inner = _Reranker()
    cache = CachedReranker(
        inner,
        model="rerank-model",
        corpus_version="corpus-v1",
        config_version="config-v1",
        max_entries=8,
        ttl_seconds=60.0,
    )

    first = cache.rerank("q", ["a", "bb"], top_n=2)
    first.append((99, 0.0))
    second = cache.rerank("q", ["a", "bb"], top_n=2)

    assert second == [(1, 2.0), (0, 1.0)]
    assert inner.calls == 1
    assert cache.stats()["hits"] == 1


def test_rerank_key_changes_for_document_order_or_top_n():
    inner = _Reranker()
    cache = CachedReranker(
        inner,
        model="rerank-model",
        corpus_version="corpus-v1",
        config_version="config-v1",
    )

    cache.rerank("q", ["a", "bb"], top_n=2)
    cache.rerank("q", ["bb", "a"], top_n=2)
    cache.rerank("q", ["a", "bb"], top_n=1)

    assert inner.calls == 3
    assert cache.stats()["hits"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("model", "other-reranker"), ("corpus_version", "corpus-v2"),
     ("config_version", "config-v2")],
)
def test_rerank_key_changes_when_model_or_version_changes(field, value):
    inner = _Reranker()
    kwargs = {
        "model": "rerank-model",
        "corpus_version": "corpus-v1",
        "config_version": "config-v1",
    }
    first = CachedReranker(inner, **kwargs)
    second = CachedReranker(inner, **{**kwargs, field: value})

    assert first.rerank("q", ["a"]) == second.rerank("q", ["a"])
    assert inner.calls == 2


def test_retrieval_gate_output_is_identical_on_cache_hit():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    kb = KnowledgeBaseStore(factory)
    kb.replace_doc_chunks(
        "test.md",
        [
            Chunk(
                category="售后政策",
                questions="退货规定",
                answer="支持七天无理由退货,商品需保持完好。",
                section_path="test.md > 售后政策",
                content_type="faq",
                is_key_clause=False,
            ),
            Chunk(
                category="数码配件",
                questions="SH-E300 降噪耳机",
                answer="SH-E300 售价 299 元,支持主动降噪。",
                section_path="test.md > 商品",
                content_type="faq",
                is_key_clause=False,
            ),
        ],
    )
    vectors = FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    embedder = CachedEmbedder(
        FakeEmbedding(),
        model="BAAI/bge-m3",
        corpus_version="corpus-test",
        config_version="config-test",
    )
    reranker = CachedReranker(
        FakeReranker(),
        model="BAAI/bge-reranker-v2-m3",
        corpus_version="corpus-test",
        config_version="config-test",
    )

    class Judge:
        def __init__(self):
            self.calls = 0

        def judge(self, query, documents):
            self.calls += 1
            return EvidenceVerdict(answered=True, reason="supported")

    judge = Judge()
    service = RetrievalService(
        embedder,
        vectors,
        kb,
        rewriter=FakeRewriter(),
        reranker=reranker,
        candidates=50,
        final_top_k=3,
        evidence_judge=judge,
    )

    cold = service.retrieve("SH-E300 降噪耳机", strategy="hybrid_rerank")
    warm = service.retrieve("SH-E300 降噪耳机", strategy="hybrid_rerank")

    assert [item.chunk_id for item in warm.items] == [item.chunk_id for item in cold.items]
    assert [item.score for item in warm.items] == [item.score for item in cold.items]
    assert warm.low_confidence is cold.low_confidence
    assert warm.reason == cold.reason
    assert warm.evidence_confidence == cold.evidence_confidence
    assert warm.evidence_entailment == cold.evidence_entailment
    assert warm.degraded == cold.degraded
    assert embedder.stats()["hits"] == 1
    assert reranker.stats()["hits"] == 1
    assert judge.calls == 2
