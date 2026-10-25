"""嵌入/重排/回答缓存:命中、顺序与损坏处理。"""
from __future__ import annotations

import json

import pytest

from scripts.eval_cache import (
    CacheError,
    EmbeddingCache,
    JsonCache,
    RerankCache,
    ResponseCache,
)


class _Embedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


class _Reranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, documents, top_n=None):
        self.calls += 1
        return [(index, float(len(doc))) for index, doc in enumerate(documents)]


def test_embedding_cache_hits_once_and_preserves_order(tmp_path):
    inner = _Embedder()
    cache = EmbeddingCache(inner, tmp_path / "embedding.json", "m")
    first = cache.embed(["a", "bb", "ccc"])
    assert first == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    second = cache.embed(["a", "bb", "ccc"])
    assert second == first
    assert inner.calls == 1
    assert cache.stats()["hits"] == 3


def test_embedding_cache_detects_model_mismatch(tmp_path):
    path = tmp_path / "embedding.json"
    EmbeddingCache(_Embedder(), path, "m1").embed(["a"])
    with pytest.raises(CacheError):
        EmbeddingCache(_Embedder(), path, "m2")


def test_rerank_cache_key_includes_documents(tmp_path):
    inner = _Reranker()
    cache = RerankCache(inner, tmp_path / "rerank.json", "m")
    first = cache.rerank("q", ["a", "bb"])
    second = cache.rerank("q", ["a", "bb"])
    third = cache.rerank("q", ["a", "bbb"])
    assert first == second
    assert third != second
    assert inner.calls == 2


def test_response_cache_key_changes_with_model_or_prompt(tmp_path):
    cache = ResponseCache(tmp_path / "responses.json", "responses")
    base = dict(prompt_sha256="p", temperature=0.0, query="q", evidence_ids=[1])
    key = cache.response_key(model="m1", **base)
    assert key != cache.response_key(model="m2", **base)
    assert key != cache.response_key(model="m1", **{**base, "prompt_sha256": "p2"})


def test_corrupt_cache_raises_with_guidance(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CacheError) as excinfo:
        JsonCache(path, "ns")
    assert "删除后重跑" in str(excinfo.value)
