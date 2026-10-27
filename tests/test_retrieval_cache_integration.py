"""Factory contracts for the optional retrieval static-call cache."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.retrieval_cache import CachedEmbedder, CachedReranker
from tests.conftest import make_session_factory
from tests.helpers import FakeEmbedding, FakeReranker, FakeVectorStore


def _settings(**overrides):
    return Settings(
        openai_api_key="sk-test",
        embedding_api_key="sk-test",
        _env_file=None,
        query_rewrite_enabled=False,
        evidence_entailment_enabled=False,
        **overrides,
    )


def _build(monkeypatch, settings):
    import app.embedding as embedding_module
    import app.rerank as rerank_module
    import app.vector_store as vector_module
    from app.graph.builder import make_retrieval_chain

    monkeypatch.setattr(embedding_module, "make_embedder", lambda _settings: FakeEmbedding())
    monkeypatch.setattr(rerank_module, "make_reranker", lambda _settings: FakeReranker())
    monkeypatch.setattr(
        vector_module,
        "KnowledgeVectorStore",
        lambda *_args, **_kwargs: FakeVectorStore(),
    )
    return make_retrieval_chain(make_session_factory(), settings)


def test_cache_is_default_off(monkeypatch):
    service, _ = _build(monkeypatch, _settings())

    assert service._embedder.__class__ is FakeEmbedding
    assert service._reranker.__class__ is FakeReranker


def test_enabled_cache_requires_explicit_corpus_version(monkeypatch):
    with pytest.raises(ValueError, match="retrieval_cache_corpus_version"):
        _build(monkeypatch, _settings(retrieval_cache_enabled=True))


def test_enabled_cache_wraps_static_models_with_derived_config_version(monkeypatch):
    service, _ = _build(
        monkeypatch,
        _settings(
            retrieval_cache_enabled=True,
            retrieval_cache_corpus_version="corpus-build-2026-10-27",
        ),
    )

    assert isinstance(service._embedder, CachedEmbedder)
    assert isinstance(service._reranker, CachedReranker)
    assert service._embedder.stats()["corpus_version"] == "corpus-build-2026-10-27"
    assert service._reranker.stats()["config_version"] == (
        service._embedder.stats()["config_version"]
    )
