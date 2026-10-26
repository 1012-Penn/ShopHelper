from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.chunking import Chunk
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.models import Base
from app.retrieval import (
    EvidenceScope,
    PolicyWindow,
    RetrievalService,
    ScopedDocument,
    filter_scoped_documents,
)
from tests.helpers import FakeEmbedding, FakeVectorStore

WINDOWS = (
    PolicyWindow("v1", date(2024, 1, 1), date(2026, 5, 31)),
    PolicyWindow("v2", date(2026, 6, 1), None),
)


def _doc(chunk_id, text, section_path):
    return ScopedDocument(chunk_id=chunk_id, text=text, section_path=section_path)


def test_explicit_other_corpus_order_is_excluded_but_policy_is_kept():
    docs = [
        _doc(1, "ORD-1001 当前未支付，商品金额 248 元。", "订单与支付 > ORD-1001"),
        _doc(2, "退货需保持商品完好。", "退货政策 > 七天无理由退货"),
    ]

    result = filter_scoped_documents(
        docs,
        EvidenceScope(subject="corpus:ORD-1002"),
        policy_windows=WINDOWS,
    )

    assert [item.chunk_id for item in result.kept] == [2]
    assert result.excluded[0].chunk_id == 1
    assert result.excluded[0].reason == "entity_conflict"
    assert result.status == "partial"


def test_same_suffix_does_not_match_legacy_subject_to_corpus_chunk():
    docs = [_doc(1, "ORD-1001 当前未支付。", "订单与支付 > ORD-1001")]
    result = filter_scoped_documents(
        docs,
        EvidenceScope(subject="legacy_demo:1001"),
        policy_windows=WINDOWS,
    )

    assert result.kept == []
    assert result.excluded[0].reason == "entity_conflict"
    assert result.status == "empty"


def test_version_specific_chunk_is_excluded_only_on_explicit_conflict():
    docs = [
        _doc(1, "签收后 7 天内支持无理由退货。", "政策版本说明 > v1 版七天无理由"),
        _doc(2, "签收后 15 天内支持无理由退货。", "政策版本说明 > v2 版十五天无理由"),
        _doc(3, "商品需保持完好。", "退货政策 > 七天无理由退货"),
        _doc(4, "v1 与 v2 的差异如下。", "政策版本说明 > 版本时间线"),
    ]

    result = filter_scoped_documents(
        docs,
        EvidenceScope(policy_version="v2"),
        policy_windows=WINDOWS,
    )

    assert [item.chunk_id for item in result.kept] == [2, 3, 4]
    assert result.excluded[0].chunk_id == 1
    assert result.excluded[0].reason == "policy_version_conflict"
    assert result.unknown_ids == [3, 4]


def test_reference_date_uses_explicit_policy_window_boundaries():
    docs = [
        _doc(1, "签收后 7 天内支持无理由退货。", "政策版本说明 > v1 版七天无理由"),
        _doc(2, "签收后 15 天内支持无理由退货。", "政策版本说明 > v2 版十五天无理由"),
    ]

    before = filter_scoped_documents(
        docs,
        EvidenceScope(reference_date=date(2026, 5, 31)),
        policy_windows=WINDOWS,
    )
    boundary = filter_scoped_documents(
        docs,
        EvidenceScope(reference_date=date(2026, 6, 1)),
        policy_windows=WINDOWS,
    )

    assert [item.chunk_id for item in before.kept] == [1]
    assert [item.chunk_id for item in boundary.kept] == [2]
    assert before.matched_ids == [1]
    assert boundary.matched_ids == [2]


def test_missing_scope_is_unknown_and_never_filters_generic_policy():
    docs = [_doc(1, "商品需保持完好。", "退货政策 > 七天无理由退货")]

    result = filter_scoped_documents(
        docs,
        EvidenceScope(),
        policy_windows=WINDOWS,
    )

    assert [item.chunk_id for item in result.kept] == [1]
    assert result.unknown_ids == [1]
    assert result.status == "unknown"


def test_retrieval_scope_exclusion_never_falls_back_to_conflicting_evidence():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    kb = KnowledgeBaseStore(factory)
    kb.replace_doc_chunks(
        "订单与支付.md",
        [
            Chunk(
                category="订单",
                questions="ORD-1001 退货 当前状态",
                answer="ORD-1001 当前未支付。",
                section_path="订单与支付 > ORD-1001 未支付订单",
                content_type="manual",
                is_key_clause=False,
            )
        ],
    )
    vectors = FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    service = RetrievalService(FakeEmbedding(), vectors, kb, reranker=None)

    result = service.retrieve(
        "ORD-1001 退货",
        strategy="bm25",
        scope=EvidenceScope(subject="corpus:ORD-1002"),
    )

    assert result.items == []
    assert result.filter_fallback is False
    assert result.scope_status == "empty"
    assert result.scope_excluded[0]["chunk_id"] == 1
    assert result.reason == "作用域过滤后无适用证据"
