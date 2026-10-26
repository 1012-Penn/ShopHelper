import json
from types import SimpleNamespace

from app.orders import (
    ORDER_CATALOG,
    extract_order_id,
    extract_order_reference,
    get_order,
    list_orders,
)


def test_legacy_bare_id_keeps_old_contract_and_declares_namespace():
    order = get_order("1001")

    assert order["product"] == "无线耳机"
    assert order["amount"] == 299.0
    assert order["status"] == "已签收"
    assert order["subject"]["canonical_id"] == "legacy_demo:1001"
    assert order["subject"]["namespace"] == "legacy_demo"
    assert order["source"] == "legacy_demo:ORDER_CATALOG"
    assert order["version"] == "legacy-demo-v1"
    assert order["as_of"] == "unknown"


def test_complete_corpus_id_never_collapses_into_same_suffix_legacy_order():
    legacy = get_order("1001")
    corpus = get_order("ORD-1001")

    assert corpus["order_id"] == "ORD-1001"
    assert corpus["amount"] == 248
    assert corpus["status"] == "未支付"
    assert corpus["subject"]["canonical_id"] == "corpus:ORD-1001"
    assert corpus["subject"]["namespace"] == "corpus"
    assert corpus["source"] == "corpus:SRC-ORDERS"
    assert corpus["as_of"] == "2026-10-24"
    assert corpus["version"].startswith("corpus-orders-v1+")
    assert corpus["subject"] != legacy["subject"]


def test_corpus_order_exposes_missing_fields_instead_of_inventing_them():
    order = get_order("ORD-1001")

    assert "product" in order["missing_fields"]
    assert order["signed_at"] is None
    assert "policy_scope" not in order


def test_signed_order_derives_policy_version_from_traceable_fact():
    order = get_order("ORD-1004")

    assert order["signed_at"] == "2026-06-08 11:30"
    assert "policy_scope" not in order


def test_unknown_namespaced_id_does_not_fall_back_to_legacy_catalog():
    result = get_order("ORD-9999")

    assert result["error"] == "未找到该订单"
    assert result["subject"]["canonical_id"] == "corpus:ORD-9999"
    assert "product" not in result


def test_extract_order_id_supports_full_corpus_id_and_old_bare_contract():
    assert extract_order_id("订单ORD-1001现在什么状态") == "ORD-1001"
    assert extract_order_id("订单号 ord-1004 的签收时间") == "ORD-1004"
    assert extract_order_id("1001能退吗") == "1001"


def test_suffix_phrase_is_ambiguous_and_returns_no_guessed_id():
    reference = extract_order_reference("尾号1001的那单怎么退款")

    assert reference.ambiguous is True
    assert reference.namespace is None
    assert reference.order_id is None
    assert extract_order_id("尾号1001的那单怎么退款") is None


def test_date_year_is_not_misread_as_bare_order_id():
    reference = extract_order_reference("假设订单在2026-05-31签收，退货窗口几天？")

    assert reference.order_id is None
    assert reference.namespace is None


def test_list_orders_defaults_to_legacy_and_explicit_namespaces_are_stable():
    legacy = list_orders()
    corpus = list_orders(namespace="corpus")
    combined = list_orders(namespace="all")

    assert len(ORDER_CATALOG) == 3
    assert [row["order_id"] for row in legacy] == ["1001", "1002", "1003"]
    assert len(corpus) == 11
    assert all(row["subject"]["namespace"] == "corpus" for row in corpus)
    assert [row["subject"]["canonical_id"] for row in combined[:3]] == [
        "legacy_demo:1001",
        "legacy_demo:1002",
        "legacy_demo:1003",
    ]


def test_query_order_tool_preserves_legacy_and_accepts_corpus_id():
    from app.tools.builtin import build_default_registry
    from tests.conftest import make_session_factory

    registry = build_default_registry(
        make_session_factory(), embedder=object(), vectors=object()
    )
    tools = {record.name: record.tool for record in registry.records()}

    legacy = json.loads(tools["query_order"].invoke({"order_id": "1001"}))
    corpus = json.loads(tools["query_order"].invoke({"order_id": "ORD-1001"}))

    assert legacy["amount"] == 299.0
    assert corpus["amount"] == 248
    assert legacy["subject"]["canonical_id"] != corpus["subject"]["canonical_id"]


def test_query_faq_passes_order_derived_policy_scope_to_retrieval():
    from app.tools.base import ToolContext, ToolRegistryV2
    from app.tools.builtin import register_builtin

    class Service:
        seen_scope = None

        def retrieve(self, keyword, strategy, category=None, scope=None):
            self.seen_scope = scope
            return SimpleNamespace(
                items=[],
                low_confidence=True,
                reason="empty",
                filter_fallback=False,
                degraded=False,
                snapshot=list,
                scope_status=scope_value,
                scope_excluded=[],
                scope_unknown_ids=[],
                scope_matched_ids=[],
            )

    scope_value = "verified"
    service = Service()
    registry = ToolRegistryV2()
    register_builtin(
        registry,
        ToolContext(
            session_factory=None,
            service=service,
            kb=SimpleNamespace(get_chunks=lambda _ids: []),
            top_k=3,
        ),
    )
    tool = {record.name: record.tool for record in registry.records()}["query_faq"]
    payload = json.loads(
        tool.invoke(
            {
                "keyword": "退货窗口",
                "order_id": "ORD-1004",
                "as_of_date": "2026-05-01",
            }
        )
    )

    assert service.seen_scope.subject == "corpus:ORD-1004"
    assert service.seen_scope.policy_version == "v2"
    assert service.seen_scope.reference_date.isoformat() == "2026-06-08"
    assert payload["scope_filter"]["reference_date"] == "2026-06-08"
