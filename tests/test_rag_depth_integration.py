"""End-to-end observability contracts for the optional deeper retrieval path."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from app.graph.knowledge import make_knowledge_nodes
from app.retrieval import RetrievalResult, Retrieved
from app.tools.base import ToolContext, ToolRegistryV2
from app.tools.builtin import register_builtin


class _Service:
    def retrieve(self, *_args, **_kwargs):
        return RetrievalResult(
            items=[Retrieved(7, 0.91)],
            rewritten="订单取消规则",
            search_text="订单取消规则",
            selection_facets=["order_summary", "cancel_rule"],
            rerank_candidates=[
                Retrieved(8, 0.95),
                Retrieved(7, 0.91),
            ],
            selection_latency_ms=0.37,
        )


class _Pool:
    def insert(self, *_args, **_kwargs):
        return None


def _kb():
    row = SimpleNamespace(
        id=7,
        category="订单",
        questions="订单取消规则",
        answer="未发货订单可以取消。",
        section_path="订单与支付 > 取消",
    )
    return SimpleNamespace(get_chunks=lambda _ids: [row])


def test_knowledge_node_exposes_evidence_selection_trace():
    node, _, _ = make_knowledge_nodes(_Service(), _kb(), _Pool())

    result = asyncio.run(node({
        "resolved_message": "订单取消规则",
        "user_message": "订单取消规则",
        "session_id": 1,
        "trace": [],
    }))

    assert result["selection_facets"] == ["order_summary", "cancel_rule"]
    assert result["rerank_pool_size"] == 2
    assert result["selection_latency_ms"] == 0.37
    assert "facets=order_summary,cancel_rule" in result["trace"][-1]
    assert "rerank_pool=2" in result["trace"][-1]


def test_query_faq_tool_exposes_evidence_selection_metadata():
    registry = ToolRegistryV2()
    register_builtin(
        registry,
        ToolContext(
            session_factory=None,
            service=_Service(),
            kb=_kb(),
            top_k=2,
        ),
    )
    tool = {record.name: record.tool for record in registry.records()}["query_faq"]

    payload = json.loads(tool.invoke({"keyword": "订单取消规则"}))

    assert payload["selection_facets"] == ["order_summary", "cancel_rule"]
    assert payload["selection_latency_ms"] == 0.37
    assert payload["rerank_pool_size"] == 2
