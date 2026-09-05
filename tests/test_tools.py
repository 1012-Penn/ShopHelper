import json

from sqlalchemy import select

from app.chunking import Chunk
from app.db import make_session_factory
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.models import Faq, Ticket
from app.tools.definitions import TOOL_LABELS, build_tools
from tests.helpers import FakeEmbedding, FakeVectorStore
from tests.test_models import create_memory_engine


def _make():
    engine = create_memory_engine()
    factory = make_session_factory(engine)
    with factory() as s:
        s.add_all(Faq(question=q, answer=a, category=c) for q, a, c in [
            ("退货政策是什么?", "七天无理由退货。", "售后"),
            ("多久能发货?", "48 小时内发出。", "物流"),
        ])
        s.commit()
    return build_tools(factory), factory


def _make_vector_tools(factory, *, vectors=None):
    """向量知识库 + 替身嵌入,驱动的 query_faq。"""
    kb = KnowledgeBaseStore(factory)
    chunks = [
        Chunk("退货政策", "邮费与运费", "普通订单邮费 8 元,满 99 元包邮;质量问题退货,运费由商家承担。",
              "退货政策.md > 退货政策 > 邮费与运费", "policy", True),
        Chunk("售后", "怎么申请退货?", "订单详情页点击「申请售后」提交。",
              "商品FAQ.md > 商品FAQ > 售后 > 怎么申请退货?", "faq", False),
    ]
    _, ids = kb.replace_doc_chunks("docs", chunks)
    vectors = vectors or FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    tools = build_tools(factory, embedder=FakeEmbedding(), vectors=vectors, top_k=3)
    return tools, kb, ids


def test_build_tools_names_and_labels():
    tools, _ = _make()
    names = {t.name for t in tools}
    assert names == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}
    assert TOOL_LABELS["query_logistics"] == "物流查询"


def test_mock_tools_return_structured_json():
    tools, _ = _make()
    by_name = {t.name: t for t in tools}
    order = json.loads(by_name["query_order"].invoke({"order_id": "1001"}))
    assert order["order_id"] == "1001" and "status" in order and "amount" in order
    product = json.loads(by_name["query_product"].invoke({"keyword": "手机"}))
    assert "name" in product and "price" in product
    logistics = json.loads(by_name["query_logistics"].invoke({"order_id": "1001"}))
    assert logistics["order_id"] == "1001" and len(logistics["traces"]) >= 1


def test_query_faq_paraphrase_hit():
    """验收 1 的替身版:「快递费多少钱」(换说法)必须召回运费说明。"""
    factory = make_session_factory(create_memory_engine())
    tools, _, _ = _make_vector_tools(factory)
    by_name = {t.name: t for t in tools}
    out = json.loads(by_name["query_faq"].invoke({"keyword": "快递费多少钱"}))
    assert out["items"], "换说法必须命中"
    assert out["items"][0]["question"] == "邮费与运费"
    assert "满 99 元包邮" in out["items"][0]["answer"]
    assert out["items"][0]["category"] == "退货政策"


def test_query_faq_contract_shape_and_miss():
    factory = make_session_factory(create_memory_engine())
    tools, _, _ = _make_vector_tools(factory)
    by_name = {t.name: t for t in tools}
    hit = json.loads(by_name["query_faq"].invoke({"keyword": "怎么申请退货"}))
    assert hit["items"][0] == {
        "question": "怎么申请退货?", "answer": "订单详情页点击「申请售后」提交。", "category": "售后",
    }
    miss = json.loads(by_name["query_faq"].invoke({"keyword": "量子力学"}))
    assert miss == {"items": []}


def test_query_faq_never_raises_on_backend_failure():
    class ExplodingEmbedder:
        def embed(self, texts):
            raise RuntimeError("嵌入服务挂了")

    factory = make_session_factory(create_memory_engine())
    tools2 = build_tools(factory, embedder=ExplodingEmbedder(), vectors=FakeVectorStore(), top_k=3)
    faq = {t.name: t for t in tools2}["query_faq"]
    assert json.loads(faq.invoke({"keyword": "退货"})) == {"items": []}  # 异常收敛为空结果


def test_build_tools_injects_default_top_k():
    """评审修复回归:注入替身但漏传 top_k 时,search 必须拿到 3 而不是 None。"""
    factory = make_session_factory(create_memory_engine())
    captured = {}

    class RecordingVectorStore(FakeVectorStore):
        def search(self, vector, top_k):
            captured["top_k"] = top_k
            return super().search(vector, top_k)

    tools, _, _ = _make_vector_tools(factory, vectors=RecordingVectorStore())
    faq = {t.name: t for t in tools}["query_faq"]
    out = json.loads(faq.invoke({"keyword": "快递费多少钱"}))
    assert captured["top_k"] == 3
    assert out["items"]  # 漏传 top_k 不应变成静默空结果


def test_create_ticket_persists():
    tools, factory = _make()
    by_name = {t.name: t for t in tools}
    result = json.loads(by_name["create_ticket"].invoke({
        "conversation_id": 1, "description": "想退货", "ticket_type": "售后",
    }))
    assert result["ticket_no"].startswith("T")
    with factory() as s:
        ticket = s.get(Ticket, result["ticket_no"])
        assert ticket is not None
        assert ticket.status == "待处理"
        assert ticket.conversation_id == 1


def test_create_ticket_rejects_bad_type():
    tools, _ = _make()
    by_name = {t.name: t for t in tools}
    try:
        by_name["create_ticket"].invoke({
            "conversation_id": 1, "description": "x", "ticket_type": "砍价",
        })
        raised = False
    except Exception:
        raised = True
    assert raised
