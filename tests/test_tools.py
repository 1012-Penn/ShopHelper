import json

from sqlalchemy import select

from app.chunking import Chunk
from app.db import make_session_factory
from app.kb import KnowledgeBaseStore, vectorize_pending
from app.models import Faq, Ticket
from app.tools.builtin import build_default_registry
from tests.helpers import FakeEmbedding, FakeReranker, FakeRewriter, FakeVectorStore
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
    registry = build_default_registry(factory)
    tools = [r.tool for r in registry.records()]
    return [r.tool for r in registry.records()], factory


def _make_vector_tools(factory, *, vectors=None):
    """向量知识库 + 全替身注入,驱动 query_faq(含 FakeReranker 质量闸门)。"""
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
    registry = build_default_registry(factory, embedder=FakeEmbedding(), vectors=vectors, top_k=3,
                                      rewriter=FakeRewriter(), reranker=FakeReranker())
    tools = [r.tool for r in registry.records()]
    return tools, kb, ids


def test_build_tools_names_and_labels():
    tools, _ = _make()
    names = {t.name for t in tools}
    # ch08:query_logistics 内置下线,物流由 MCP Server 接管;插件(query_server_time)由 main 接线时装载
    assert names == {"query_order", "query_product", "query_faq", "create_ticket"}


def test_mock_tools_return_structured_json():
    tools, _ = _make()
    by_name = {t.name: t for t in tools}
    order = json.loads(by_name["query_order"].invoke({"order_id": "1001"}))
    assert order["order_id"] == "1001" and "status" in order and "amount" in order
    product = json.loads(by_name["query_product"].invoke({"keyword": "手机"}))
    assert "name" in product and "price" in product


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
    first = hit["items"][0]
    assert {"n", "chunk_id", "question", "answer", "category", "section_path"} <= set(first)
    assert first["question"] == "怎么申请退货?" and first["n"] == 1
    miss = json.loads(by_name["query_faq"].invoke({"keyword": "量子力学"}))
    assert miss["items"] == [] and miss["low_confidence"] is True and miss["reason"]


def test_query_faq_never_raises_on_backend_failure():
    class ExplodingEmbedder:
        def embed(self, texts):
            raise RuntimeError("嵌入服务挂了")

    factory = make_session_factory(create_memory_engine())
    registry = build_default_registry(factory, embedder=ExplodingEmbedder(), vectors=FakeVectorStore(), top_k=3)
    tools2 = [r.tool for r in registry.records()]
    faq = {t.name: t for t in tools2}["query_faq"]
    out = json.loads(faq.invoke({"keyword": "退货"}))
    assert out["items"] == [] and out["low_confidence"] is True  # 异常收敛为空 + 低置信标志
    assert "检索服务暂不可用" in out["reason"]


def test_build_tools_injects_default_top_k():
    """评审修复回归:注入替身但漏传 top_k 时,检索拿到的候选上限必须是正整数而非 None。"""
    factory = make_session_factory(create_memory_engine())
    captured = {}

    class RecordingVectorStore(FakeVectorStore):
        def hybrid_search(self, vector, query_text, top_k, filter_expr=None):
            captured["top_k"] = top_k
            return super().hybrid_search(vector, query_text, top_k, filter_expr)

    tools, _, _ = _make_vector_tools(factory, vectors=RecordingVectorStore())
    faq = {t.name: t for t in tools}["query_faq"]
    out = json.loads(faq.invoke({"keyword": "快递费多少钱"}))
    assert isinstance(captured["top_k"], int) and captured["top_k"] > 0
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


def _make_v3_tools(factory, *, vectors=None, top_k=3):
    """v3 全替身注入(rewriter/reranker 显式给),驱动精排与反漏斗断言。"""
    kb = KnowledgeBaseStore(factory)
    chunks = [
        Chunk("售后政策", "退货的规定是什么", "支持七天无理由退货,商品需保持完好。",
              "d.md > 退货的规定是什么", "faq", False),
        Chunk("数码配件", "SH-E300 无线降噪耳机", "SH-E300 售价 299 元,支持主动降噪。",
              "d.md > SH-E300", "faq", False),
    ]
    kb.replace_doc_chunks("d.md", chunks)
    vectors = vectors or FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    registry = build_default_registry(factory, embedder=FakeEmbedding(), vectors=vectors, top_k=top_k,
                                      rewriter=FakeRewriter(), reranker=FakeReranker())
    tools = [r.tool for r in registry.records()]
    return tools, kb, vectors


def test_query_faq_v3_contract():
    factory = make_session_factory(create_memory_engine())
    tools, _, _ = _make_v3_tools(factory)
    out = json.loads({t.name: t for t in tools}["query_faq"].invoke({"keyword": "SH-E300 降噪"}))
    assert out["low_confidence"] is False and out["filter_fallback"] is False
    assert [it["n"] for it in out["items"]] == list(range(1, len(out["items"]) + 1))  # n 连续从 1
    first = out["items"][0]
    assert first["chunk_id"] == 2 and first["section_path"] == "d.md > SH-E300"


def test_query_faq_v3_category_passthrough():
    factory = make_session_factory(create_memory_engine())
    tools, _, _ = _make_v3_tools(factory)
    faq = {t.name: t for t in tools}["query_faq"]
    # 品类过滤不命中 → 回退无过滤,filter_fallback=true
    out = json.loads(faq.invoke({"keyword": "退货", "category": "不存在的品类"}))
    assert out["filter_fallback"] is True and out["items"]
    # 品类过滤命中 → 只剩该品类
    out2 = json.loads(faq.invoke({"keyword": "退货", "category": "售后政策"}))
    assert out2["filter_fallback"] is False
    assert all(it["category"] == "售后政策" for it in out2["items"])


def test_query_faq_v3_lost_in_middle_arrangement():
    """精排名次 n 与摆放解耦:n=1 在首位、n=2 在末位(4 条证据时摆放序 [1,3,4,2])。"""
    factory = make_session_factory(create_memory_engine())
    kb = KnowledgeBaseStore(factory)
    chunks = [Chunk("售后政策", f"退货问题{i}", f"退货说明{i},七天无理由。",
                    f"d.md > 退货问题{i}", "faq", False) for i in range(1, 4)]
    chunks.append(Chunk("售后政策", "退货政策是什么", "支持七天无理由退货,需保持完好。",
                        "d.md > 退货政策是什么", "faq", False))
    kb.replace_doc_chunks("d.md", chunks)
    vectors = FakeVectorStore()
    vectorize_pending(kb, vectors, FakeEmbedding())
    registry = build_default_registry(factory, embedder=FakeEmbedding(), vectors=vectors, top_k=4,
                                      rewriter=FakeRewriter(), reranker=FakeReranker())
    tools = [r.tool for r in registry.records()]
    out = json.loads({t.name: t for t in tools}["query_faq"].invoke({"keyword": "退货"}))
    ns = [it["n"] for it in out["items"]]
    assert len(ns) == 4 and ns[0] == 1 and ns[-1] == 2  # 反漏斗:n=1 首位、n=2 末位


def test_query_faq_v3_exception_converges():
    """内部异常(如 milvus 库锁)收敛为空结果 + low_confidence,不抛错。"""
    class Broken:
        def ensure_collection(self):
            raise RuntimeError("db lock")

        def __getattr__(self, name):
            raise RuntimeError("db lock")

    factory = make_session_factory(create_memory_engine())
    registry = build_default_registry(factory, embedder=FakeEmbedding(), vectors=Broken(), top_k=3,
                                      rewriter=FakeRewriter(), reranker=FakeReranker())
    tools = [r.tool for r in registry.records()]
    out = json.loads({t.name: t for t in tools}["query_faq"].invoke({"keyword": "邮费"}))
    assert out["items"] == [] and out["low_confidence"] is True and "检索服务暂不可用" in out["reason"]
