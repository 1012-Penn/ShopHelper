"""退款售后确定性子流程:槽位→取数→扩写→强制政策检索。"""
import asyncio
import threading

from app.graph.refund import make_refund_nodes
from app.prompts import ExpandedQueries
from app.retrieval import Retrieved
from tests.helpers import StubExtractModel


class StubService:
    """retrieve 替身:按查询文本命中关键词返回预置 chunk 序列(带分),记录调用。"""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def retrieve(self, query, strategy="hybrid_rerank", category=None, use_rewrite=True):
        self.calls.append((query, use_rewrite))
        for key, items in self.mapping.items():
            if key in query:
                return type("R", (), {"items": [Retrieved(chunk_id=c, score=s) for c, s in items],
                                      "low_confidence": False, "reason": "",
                                      "filter_fallback": False, "rewritten": "",
                                      "search_text": "", "degraded": False})()
        return type("R", (), {"items": [], "low_confidence": True, "reason": "空",
                              "filter_fallback": False, "rewritten": "",
                              "search_text": "", "degraded": False})()


class StubKB:
    """按 chunk_id 生成确定性行:question/answer 都带 id 可断言对应关系。"""

    def get_chunks(self, ids):
        rows = []
        for cid in ids:
            rows.append(type("Row", (), {"questions": f"问题{cid}\n变体", "answer": f"答案{cid}",
                                         "category": "退货政策",
                                         "section_path": f"退货政策.md > 节{cid}", "id": cid})())
        return rows


class StubPool:
    def __init__(self):
        self.inserted = []

    def insert(self, source, session_id, question, reason, retrieved_chunks=None):
        self.inserted.append((source, question, reason, retrieved_chunks))


def _state(**kw):
    base = dict(session_id=1, user_message="这个能退吗", resolved_message="订单1001的无线耳机能退吗",
                intent="退款退货", intent_confidence=0.9, evidence=[], low_confidence=False,
                low_reason="", gate_passed=False, agent_steps=0, final_reply="",
                suggested_actions=[], trace=[], messages=[], history=[], order={},
                expand_queries=[], refund_flow=False, pending_order_id="",
                resume_order_id="", resume_question="")
    base.update(kw)
    return base


def _nodes(mapping=None, top_k=3):
    service = StubService(mapping if mapping is not None else {"退货": [(11, 0.9), (12, 0.5)]})
    pool = StubPool()
    expander = StubExtractModel(result=ExpandedQueries(
        queries=["无线耳机退货条件", "七天无理由时效", "耳机类特殊品类限制"]))
    nodes = make_refund_nodes(expander, service, StubKB(), pool, top_k=top_k)
    return nodes, service, pool


async def test_prepare_order_extracts_from_resolved():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    out = await prep(_state(), writer=None)
    assert out["pending_order_id"] == "1001" and out["refund_flow"] is True
    assert "node=prepare_order order=1001" in out["trace"][-1]


async def test_prepare_order_resume_id_priority():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    out = await prep(_state(resume_order_id="1003",
                            resolved_message="订单1003的硅胶手机壳能退吗"), writer=None)
    assert out["pending_order_id"] == "1003"


async def test_prepare_order_missing_goes_empty():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    out = await prep(_state(resolved_message="这个能退吗"), writer=None)
    assert out["pending_order_id"] == ""


async def test_ask_order_emits_selector_frame_and_user_only_messages():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    frames = []
    out = await ask(_state(resolved_message="这个能退吗"), writer=frames.append)
    assert frames[0]["type"] == "order_selector"
    assert len(frames[0]["items"]) == 3
    assert frames[0]["items"][0]["order_id"] == "1001"
    assert frames[0]["question"] == "这个能退吗"
    assert frames[0]["original"] == "这个能退吗"
    assert "messages" not in out  # ch07:user 消息由 initial_state 带入,ask_order 零吐
    assert "node=ask_order" in out["trace"][-1]


async def test_fetch_order_hit_and_miss():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    frames = []
    out = await fetch(_state(pending_order_id="1001"), writer=frames.append)
    assert out["order"]["product"] == "无线耳机"
    assert frames and frames[0]["label"] == "订单查询"
    out2 = await fetch(_state(pending_order_id="9999"), writer=None)
    assert out2["order"]["error"] == "未找到该订单"


async def test_expand_emits_queries_and_badge():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    frames = []
    out = await expand(_state(), writer=frames.append)
    assert len(out["expand_queries"]) == 3
    assert frames[0]["label"] == "查询扩写"
    assert "node=expand n=3" in out["trace"][-1]


async def test_expand_failure_falls_back_to_resolved():
    bad = make_refund_nodes(StubExtractModel(error=RuntimeError("挂了")),
                            StubService({}), StubKB(), StubPool(), top_k=3)[3]
    out = await bad(_state(), writer=None)
    assert out["expand_queries"] == ["订单1001的无线耳机能退吗"]


async def test_policy_retrieve_merges_dedup_renumbers():
    mapping = {"无线耳机退货条件": [(11, 0.9), (12, 0.5)],
               "七天无理由时效": [(11, 0.7), (13, 0.85)],
               "耳机类特殊品类限制": [(14, 0.6)]}
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes(mapping, top_k=3)
    frames = []
    out = await retrieve(_state(expand_queries=list(mapping.keys())), writer=frames.append)
    # 11 两条查询都命中 → 去重留最高分 0.9;合并序 11(0.9) > 13(0.85) > 14(0.6),12(0.5) 出局
    assert [e["n"] for e in out["evidence"]] == [1, 2, 3]
    assert [e["chunk_id"] for e in out["evidence"]] == [11, 13, 14]
    assert out["evidence"][0]["question"] == "问题11" and out["evidence"][0]["answer"] == "答案11"
    assert [f["type"] for f in frames] == ["tool_status", "citations"]
    assert len(frames[1]["items"]) == 3
    assert "node=policy_retrieve" in out["trace"][-1]
    assert all(use_rewrite is False for _, use_rewrite in service.calls)  # 扩写产物不二次改写
    assert [query for query, _ in service.calls] == list(mapping.keys())  # 顺序逐条检索
    assert pool.inserted == []


async def test_policy_retrieve_excludes_low_confidence_candidates():
    class LowConfidenceService:
        def retrieve(self, query, strategy="hybrid_rerank", category=None, use_rewrite=True):
            candidates = {"政策一": [Retrieved(chunk_id=11, score=0.9),
                                  Retrieved(chunk_id=12, score=0.8)],
                          "政策二": [Retrieved(chunk_id=11, score=0.95),
                                  Retrieved(chunk_id=13, score=0.88)]}[query]
            return type("R", (), {
                "items": candidates,
                "low_confidence": True,
                "reason": "reranker unavailable",
                "degraded": True,
            })()

    pool = StubPool()
    expander = StubExtractModel(result=ExpandedQueries(queries=["政策一", "政策二"]))
    retrieve = make_refund_nodes(expander, LowConfidenceService(), StubKB(), pool, top_k=2)[4]

    out = await retrieve(_state(expand_queries=["政策一", "政策二"]), writer=None)

    assert out["evidence"] == [] and out["low_confidence"] is True
    assert "reranker unavailable" in out["low_reason"]
    assert pool.inserted and pool.inserted[0][0] == "retrieval_low_conf"
    assert [item["chunk_id"] for item in out["retrieved_chunks"]] == [11, 13]
    assert out["retrieved_chunks"][0]["score"] == 0.95
    assert out["retrieved_chunks"][0]["answer"] == "答案11"
    assert [item["chunk_id"] for item in pool.inserted[0][3]] == [11, 13]


async def test_policy_retrieve_mixed_confidence_only_uses_trusted_evidence():
    class MixedService:
        def retrieve(self, query, strategy="hybrid_rerank", category=None, use_rewrite=True):
            if query == "可信查询":
                items, low = [Retrieved(11, 0.8)], False
            else:
                items, low = [Retrieved(12, 0.99)], True
            return type("R", (), {"items": items, "low_confidence": low,
                                   "reason": "重排失败" if low else ""})()

    frames = []
    retrieve = make_refund_nodes(StubExtractModel(), MixedService(), StubKB(), StubPool())[4]
    out = await retrieve(_state(expand_queries=["可信查询", "降级查询"]), writer=frames.append)

    assert [item["chunk_id"] for item in out["evidence"]] == [11]
    assert [item["chunk_id"] for item in frames[-1]["items"]] == [11]


async def test_policy_retrieve_offloads_blocking_retrieval_and_chunk_lookup():
    retrieve_started, retrieve_release = threading.Event(), threading.Event()
    chunks_started, chunks_release = threading.Event(), threading.Event()

    class BlockingService:
        def retrieve(self, query, strategy="hybrid_rerank", category=None, use_rewrite=True):
            retrieve_started.set()
            retrieve_release.wait()
            return type("R", (), {"items": [Retrieved(11, 0.9)],
                                   "low_confidence": False, "reason": ""})()

    class BlockingKB:
        def get_chunks(self, ids):
            chunks_started.set()
            chunks_release.wait()
            return [type("Row", (), {"id": 11, "questions": "Q", "answer": "A",
                                     "category": "C", "section_path": "S"})()]

    expander = StubExtractModel(result=ExpandedQueries(queries=["政策一"]))
    retrieve = make_refund_nodes(expander, BlockingService(), BlockingKB(), StubPool())[4]
    loop = asyncio.get_running_loop()

    async def prove_loop_is_live(started, release):
        await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2)
        progressed = asyncio.Event()
        loop.call_soon(progressed.set)
        await asyncio.wait_for(progressed.wait(), timeout=2)
        assert not release.is_set(), "blocking call was released before the loop progressed"

    watchdog = threading.Timer(3, lambda: (retrieve_release.set(), chunks_release.set()))
    watchdog.start()
    try:
        task = asyncio.create_task(retrieve(_state(expand_queries=["政策一"]), writer=None))
        await prove_loop_is_live(retrieve_started, retrieve_release)
        retrieve_release.set()
        await prove_loop_is_live(chunks_started, chunks_release)
        chunks_release.set()
        out = await task
    finally:
        retrieve_release.set()
        chunks_release.set()
        watchdog.cancel()

    assert out["evidence"][0]["answer"] == "A"


async def test_policy_retrieve_empty_evidence_lands_pool():
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes({})
    out = await retrieve(_state(expand_queries=["无人问津的问题"]), writer=None)
    assert out["evidence"] == [] and out["low_confidence"] is True
    assert pool.inserted and pool.inserted[0][0] == "retrieval_low_conf"


async def test_policy_retrieve_kb_error_graceful_fallback():
    class BrokenKB:
        def get_chunks(self, ids):
            raise RuntimeError("数据库连不上了")

    mapping = {"无线耳机退货条件": [(11, 0.9)]}
    service = StubService(mapping)
    pool = StubPool()
    from app.prompts import ExpandedQueries
    expander = StubExtractModel(result=ExpandedQueries(queries=["无线耳机退货条件"]))
    (prep, ask, fetch, expand, retrieve) = make_refund_nodes(expander, service, BrokenKB(), pool, top_k=3)
    out = await retrieve(_state(expand_queries=["无线耳机退货条件"]), writer=None)
    assert out["evidence"] == []
    assert out["low_confidence"] is True
    assert pool.inserted and pool.inserted[0][0] == "retrieval_low_conf"


async def test_policy_retrieve_partial_chunks_aligned():
    """当检索出的 chunk 某一行在 DB 中缺失时,其余行应按 id 精准对齐而非按索引错位。"""
    class PartialKB:
        def get_chunks(self, ids):
            # 只有 11 和 14 存在,13 缺失
            rows = []
            for cid in ids:
                if cid != 13:
                    rows.append(type("Row", (), {
                        "id": cid,
                        "questions": f"问题{cid}\n变体",
                        "answer": f"答案{cid}",
                        "category": "退货政策",
                        "section_path": f"退货政策.md > 节{cid}",
                    })())
            return rows

    mapping = {"无线耳机退货条件": [(11, 0.9)],
               "七天无理由时效": [(13, 0.85)],
               "耳机类特殊品类限制": [(14, 0.6)]}
    service = StubService(mapping)
    pool = StubPool()
    from app.prompts import ExpandedQueries
    expander = StubExtractModel(result=ExpandedQueries(queries=list(mapping.keys())))
    (prep, ask, fetch, expand, retrieve) = make_refund_nodes(expander, service, PartialKB(), pool, top_k=3)
    out = await retrieve(_state(expand_queries=list(mapping.keys())), writer=None)
    # 13 缺失,剩余 11 和 14;14 必须对齐到 chunk_id=14,不能错位对齐到 13
    assert [e["chunk_id"] for e in out["evidence"]] == [11, 14]
    assert out["evidence"][0]["answer"] == "答案11"
    assert out["evidence"][1]["answer"] == "答案14"
    assert [e["n"] for e in out["evidence"]] == [1, 2]


async def test_prepare_order_rejects_unknown_bare_digits():
    """I-2/M-3 回归:裸数字(年份/尾号)必须命中目录;商品名不再硬绑单号。"""
    (prep, ask, fetch, expand, retrieve), service, pool = _nodes()
    out = await prep(_state(resolved_message="我2024年买的东西能退吗"), writer=None)
    assert out["pending_order_id"] == ""          # 年份误抓 → 走选择器
    out2 = await prep(_state(resolved_message="尾号5678能退吗"), writer=None)
    assert out2["pending_order_id"] == ""         # 尾号误抓 → 走选择器
    out3 = await prep(_state(resolved_message="SH-E300能退吗"), writer=None)
    assert out3["pending_order_id"] == ""         # 商品名不硬绑订单(I-2 移除)
    out4 = await prep(_state(resolved_message="订单424242能退吗"), writer=None)
    assert out4["pending_order_id"] == "424242"   # 带上下文的未知单号放行 → 未找到话术
