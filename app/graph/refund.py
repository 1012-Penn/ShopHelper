"""退款/售后确定性子流程(ch06):槽位检查→订单数据→Query 扩写→强制政策检索;窄化问题交主力 Agent。

扩写只在检索侧现查现用;缺订单号不让模型猜——发 order_selector 帧结束本轮,点选后 resume 旁路再进。
"""
import asyncio
import logging

from app.orders import extract_order_id, get_order, list_orders
from app.prompts import EXPAND_PROMPT

logger = logging.getLogger(__name__)


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def make_refund_nodes(expander, service, kb, pool, top_k: int = 10):
    async def prepare_order_node(state, writer=None, config=None) -> dict:
        msg = state.get("resolved_message") or ""
        # require_known:裸数字兜底(年份/尾号)必须命中目录才算单号;带「订单」上下文的放行(走未找到话术)
        oid = state.get("resume_order_id") or extract_order_id(msg, require_known=True) or ""
        return {"refund_flow": True, "pending_order_id": oid,
                "trace": [*state["trace"], f"node=prepare_order order={oid or 'missing'}"]}

    async def ask_order_node(state, writer=None, config=None) -> dict:
        orders = list_orders()
        _emit(writer, {"type": "order_selector",
                       "items": [{"order_id": o["order_id"], "product": o["product"],
                                  "amount": o["amount"], "status": o["status"]}
                                 for o in orders],
                       "question": state["resolved_message"],
                       "original": state["user_message"],
                       "intent": state["intent"]})
        return {"trace": [*state["trace"], f"node=ask_order n={len(orders)}"]}

    async def fetch_order_node(state, writer=None, config=None) -> dict:
        order = get_order(state["pending_order_id"])
        _emit(writer, {"type": "tool_status", "name": "query_order", "label": "订单查询"})
        return {"order": order,
                "trace": [*state["trace"], f"node=fetch_order found={'error' not in order}"]}

    async def expand_node(state, writer=None, config=None) -> dict:
        queries: list[str] = []
        try:
            result = await expander.ainvoke(
                EXPAND_PROMPT.format(query=state["resolved_message"]), config)
            raw = result.queries if hasattr(result, "queries") else (result or {}).get("queries", [])
            queries = [q.strip() for q in raw if isinstance(q, str) and q.strip()][:4]
        except Exception:
            logger.warning("expand 上游失败,退化为原句", exc_info=True)
        if not queries:
            queries = [state["resolved_message"]]
        _emit(writer, {"type": "tool_status", "name": "expand", "label": "查询扩写"})
        return {"expand_queries": queries,
                "trace": [*state["trace"], f"node=expand n={len(queries)}"]}

    async def policy_retrieve_node(state, writer=None, config=None) -> dict:
        merged: dict[int, float] = {}
        diagnostic: dict[int, dict] = {}
        rejected_reasons: list[str] = []
        for q in state["expand_queries"]:
            try:
                result = await asyncio.to_thread(
                    service.retrieve, q, strategy="hybrid_rerank", use_rewrite=False)
            except Exception:
                logger.warning("政策检索失败,该查询按空结果处理", exc_info=True)
                rejected_reasons.append("政策检索服务暂不可用")
                continue
            if result.low_confidence:
                if result.reason:
                    rejected_reasons.append(result.reason)
                else:
                    rejected_reasons.append("政策检索结果置信度低")
            snapshot_fn = getattr(result, "snapshot", None)
            try:
                snapshots = snapshot_fn(top_k) if callable(snapshot_fn) else [
                    {"rank": i + 1, "chunk_id": item.chunk_id, "score": item.score,
                     "rerank_score": item.score}
                    for i, item in enumerate(result.items[:top_k])
                ]
            except Exception:
                logger.warning("政策候选快照构建失败,回退到 items", exc_info=True)
                snapshots = [
                    {"rank": i + 1, "chunk_id": item.chunk_id, "score": item.score,
                     "rerank_score": item.score}
                    for i, item in enumerate(result.items[:top_k])
                ]
            for item in snapshots:
                cid = item.get("chunk_id")
                if cid is None:
                    continue
                score = item.get("score", item.get("rerank_score", 0.0))
                previous = diagnostic.get(cid)
                if previous is None or score > previous["score"]:
                    diagnostic[cid] = {**item, "chunk_id": cid, "score": score,
                                       "rerank_score": item.get("rerank_score", score)}
            if result.low_confidence:
                continue
            for r in result.items:
                if r.chunk_id not in merged or r.score > merged[r.chunk_id]:
                    merged[r.chunk_id] = r.score
        top = sorted(merged.items(), key=lambda kv: -kv[1])[:top_k]
        snapshots = sorted(diagnostic.values(), key=lambda item: -item["score"])[:top_k]
        snapshot_ids = [item["chunk_id"] for item in snapshots]
        chunk_ids = list(dict.fromkeys([*snapshot_ids, *(cid for cid, _ in top)]))
        try:
            rows = await asyncio.to_thread(kb.get_chunks, chunk_ids)
        except Exception:
            logger.warning("政策检索获取 chunk 详情失败,按空证据处理", exc_info=True)
            rejected_reasons.append("政策检索详情暂不可用")
            rows = []
        by_id = {r.id: r for r in rows if hasattr(r, "id")}
        if not by_id:
            by_id = dict(zip(chunk_ids, rows))
        valid_items = [(cid, sc, by_id[cid]) for cid, sc in top if cid in by_id]
        evidence = [
            {"n": i + 1, "chunk_id": cid,
             "question": (row.questions.splitlines() or [""])[0] if getattr(row, "questions", None) else "",
             "answer": getattr(row, "answer", "") or "",
             "category": getattr(row, "category", "") or "",
             "section_path": getattr(row, "section_path", "") or ""}
            for i, (cid, _score, row) in enumerate(valid_items)
        ]
        for item in snapshots:
            row = by_id.get(item["chunk_id"])
            if row is not None:
                item.update({"question": (row.questions.splitlines() or [""])[0]
                             if getattr(row, "questions", None) else "",
                             "answer": getattr(row, "answer", "") or "",
                             "category": getattr(row, "category", "") or "",
                             "section_path": getattr(row, "section_path", "") or ""})
        _emit(writer, {"type": "tool_status", "name": "query_faq", "label": "政策检索"})
        low = not evidence
        low_reason = ";".join(dict.fromkeys(rejected_reasons))
        if low and not low_reason:
            low_reason = "子流程政策检索无证据"
        if evidence:
            _emit(writer, {"type": "citations", "items": list(evidence)})
        else:
            try:
                # 落池可能内联标准化/查重(LLM),放线程池避免阻塞事件循环
                await asyncio.to_thread(
                    pool.insert, "retrieval_low_conf", state["session_id"],
                    state["user_message"], low_reason, retrieved_chunks=snapshots)
            except Exception:
                logger.warning("低置信度池写入失败", exc_info=True)
        top1 = f"{top[0][1]:.2f}" if top else "0.00"
        return {"evidence": evidence, "low_confidence": low,
                "retrieved_chunks": snapshots,
                "low_reason": low_reason if low else "",
                "trace": [*state["trace"],
                          f"node=policy_retrieve queries={len(state['expand_queries'])} "
                          f"merged={len(merged)} top1={top1} n={len(evidence)}"]}

    return prepare_order_node, ask_order_node, fetch_order_node, expand_node, policy_retrieve_node
