"""退款/售后确定性子流程(ch06):槽位检查→订单数据→Query 扩写→强制政策检索;窄化问题交主力 Agent。

扩写只在检索侧现查现用;缺订单号不让模型猜——发 order_selector 帧结束本轮,点选后 resume 旁路再进。
"""
import logging

from app.orders import extract_order_id, get_order, list_orders
from app.prompts import EXPAND_PROMPT

logger = logging.getLogger(__name__)


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def make_refund_nodes(expander, service, kb, pool, top_k: int = 10):
    async def prepare_order_node(state, writer=None) -> dict:
        oid = state.get("resume_order_id") or extract_order_id(state["resolved_message"]) or ""
        return {"refund_flow": True, "pending_order_id": oid,
                "trace": [*state["trace"], f"node=prepare_order order={oid or 'missing'}"]}

    async def ask_order_node(state, writer=None) -> dict:
        _emit(writer, {"type": "order_selector",
                       "items": [{"order_id": o["order_id"], "product": o["product"],
                                  "amount": o["amount"], "status": o["status"]}
                                 for o in list_orders()],
                       "question": state["resolved_message"],
                       "original": state["user_message"]})
        return {"messages": [{"role": "user", "content": state["user_message"]}],
                "trace": [*state["trace"], "node=ask_order n=3"]}

    async def fetch_order_node(state, writer=None) -> dict:
        order = get_order(state["pending_order_id"])
        _emit(writer, {"type": "tool_status", "name": "query_order", "label": "订单查询"})
        return {"order": order,
                "trace": [*state["trace"], f"node=fetch_order found={'error' not in order}"]}

    async def expand_node(state, writer=None) -> dict:
        queries: list[str] = []
        try:
            result = await expander.ainvoke(EXPAND_PROMPT.format(query=state["resolved_message"]))
            raw = result.queries if hasattr(result, "queries") else (result or {}).get("queries", [])
            queries = [q.strip() for q in raw if isinstance(q, str) and q.strip()][:4]
        except Exception:
            logger.warning("expand 上游失败,退化为原句", exc_info=True)
        if not queries:
            queries = [state["resolved_message"]]
        _emit(writer, {"type": "tool_status", "name": "expand", "label": "查询扩写"})
        return {"expand_queries": queries,
                "trace": [*state["trace"], f"node=expand n={len(queries)}"]}

    async def policy_retrieve_node(state, writer=None) -> dict:
        merged: dict[int, float] = {}
        for q in state["expand_queries"]:
            try:
                result = service.retrieve(q, strategy="hybrid_rerank", use_rewrite=False)
            except Exception:
                logger.warning("政策检索失败,该查询按空结果处理", exc_info=True)
                continue
            for r in result.items:
                if r.chunk_id not in merged or r.score > merged[r.chunk_id]:
                    merged[r.chunk_id] = r.score
        top = sorted(merged.items(), key=lambda kv: -kv[1])[:top_k]
        rows = kb.get_chunks([cid for cid, _ in top])
        evidence = [
            {"n": i + 1, "chunk_id": cid,
             "question": (row.questions.splitlines() or [""])[0], "answer": row.answer,
             "category": row.category, "section_path": row.section_path or ""}
            for i, (cid, _score), row in zip(range(len(top)), top, rows)
        ]
        _emit(writer, {"type": "tool_status", "name": "query_faq", "label": "政策检索"})
        low = not evidence
        if evidence:
            _emit(writer, {"type": "citations", "items": list(evidence)})
        else:
            pool.insert("retrieval_low_conf", state["session_id"],
                        state["user_message"], "子流程政策检索无证据")
        top1 = f"{top[0][1]:.2f}" if top else "0.00"
        return {"evidence": evidence, "low_confidence": low,
                "low_reason": "子流程政策检索无证据" if low else "",
                "trace": [*state["trace"],
                          f"node=policy_retrieve queries={len(state['expand_queries'])} "
                          f"merged={len(merged)} top1={top1} n={len(evidence)}"]}

    return prepare_order_node, ask_order_node, fetch_order_node, expand_node, policy_retrieve_node
