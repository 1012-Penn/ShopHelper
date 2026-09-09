"""log 节点:整轮收尾——落库、拒答落池、trace 逐行日志、actions 帧下发。

ch07 落库契约:messages 表只落 user 行与含文本 assistant 行;assistant 纯工具调用行与
tool 行不落库(工具轨迹只活在当轮 State/checkpoint,跨轮事实靠摘要延续)。
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage

from app.guard import is_refusal
from app.graph.simple import ACTION_LABELS

logger = logging.getLogger(__name__)


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def _turn_rows(state) -> list[dict]:
    """按 turn_user_msg_id 定位本轮起点,切出新消息并过滤落库形态。

    assistant 只落最后一条含文本消息:ReAct 中间步骤的「我先查一下」碎片与回灌用
    AI/Tool 消息不是对话原文,落库会污染回载与下一轮的层 1 历史。
    """
    msgs = state.get("messages") or []
    turn_id = state.get("turn_user_msg_id")
    start = next((i for i, m in enumerate(msgs) if getattr(m, "id", None) == turn_id), None)
    if start is None:
        logger.warning("session=%s 未找到本轮 user 消息 %s,本轮回避落库",
                       state["session_id"], turn_id)
        return []
    rows: list[dict] = []
    last_ai = None
    for m in msgs[start:]:
        if isinstance(m, HumanMessage):
            rows.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            last_ai = m
    if last_ai is not None:
        rows.append({"role": "assistant", "content": last_ai.content})
    return rows


def make_log_node(store, pool):
    async def log_node(state, writer=None) -> dict:
        await store.append(state["session_id"], _turn_rows(state))
        if is_refusal(state["final_reply"]):
            snapshot = state.get("retrieved_chunks") or []
            reason = "模型自评证据不足:" + (state["final_reply"] or "")[:200]
            if snapshot:
                pool.insert("self_check", state["session_id"], state["user_message"], reason, snapshot)
            else:
                pool.insert("self_check", state["session_id"], state["user_message"], reason)
        for line in state["trace"]:
            logger.info("[ch05] session=%s %s", state["session_id"], line)
        if state["suggested_actions"]:
            items = []
            for a in state["suggested_actions"]:
                order = state.get("order") or {}
                if a == "refund_form" and (not order or order.get("error")):
                    continue  # 未知订单不给退款表单入口
                item = {"action": a, "label": ACTION_LABELS.get(a, a)}
                if a == "refund_form" and order:
                    item["order_id"] = order.get("order_id", "")
                items.append(item)
            if items:
                _emit(writer, {"type": "actions", "items": items})
        return {"trace": [*state["trace"], "node=log persisted"]}

    return log_node
