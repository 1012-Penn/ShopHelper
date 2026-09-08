"""log 节点:整轮收尾——落库、拒答落池、trace 逐行日志、actions 帧下发。"""
import logging

from app.guard import is_refusal
from app.graph.simple import ACTION_LABELS

logger = logging.getLogger(__name__)


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def make_log_node(store, pool):
    async def log_node(state, writer=None) -> dict:
        await store.append(state["session_id"], state["messages"])
        if is_refusal(state["final_reply"]):
            pool.insert("self_check", state["session_id"], state["user_message"],
                        "模型自评证据不足:" + (state["final_reply"] or "")[:200])
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
