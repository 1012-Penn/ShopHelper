"""入口节点:指代消解(本章原样透传)+ 意图识别(简单 prompt 出 JSON)+ 写死分流。"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts import INTENT_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")
# 解析失败/超纲一律归商品咨询:走知识路径最安全,检索闸可拦
INTENT_FALLBACK = "商品咨询"

KNOWLEDGE_INTENTS = {"商品咨询", "退款退货"}
BUSINESS_INTENTS = {"物流", "订单", "售后"}


def _safe_intent(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return INTENT_FALLBACK
    intent = obj.get("intent") if isinstance(obj, dict) else None
    return intent if intent in INTENTS else INTENT_FALLBACK


def resolve_node(state) -> dict:
    return {"resolved_message": state["user_message"],
            "trace": [*state["trace"], "node=resolve passthrough"]}


async def intent_node(state, model) -> dict:
    resp = await model.ainvoke([SystemMessage(content=INTENT_PROMPT),
                                HumanMessage(content=state["resolved_message"])])
    intent = _safe_intent(resp.content)
    return {"intent": intent,
            "trace": [*state["trace"], f"node=intent intent={intent}"]}


def route(state) -> str:
    if state["intent"] in KNOWLEDGE_INTENTS:
        return "retrieve"
    if state["intent"] in BUSINESS_INTENTS:
        return "agent"
    return "complaint" if state["intent"] == "投诉" else "chitchat"
