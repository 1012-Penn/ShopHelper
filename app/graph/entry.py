"""入口节点:指代消解(ch06 正式版)+ 意图识别(四件套,置信度)+ 写死分流。"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts import INTENT_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")
# 解析失败/超纲一律归「其他」:交主力 Agent 需求澄清,不硬塞业务意图
INTENT_FALLBACK = "其他"

KNOWLEDGE_INTENTS = {"商品咨询", "退款退货"}
BUSINESS_INTENTS = {"物流", "订单", "售后"}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_intent(raw: str) -> tuple[str, float, bool]:
    """裸 JSON 契约的宽松解析:剥代码围栏 → json.loads → 枚举校验;畸形/超纲归其他(malformed=true)。"""
    text = (raw or "").strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return INTENT_FALLBACK, 0.0, True
    intent = obj.get("intent") if isinstance(obj, dict) else None
    if intent not in INTENTS:
        return INTENT_FALLBACK, 0.0, True
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return intent, min(max(conf, 0.0), 1.0), False


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
