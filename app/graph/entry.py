"""入口节点:指代消解+改写(ch06 正式版,带 resume 旁路)+ 意图识别(四件套,置信度+降级路)+ 写死分流。"""
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts import INTENT_PROMPT, RESOLVE_PROMPT, ResolvedQuestion

logger = logging.getLogger(__name__)

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")
# 解析失败/超纲一律归「其他」:交主力 Agent 需求澄清,不硬塞业务意图
INTENT_FALLBACK = "其他"

KNOWLEDGE_INTENTS = {"商品咨询"}
REFUND_INTENTS = {"退款退货", "售后"}
BUSINESS_INTENTS = {"物流", "订单"}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

_ROLE_ZH = {"user": "用户", "assistant": "客服"}


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


def _history_text(history: list) -> str:
    return "\n".join(f"{_ROLE_ZH.get(h.get('role'), h.get('role'))}:{h.get('content')}"
                     for h in history if h.get("content")) or "(无)"


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def make_resolve_node(resolver):
    """resolver = with_structured_output(ResolvedQuestion) 产物;resume 旁路不走 LLM;异常兜底透传。"""

    async def resolve_node(state, writer=None) -> dict:
        if state.get("resume_question"):
            q = state["resume_question"]
            _emit(writer, {"type": "resolved", "changed": True, "question": q})
            return {"resolved_message": q,
                    "trace": [*state["trace"], f"node=resolve resume_bypass q={q}"]}
        result = None
        try:
            result = await resolver.ainvoke(
                RESOLVE_PROMPT.format(history=_history_text(state["history"]),
                                      query=state["user_message"]))
        except Exception:
            logger.warning("resolve 上游失败,原样透传", exc_info=True)
        resolved = str(getattr(result, "resolved", "") or "").strip()
        if not resolved:
            resolved = state["user_message"]
        changed = bool(getattr(result, "changed", False)) and resolved != state["user_message"]
        if changed:
            _emit(writer, {"type": "resolved", "changed": True, "question": resolved})
        return {"resolved_message": resolved,
                "trace": [*state["trace"], f"node=resolve changed={changed} q={resolved}"]}

    return resolve_node


def make_intent_node(model, escalator=None, floor: float = 0.6):
    """默认单次大模型;escalator 给定时小模型先判、confidence<floor 大模型重判一次取其结果。"""

    async def intent_node(state) -> dict:
        messages = [SystemMessage(content=INTENT_PROMPT),
                    HumanMessage(content=state["resolved_message"])]
        resp = await model.ainvoke(messages)
        intent, conf, malformed = parse_intent(getattr(resp, "content", ""))
        escalated = False
        if escalator is not None and conf < floor:
            resp2 = await escalator.ainvoke(messages)
            intent, conf, malformed = parse_intent(getattr(resp2, "content", ""))
            escalated = True
        line = f"node=intent intent={intent} confidence={conf:.2f}"
        if escalated:
            line += " escalated=true"
        if malformed:
            line += " malformed=true"
        return {"intent": intent, "intent_confidence": conf,
                "trace": [*state["trace"], line]}

    return intent_node


def route(state) -> str:
    if state["intent"] in KNOWLEDGE_INTENTS:
        return "retrieve"
    if state["intent"] in REFUND_INTENTS:
        return "prepare_order"
    if state["intent"] in BUSINESS_INTENTS:
        return "agent"
    if state["intent"] == "投诉":
        return "complaint"
    if state["intent"] == "闲聊":
        return "chitchat"
    return "agent"  # 其他/兜底:交主力 Agent 需求澄清
