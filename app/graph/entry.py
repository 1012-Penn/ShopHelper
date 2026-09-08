"""入口节点:指代消解+改写(ch06 正式版,带 resume 旁路)+ 意图识别(四件套,置信度+降级路)+ 写死分流。"""
import json
import logging
import math
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


def parse_intent(raw: str) -> tuple[str, float, bool]:
    """裸 JSON 契约的宽松解析:剥代码围栏 → json.loads → 枚举校验;畸形/超纲归其他(malformed=true)。"""
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    text = (str(raw) if raw is not None else "").strip()
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
        if math.isnan(conf) or math.isinf(conf):
            conf = 0.0
    except (TypeError, ValueError):
        conf = 0.0
    return intent, min(max(conf, 0.0), 1.0), False


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def _one_line(text: str) -> str:
    """trace 单行化:用户原文含换行会伪造日志行(log forging)。"""
    return " ".join(str(text).split())


def make_resolve_node(resolver):
    """resolver = with_structured_output(ResolvedQuestion) 产物;resume 旁路不走 LLM;异常兜底透传。

    ch07:{history} 槽位改喂分层历史的文本渲染(摘要行+滑窗),由路由层预算好放进 layered。
    """

    async def resolve_node(state, writer=None) -> dict:
        if state.get("resume_question"):
            q = state["resume_question"]
            # 旁路不发 resolved 帧:选择器轮已展示过同文灰字, resume 轮重发会重复渲染
            return {"resolved_message": q,
                    "trace": [*state["trace"], f"node=resolve resume_bypass q={_one_line(q)}"]}
        result = None
        try:
            result = await resolver.ainvoke(
                RESOLVE_PROMPT.format(
                    history=(state.get("layered") or {}).get("history_text") or "(无)",
                    query=state["user_message"]))
        except Exception:
            logger.warning("resolve 上游失败,原样透传", exc_info=True)
        if isinstance(result, dict):
            resolved = str(result.get("resolved") or "").strip()
            changed_raw = result.get("changed", False)
        else:
            resolved = str(getattr(result, "resolved", "") or "").strip()
            changed_raw = getattr(result, "changed", False)
        if not resolved:
            resolved = state["user_message"]
        changed = bool(changed_raw) and resolved != state["user_message"]
        if changed:
            _emit(writer, {"type": "resolved", "changed": True, "question": resolved})
        return {"resolved_message": resolved,
                "trace": [*state["trace"],
                          f"node=resolve changed={changed} q={_one_line(resolved)}"]}

    return resolve_node


def make_intent_node(model, escalator=None, floor: float = 0.6):
    """默认单次大模型;escalator 给定时小模型先判、confidence<floor 大模型重判一次取其结果。"""

    async def intent_node(state) -> dict:
        messages = [SystemMessage(content=INTENT_PROMPT),
                    HumanMessage(content=state["resolved_message"])]
        escalated = False
        upstream_error = False
        try:
            resp = await model.ainvoke(messages)
            intent, conf, malformed = parse_intent(getattr(resp, "content", ""))
        except Exception:
            logger.warning("intent 上游调用失败,降级为兜底意图", exc_info=True)
            intent, conf, malformed = INTENT_FALLBACK, 0.0, True
            upstream_error = True

        if escalator is not None and conf < floor and not upstream_error:
            try:
                resp2 = await escalator.ainvoke(messages)
                intent, conf, malformed = parse_intent(getattr(resp2, "content", ""))
                escalated = True
            except Exception:
                logger.warning("intent escalator 上游调用失败,保留初次判定", exc_info=True)

        line = f"node=intent intent={intent} confidence={conf:.2f}"
        if escalated:
            line += " escalated=true"
        if malformed:
            line += " malformed=true"
        if upstream_error:
            line += " upstream_error=true"
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
