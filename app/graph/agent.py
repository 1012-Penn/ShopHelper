"""主力 Agent 节点:手写 ReAct 循环——每步 bind_tools 流式 → tool_calls 执行回灌 → 收敛/熔断。

ch07 组装纪律:system 只装静态人设与红线(每轮逐字节一致,护住上游前缀缓存);早期梗概、
检索证据、订单数据与窄化指令合成一条背景块 HumanMessage 挂在当前用户消息之后——梗概绝不
单独占一条 system(上游模板会把所有 system 上提合并渲染,工具定义被挤到可变内容之后)。
工具结果入库前按 TOOL_RESULT_MAX_TOKENS 截断;每步调模型前把实际上下文原样打进 model_ctx。
"""
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.context import build_background, truncate_tool_result
from app.guard import is_refusal
from app.history import estimate_tokens
from app.prompts import SERVICE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

ACTIONS_ON_REFUSAL = ["transfer_human", "create_ticket"]

# ch06 退款/售后子流程窄化指令:只答「这一单」的问题,需求澄清仍在 Agent 内做
NARROW_INSTRUCTIONS = {
    "退款退货": ("用户处于退款退货流程。请仅依据下方订单数据与参考知识,"
                 "判断这一单(订单{order_id})能不能退,并说明依据条款;不要回答与这一单无关的问题。"),
    "售后": ("用户处于售后流程。请仅依据下方订单数据与参考知识,"
             "说明这一单(订单{order_id})的售后问题该怎么处理(修/换/退);不要回答与这一单无关的问题。"),
}

_ROLE_LOG = {"HumanMessage": "user", "AIMessage": "assistant", "ToolMessage": "tool",
             "SystemMessage": "system"}


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


def _log_model_ctx(state, step: int, messages: list, system: str) -> None:
    """每步调模型前,把实际发出的上下文原样打进 app.log(system 只计 token 不落正文)。"""
    window = messages[1:]
    lines = [f"{_ROLE_LOG.get(type(m).__name__, type(m).__name__)}: {m.content}" for m in window]
    tokens = estimate_tokens(system) + sum(estimate_tokens(m.content or "") for m in window)
    logger.info("[model_ctx] session=%s step=%d 条数=%d tokens≈%d\n摘要全文:%s\n滑窗逐条:\n%s",
                state["session_id"], step, len(window), tokens,
                state.get("layered", {}).get("summary_text") or "(无)",
                "\n".join(lines) or "(无)")


def _merge_tool_call_chunks(chunks: list[dict]) -> list[dict]:
    """流式 tool_call_chunks 按 index 拼装 → [{"name","args"(dict),"id"}](自 ch04 迁入)。"""
    merged: dict[int, dict] = {}
    for c in chunks:
        slot = merged.setdefault(c.get("index", 0), {"name": None, "args": "", "id": None})
        if c.get("name"):
            slot["name"] = c["name"]
        if c.get("args"):
            slot["args"] += c["args"]
        if c.get("id"):
            slot["id"] = c["id"]
    calls = []
    for slot in merged.values():
        try:
            args = json.loads(slot["args"] or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": slot["name"] or "", "args": args, "id": slot["id"] or ""})
    return calls


def _write_intercept(writer, state):
    """引擎拦截写操作时的回调:推工单预览卡片(无状态回传,前端确认后经 confirm 端点执行)。"""

    def _intercept(record, args):
        _emit(writer, {"type": "ticket_preview",
                       "ticket_type": args.get("ticket_type", "售后"),
                       "description": args.get("description", ""),
                       "conversation_id": state["session_id"]})

    return _intercept


def make_agent_node(model, engine, settings, pool):
    async def agent_node(state, writer=None) -> dict:
        lay = state.get("layered") or {}
        registry = engine.registry
        # 纯静态 system:证据/订单数据/窄化指令一律不进(ch07 组装纪律)
        system = SERVICE_PROMPT_TEMPLATE.format()
        instruction = ""
        if state.get("refund_flow") and state.get("order"):
            # 占位符先填充:I-1 回归修复——ch06 重写后 format 被丢,字面量 {order_id} 泄漏进背景块
            instruction = NARROW_INSTRUCTIONS.get(state.get("intent"), "").format(
                order_id=state["order"].get("order_id", ""))
        background = build_background(lay.get("summary_text") or "",
                                      state.get("evidence") or [],
                                      state.get("order") if instruction else None,
                                      instruction)
        messages = [SystemMessage(content=system),
                    *(lay.get("layer2_msgs") or []),
                    *(lay.get("layer1_msgs") or []),
                    HumanMessage(content=state["resolved_message"])]
        if background:
            messages.append(HumanMessage(content=background))
        # citations 继承既有证据再累计:前端 citations 帧是覆盖语义,重发必须带全量,
        # 否则子流程/知识路径先发的证据会被 agent 轮内 query_faq 的重发冲掉
        spent, steps, n_offset = 0, 0, len(state.get("evidence") or [])
        citations: list[dict] = [dict(it) for it in (state.get("evidence") or [])]
        emitted: list = []  # 本轮新增消息(AI/Tool),经 add_messages 并入完整历史
        while steps < settings.max_agent_steps and spent < settings.agent_token_budget:
            steps += 1
            _log_model_ctx(state, steps, messages, system)
            bound = model.bind_tools(registry.bind_tools())
            parts: list[str] = []
            call_chunks: list[dict] = []
            async for chunk in bound.astream(messages):
                if chunk.text:
                    parts.append(chunk.text)
                    _emit(writer, {"type": "token", "content": chunk.text})
                call_chunks.extend(getattr(chunk, "tool_call_chunks", None) or [])
            text = "".join(parts)
            spent += estimate_tokens(text)
            tool_calls = _merge_tool_call_chunks(call_chunks)
            if not tool_calls:
                emitted.append(AIMessage(content=text))
                actions = ACTIONS_ON_REFUSAL if is_refusal(text) else (
                    ["refund_form"] if state.get("refund_flow")
                    and state.get("intent") == "退款退货" else [])
                return {"final_reply": text, "agent_steps": steps, "messages": emitted,
                        "suggested_actions": actions,
                        "trace": [*state["trace"], f"node=agent steps={steps} converged"]}
            emitted.append(AIMessage(content=text, tool_calls=tool_calls))
            messages.append(AIMessage(content=text, tool_calls=tool_calls))
            for tc in tool_calls:
                _emit(writer, {"type": "tool_status", "name": tc["name"],
                               "label": registry.labels().get(tc["name"], tc["name"])})
                result = await engine.execute(
                    tc["name"], json.dumps(tc["args"], ensure_ascii=False),
                    conversation_id=state["session_id"], tool_call_id=tc["id"],
                    on_write_intercept=_write_intercept(writer, state))
                result = truncate_tool_result(result, settings.tool_result_max_tokens)
                parsed = _safe_json(result)
                if isinstance(parsed, dict):
                    if parsed.get("low_confidence"):
                        pool.insert("retrieval_low_conf", state["session_id"],
                                    state["user_message"], str(parsed.get("reason") or ""))
                    tool_items = [it for it in parsed.get("items") or []
                                  if isinstance(it, dict) and "n" in it]
                    if tool_items:
                        for it in tool_items:
                            it["n"] = n_offset + it["n"]
                            citations.append(it)
                        n_offset += len(tool_items)
                        result = json.dumps(parsed, ensure_ascii=False)
                        _emit(writer, {"type": "citations", "items": list(citations)})
                emitted.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        # 熔断:回溯最近一条 assistant 文本作收敛答复(工具 JSON 不是给人看的答案);全程无文本则给引导语。
        # cutoff 若来自 emitted 中既有文本则不重复追加(落库/历史只留一份)
        cutoff = next((m.content for m in reversed(emitted)
                       if isinstance(m, AIMessage) and m.content), None)
        if cutoff is None:
            cutoff = "问题比较复杂,请您稍后再试或换种问法。"
            emitted.append(AIMessage(content=cutoff))
            _emit(writer, {"type": "token", "content": cutoff})
        return {"final_reply": cutoff, "agent_steps": steps, "messages": emitted,
                "suggested_actions": [],
                "trace": [*state["trace"], f"node=agent steps={steps} cutoff"]}

    return agent_node


def _safe_json(raw):
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None
