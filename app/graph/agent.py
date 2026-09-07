"""主力 Agent 节点:手写 ReAct 循环——每步 bind_tools 流式 → tool_calls 执行回灌 → 收敛/熔断。

区别于 ch04 单轮两段式(一次工具后强制收敛):这里每步都带工具,复杂问题按中间结果多走几步。
query_faq 语义自 ch04 路由迁入:低置信落池、证据编号 n_offset 续排并重写回灌、citations 帧全量重发
(前端 byN 覆盖语义天然消解,ch04 已验证)。
"""
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.guard import is_refusal
from app.history import estimate_tokens, history_to_messages
from app.prompts import SERVICE_PROMPT_TEMPLATE, build_evidence_block

logger = logging.getLogger(__name__)

ACTIONS_ON_REFUSAL = ["transfer_human", "create_ticket"]


def _emit(writer, frame: dict) -> None:
    if writer is not None:
        writer(frame)


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


def make_agent_node(model, registry, settings, pool):
    async def agent_node(state, writer=None) -> dict:
        system = SERVICE_PROMPT_TEMPLATE.format()
        if state.get("evidence"):
            system += "\n\n参考知识(回答须带 [n] 角标):\n" + build_evidence_block(state["evidence"])
        messages = [SystemMessage(content=system),
                    *history_to_messages(state["history"]),
                    HumanMessage(content=state["resolved_message"])]
        pending = [{"role": "user", "content": state["user_message"]}]
        spent, steps, n_offset = 0, 0, 0
        citations: list[dict] = []
        while steps < settings.agent_max_steps and spent < settings.agent_token_budget:
            steps += 1
            bound = model.bind_tools(registry.tools)
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
                pending.append({"role": "assistant", "content": text})
                actions = ACTIONS_ON_REFUSAL if is_refusal(text) else []
                return {"final_reply": text, "agent_steps": steps, "messages": pending,
                        "suggested_actions": actions,
                        "trace": [*state["trace"], f"node=agent steps={steps} converged"]}
            pending.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
            messages.append(AIMessage(content=text, tool_calls=tool_calls))
            for tc in tool_calls:
                _emit(writer, {"type": "tool_status", "name": tc["name"],
                               "label": registry.labels().get(tc["name"], tc["name"])})
                result = await registry.execute(tc["name"], json.dumps(tc["args"], ensure_ascii=False))
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
                pending.append({"role": "tool", "content": result, "tool_call_id": tc["id"]})
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        cutoff = pending[-1].get("content") or "问题比较复杂,请您稍后再试或换种问法。"
        return {"final_reply": cutoff, "agent_steps": steps, "messages": pending,
                "suggested_actions": [],
                "trace": [*state["trace"], f"node=agent steps={steps} cutoff"]}

    return agent_node


def _safe_json(raw):
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None
