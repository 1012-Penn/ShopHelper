import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.history import estimate_tokens, trim_history
from app.prompts import SERVICE_PROMPT_TEMPLATE
from app.schemas import ChatRequest

router = APIRouter()


def sse_frame(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _merge_tool_call_chunks(chunks: list[dict]) -> list[dict]:
    """流式 tool_call_chunks 按 index 拼装 → [{"name","args"(dict),"id"}]。"""
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


def _to_prompt_messages(trimmed: list[dict], new_user_content: str) -> list:
    """裁剪后的完整消息列表(System 在首位)→ LangChain 消息对象;新 user 消息追加在尾部。

    完整还原工具轨迹:assistant(tool_calls) + tool(tool_call_id),跨轮回灌不失真。
    """
    messages: list = []
    for m in trimmed:
        if m["role"] == "system":
            messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            if m.get("tool_calls"):
                messages.append(AIMessage(content=m.get("content") or "", tool_calls=m["tool_calls"]))
            elif m.get("content"):
                messages.append(AIMessage(content=m["content"]))
        elif m["role"] == "tool":
            messages.append(ToolMessage(content=m.get("content") or "", tool_call_id=m.get("tool_call_id") or ""))
    messages.append(HumanMessage(content=new_user_content))
    return messages


@router.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    settings = request.app.state.settings
    store = request.app.state.store
    registry = request.app.state.registry
    model = request.app.state.chat_model

    if estimate_tokens(body.message) > settings.history_token_budget:
        raise HTTPException(status_code=400, detail="消息过长,超出会话历史预算")

    session_id = await store.resolve(body.session_id)
    history = await store.get_history(session_id)
    # System 拼在 [0] 一起交给 trim_history,满足其"messages[0] 永不裁剪"的契约,
    # 且 System 不计入预算(spec §6)
    full_history = [
        {"role": "system", "content": SERVICE_PROMPT_TEMPLATE.format()},
        *history,
    ]
    prompt_messages = _to_prompt_messages(
        trim_history(full_history, settings.history_token_budget),
        body.message,
    )

    async def event_stream():
        yield sse_frame({"type": "session", "session_id": session_id})
        # 整轮消息先攒在 pending,全部成功才落库;中途出错/断开不落库(spec §4.1)
        pending: list[dict] = [{"role": "user", "content": body.message}]
        try:
            # 第一段:绑工具流式调用。文本边收边吐,tool_call_chunks 静默拼装
            bound = model.bind_tools(registry.tools)
            parts: list[str] = []
            call_chunks: list[dict] = []
            async for chunk in bound.astream(prompt_messages):
                if chunk.text:
                    parts.append(chunk.text)
                    yield sse_frame({"type": "token", "content": chunk.text})
                call_chunks.extend(getattr(chunk, "tool_call_chunks", None) or [])
            tool_calls = _merge_tool_call_chunks(call_chunks)

            if tool_calls:
                pending.append({
                    "role": "assistant", "content": "".join(parts) or None, "tool_calls": tool_calls,
                })
                prompt_messages.append(AIMessage(content="".join(parts), tool_calls=tool_calls))
                for tc in tool_calls:
                    yield sse_frame({
                        "type": "tool_status",
                        "name": tc["name"],
                        "label": registry.labels().get(tc["name"], tc["name"]),
                    })
                    result = await registry.execute(tc["name"], json.dumps(tc["args"], ensure_ascii=False))
                    pending.append({"role": "tool", "content": result, "tool_call_id": tc["id"]})
                    prompt_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                # 第二段:不绑工具,强制单轮收敛;最终回答逐 token 流式吐出
                final_parts: list[str] = []
                async for chunk in model.astream(prompt_messages):
                    if chunk.text:
                        final_parts.append(chunk.text)
                        yield sse_frame({"type": "token", "content": chunk.text})
                pending.append({"role": "assistant", "content": "".join(final_parts)})
            else:
                pending.append({"role": "assistant", "content": "".join(parts)})

            await store.append(session_id, pending)
            yield sse_frame({"type": "done"})
        except Exception as exc:  # 上游/内部错误:下发 error 后收流,本轮不落库
            yield sse_frame({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
