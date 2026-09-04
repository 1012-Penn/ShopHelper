import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.history import estimate_tokens, trim_history
from app.prompts import SERVICE_PROMPT_TEMPLATE
from app.schemas import ChatRequest

router = APIRouter()


def sse_frame(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _to_prompt_messages(trimmed: list[dict], new_user_content: str) -> list:
    """裁剪后的完整消息列表(System 在首位)→ LangChain 消息对象;新 user 消息追加在尾部。"""
    messages: list = []
    for m in trimmed:
        if m["role"] == "system":
            messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=new_user_content))
    return messages


@router.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    settings = request.app.state.settings
    store = request.app.state.sessions
    model = request.app.state.chat_model

    if estimate_tokens(body.message) > settings.history_token_budget:
        raise HTTPException(status_code=400, detail="消息过长,超出会话历史预算")

    session_id = await store.resolve(body.session_id)
    session = await store.get(session_id)
    # System 拼在 [0] 一起交给 trim_history,满足其"messages[0] 永不裁剪"的契约,
    # 且 System 不计入预算(spec §6)
    full_history = [
        {"role": "system", "content": SERVICE_PROMPT_TEMPLATE.format()},
        *session.messages,
    ]
    prompt_messages = _to_prompt_messages(
        trim_history(full_history, settings.history_token_budget),
        body.message,
    )

    async def event_stream():
        yield sse_frame({"type": "session", "session_id": session_id})
        parts: list[str] = []
        try:
            async for chunk in model.astream(prompt_messages):
                text = chunk.text
                if not text:
                    continue
                parts.append(text)
                yield sse_frame({"type": "token", "content": text})
            # 全部成功结束才落库;中途出错/断开不落库(spec §4.1)
            await store.append(session_id, body.message, "".join(parts))
            yield sse_frame({"type": "done"})
        except Exception as exc:  # 上游/内部错误:下发 error 后收流,本轮不落库
            yield sse_frame({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
