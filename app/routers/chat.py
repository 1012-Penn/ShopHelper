"""ch05:/api/chat 内部替换为跑 LangGraph 图,SSE 帧契约对前端保持兼容 + 新增 actions 帧。"""
import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.graph.builder import initial_state
from app.history import estimate_tokens, trim_history
from app.prompts import SERVICE_PROMPT_TEMPLATE
from app.schemas import ChatRequest

router = APIRouter()

_QUEUE_DONE = object()  # 队列终结哨兵:图任务结束(含异常)后补投,排空循环据此收尾


def sse_frame(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    settings = request.app.state.settings
    store = request.app.state.store
    graph = request.app.state.graph

    if estimate_tokens(body.message) > settings.history_token_budget:
        raise HTTPException(status_code=400, detail="消息过长,超出会话历史预算")

    session_id = await store.resolve(body.session_id)
    history = await store.get_history(session_id)
    # System 拼在 [0] 交给 trim_history 满足"messages[0] 永不裁剪";System 不下发,由 Agent 节点自拼
    trimmed = trim_history([{"role": "system", "content": SERVICE_PROMPT_TEMPLATE.format()},
                            *history], settings.history_token_budget)

    async def event_stream():
        yield sse_frame({"type": "session", "session_id": session_id})
        queue: asyncio.Queue = asyncio.Queue()
        config = {"configurable": {"thread_id": str(session_id), "sink": queue}}

        async def run_graph():
            try:
                await graph.ainvoke(
                    initial_state(session_id, body.message, trimmed[1:]), config)
            finally:
                await queue.put(_QUEUE_DONE)  # 先冲刷已产帧再终结,异常也不丢序

        task = asyncio.create_task(run_graph())
        try:
            while True:
                frame = await queue.get()
                if frame is _QUEUE_DONE:
                    break
                yield sse_frame(frame)
            await task  # 传播图内异常
            yield sse_frame({"type": "done"})
        except Exception as exc:  # 图内节点异常:下发 error 后收流,本轮不落库(log 未达)
            yield sse_frame({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
