"""ch07:/api/chat 轮前上下文编排(级联降级→摘要触发→分层装配→history_ctx) + 跑图,SSE 帧契约不变。

分层视图以 MySQL 为构建源(messages 行带自增 id 与锚点对齐);checkpoint 的 State.messages
只承载图内完整轨迹,两套真源各走各的。摘要任务投后台,不阻塞本轮 SSE。
"""
import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.context import build_layered, cascade_layer1, layer2_tokens, split_layers
from app.graph.builder import initial_state
from app.history import estimate_tokens
from app.schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()

_QUEUE_DONE = object()  # 队列终结哨兵:图任务结束(含异常)后补投,排空循环据此收尾


def sse_frame(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _anchor_str(v: int | None) -> str:
    return "无" if v is None else str(v)


async def prepare_turn(request: Request, session_id: int) -> tuple[dict, list, dict]:
    """轮前维护与装配:层 1 超预算先同步降级,层 2 超预算投后台摘要,再组装模型面分层上下文。

    返回 (layered, 未回灌时的全量行, anchors);history_ctx 每轮必打(不进 Agent 的闲聊轮也看得到)。
    """
    settings = request.app.state.settings
    store = request.app.state.store
    budgets = request.app.state.budgets

    rows = await store.get_rows(session_id)
    anchors = await store.get_anchors(session_id)

    # 级联第一步:层 1 超预算 → 降级一批(纯锚点移动,同步生效)
    new_lf, t0, t1 = cascade_layer1(rows, anchors["layer1_from"], budgets.layer1)
    if new_lf != anchors["layer1_from"]:
        await store.set_layer1_from(session_id, new_lf)
        logger.info("[ctx] session=%s 层1 降级 %s→%s token %d→%d",
                    session_id, _anchor_str(anchors["layer1_from"]), _anchor_str(new_lf), t0, t1)
        anchors["layer1_from"] = new_lf
    l1_rows, l2_rows = split_layers(rows, anchors["summary_upto"], anchors["layer1_from"])

    # 级联第二步:层 2 渲染态超预算 → 后台摘要(触发看用量不数条数,工具结果不落消息表)
    l2_tok = layer2_tokens(l2_rows, settings.ctx_layer2_assistant_head_chars)
    if l2_tok > budgets.layer2:
        logger.info("[ctx] session=%s summary trigger 层2 约 %d token > 预算 %d"
                    "(后台执行,不阻塞本轮回复)", session_id, l2_tok, budgets.layer2)
        request.app.state.summary_service.maybe_trigger(session_id,
                                                        upto_id=anchors["layer1_from"])

    layered = build_layered(l1_rows, l2_rows, anchors["summary_text"],
                            settings.ctx_layer2_assistant_head_chars, budgets.sliding)
    logger.info("[history_ctx] session=%s 摘要=%d段 滑窗=%d条 层2=%d 层1=%d tokens≈%d\n%s",
                session_id, anchors["summary_seqs"], layered["n_window"],
                layered["tokens_l2"], layered["tokens_l1"],
                layered["tokens_l1"] + layered["tokens_l2"], layered["history_text"])
    return layered, rows, anchors


@router.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    settings = request.app.state.settings
    store = request.app.state.store
    graph = request.app.state.graph
    observability = request.app.state.observability

    if estimate_tokens(body.message) > settings.max_user_input_tokens:
        raise HTTPException(status_code=400, detail="消息过长,超出单条输入预算")

    # MCP 现问现拿:每轮聊天前 diff 同步一次(TTL 防抖),Server 侧新工具不重启即用
    await request.app.state.mcp_service.sync()

    session_id = await store.resolve(body.session_id)
    layered, rows, _anchors = await prepare_turn(request, session_id)

    # 完整历史回灌:checkpoint 为空(进程重启/首 touch)时从 MySQL 灌一次,add_messages 按 id 去重
    hydrated = session_id in request.app.state.hydrated_threads
    if not hydrated:
        request.app.state.hydrated_threads.add(session_id)

    async def event_stream():
        yield sse_frame({"type": "session", "session_id": session_id})
        queue: asyncio.Queue = asyncio.Queue()
        trace = observability.start_request(session_id, body.message)
        config = {
            "configurable": {"thread_id": str(session_id), "sink": queue},
            "metadata": trace.metadata(),
            "callbacks": [trace.usage_callback],
        }

        async def run_graph():
            started_at = time.perf_counter()
            try:
                with trace.activate():
                    result = await graph.ainvoke(
                        initial_state(session_id, body.message, resume=body.resume,
                                      history_rows=None if hydrated else rows, layered=layered),
                        config)
                    trace.finish(
                        intent=result.get("intent", ""), answer=result.get("final_reply", ""),
                        duration_ms=int((time.perf_counter() - started_at) * 1000),
                    )
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
        finally:
            # 客户端提前断开(GeneratorExit/CancelledError)时取消孤儿图任务:
            # 不再烧 token,也保证未达 log 节点的回合绝不落库
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
