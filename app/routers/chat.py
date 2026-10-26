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
from app.execution.store import (RequestConflict, ResumeConflict, SessionBusy,
                                 UnsafeResume)
from app.execution.context import current_lease_guard, current_run_id, current_run_owner
from app.graph.builder import initial_state
from app.history import estimate_tokens
from app.schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()

_QUEUE_DONE = object()  # 队列终结哨兵:图任务结束(含异常)后补投,排空循环据此收尾
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


def sse_frame(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def persisted_sse_frame(event_id: int, event: dict) -> str:
    """可重放 SSE 帧:id 用于 Last-Event-ID,data 与旧客户端 JSON 契约兼容。"""
    return f"id: {event_id}\n{sse_frame(event)}"


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


async def _persisted_event_stream(run_id: str, after_id: int, service):
    """只读持久事件。断连时本生成器停止,图任务与服务持有引用不随之取消。"""
    cursor = after_id
    while True:
        events = await service.events_after(run_id, cursor)
        for item in events:
            cursor = item.id
            yield persisted_sse_frame(item.id, item.event)
        status = await service.status(run_id)
        if status["status"] in _TERMINAL_STATUSES:
            # terminal 状态提交后可能还有最后一个 done/error 事件要读一次。
            final_events = await service.events_after(run_id, cursor)
            for item in final_events:
                cursor = item.id
                yield persisted_sse_frame(item.id, item.event)
            return
        if status["status"] in {"queued", "running", "cancelling"}:
            recovered = await asyncio.to_thread(
                service.store.recover_expired_lease, status["conversation_id"])
            if recovered:
                status = await service.status(run_id)
                if status["status"] in _TERMINAL_STATUSES:
                    continue
        await asyncio.sleep(service.policy.poll_seconds)


async def _execute_chat_run(request: Request, run_id: str, session_id: int,
                            body: ChatRequest) -> None:
    app = request.app
    api = app.state
    execution = api.execution_service
    started_at = time.perf_counter()
    trace = None
    queue: asyncio.Queue = asyncio.Queue()
    lease_lost = False

    async def renew_lease() -> None:
        nonlocal lease_lost
        interval = max(execution.policy.lease_seconds / 3, 0.1)
        while True:
            await asyncio.sleep(interval)
            renewed = await asyncio.to_thread(
                api.execution_store.renew_lease,
                run_id,
                execution.owner_token,
                lease_seconds=execution.policy.lease_seconds,
            )
            if not renewed:
                lease_lost = True
                logger.error("chat run=%s lost session lease; refusing further execution", run_id)
                return

    async def persist_frames() -> None:
        while True:
            frame = await queue.get()
            if frame is _QUEUE_DONE:
                return
            await execution.persist_event(run_id, frame)

    pump = asyncio.create_task(persist_frames())
    heartbeat = asyncio.create_task(renew_lease())
    try:
        await execution.mark_running(run_id)
        await api.mcp_service.sync()
        layered, rows, _anchors = await prepare_turn(request, session_id)
        hydrated = session_id in api.hydrated_threads
        if not hydrated:
            api.hydrated_threads.add(session_id)

        trace = api.observability.start_request(session_id, body.message)
        config = {
            "configurable": {"thread_id": str(session_id), "sink": queue},
            "metadata": trace.metadata(),
            "callbacks": [trace.usage_callback],
        }
        token = current_run_id.set(run_id)
        owner_token = current_run_owner.set(execution.owner_token)
        guard_token = current_lease_guard.set(
            lambda: api.execution_store.lease_owned_by(
                run_id, execution.owner_token))
        try:
            with trace.activate():
                result = await asyncio.wait_for(
                    api.graph.ainvoke(
                        initial_state(session_id, body.message, resume=body.resume,
                                      history_rows=None if hydrated else rows, layered=layered),
                        config,
                    ),
                    timeout=execution.policy.max_run_seconds,
                )
        finally:
            current_lease_guard.reset(guard_token)
            current_run_owner.reset(owner_token)
            current_run_id.reset(token)
        api.round_capture.record(
            session_id, body.message, result.get("retrieved_chunks") or [])
        if lease_lost:
            raise RuntimeError("session lease lost during execution")
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        trace.finish(
            intent=result.get("intent", ""), answer=result.get("final_reply", ""),
            duration_ms=duration_ms,
        )
        await queue.put(_QUEUE_DONE)
        await pump
        await execution.persist_event(run_id, {"type": "done", "run_id": run_id})
        await execution.finish(
            run_id, status="completed",
            result={
                "answer": result.get("final_reply", ""),
                "intent": result.get("intent", ""),
                "session_id": session_id,
            },
            usage=trace.usage_callback.snapshot(),
            duration_ms=duration_ms,
        )
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
    except asyncio.CancelledError:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        await queue.put(_QUEUE_DONE)
        await asyncio.gather(pump, return_exceptions=True)
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        if trace is not None:
            trace.fail(error="run cancelled", duration_ms=duration_ms)
        await execution.persist_event(run_id, {
            "type": "error", "message": "run cancelled", "reason": "cancelled"})
        try:
            current = await execution.status(run_id)
            terminal_status = ("interrupted"
                               if current["write_state"] != "none" else "cancelled")
        except Exception:
            terminal_status = "interrupted"
        await execution.finish(
            run_id, status=terminal_status, error="run cancelled",
            usage=trace.usage_callback.snapshot() if trace is not None else {},
            duration_ms=duration_ms,
        )
        raise
    except Exception as exc:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        await queue.put(_QUEUE_DONE)
        await asyncio.gather(pump, return_exceptions=True)
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        if trace is not None:
            trace.fail(error=str(exc), duration_ms=duration_ms)
        await execution.persist_event(run_id, {"type": "error", "message": str(exc)})
        try:
            current = await execution.status(run_id)
            terminal_status = ("interrupted"
                               if current["write_state"] != "none" else "failed")
        except Exception:
            terminal_status = "interrupted"
        await execution.finish(
            run_id, status=terminal_status, error=str(exc),
            usage=trace.usage_callback.snapshot() if trace is not None else {},
            duration_ms=duration_ms,
        )
        logger.warning("chat run=%s failed: %s", run_id, exc, exc_info=True)


@router.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    settings = request.app.state.settings
    store = request.app.state.store
    execution_store = request.app.state.execution_store
    execution = request.app.state.execution_service

    if estimate_tokens(body.message) > settings.max_user_input_tokens:
        raise HTTPException(status_code=400, detail="消息过长,超出单条输入预算")

    session_id = await store.resolve(body.session_id)
    payload = {
        "message": body.message,
        "session_id": session_id,
        "resume": body.resume.model_dump(mode="json") if body.resume else None,
    }
    try:
        reserved = await asyncio.to_thread(
            execution_store.reserve_run,
            conversation_id=session_id,
            client_request_id=body.client_request_id,
            request_payload=payload,
            owner_token=execution.owner_token,
            lease_seconds=execution.policy.lease_seconds,
        )
    except RequestConflict as exc:
        raise HTTPException(status_code=409, detail="client_request_id 对应不同请求") from exc
    except SessionBusy as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": "会话已有执行中的 run", "run_id": exc.run_id},
        ) from exc

    if reserved.created:
        await execution.persist_event(reserved.run_id, {
            "type": "session",
            "session_id": session_id,
            "run_id": reserved.run_id,
            "client_request_id": body.client_request_id,
        })
        task = asyncio.create_task(
            _execute_chat_run(request, reserved.run_id, session_id, body))
        execution.track(reserved.run_id, task)

    return StreamingResponse(
        _persisted_event_stream(reserved.run_id, 0, execution),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Run-ID": reserved.run_id,
        },
    )


@router.get("/api/chat/runs/{run_id}")
async def get_chat_run(run_id: str, request: Request):
    try:
        execution = request.app.state.execution_service
        status = await execution.status(run_id)
        if status["status"] in {"queued", "running", "cancelling"}:
            recovered = await asyncio.to_thread(
                request.app.state.execution_store.recover_expired_lease,
                status["conversation_id"],
            )
            if recovered:
                status = await execution.status(run_id)
        return status
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run 不存在") from exc


@router.get("/api/chat/runs/{run_id}/events")
async def replay_chat_events(run_id: str, request: Request) -> StreamingResponse:
    raw_cursor = (request.headers.get("Last-Event-ID")
                  or request.query_params.get("cursor") or "0")
    try:
        cursor = int(raw_cursor)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="cursor 必须是整数") from exc
    if cursor < 0:
        raise HTTPException(status_code=400, detail="cursor 不能为负数")
    try:
        await request.app.state.execution_service.status(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run 不存在") from exc
    return StreamingResponse(
        _persisted_event_stream(run_id, cursor, request.app.state.execution_service),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/chat/runs/{run_id}/resume")
async def resume_chat_run(run_id: str, request: Request) -> StreamingResponse:
    execution = request.app.state.execution_service
    try:
        status = await execution.status(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run 不存在") from exc
    if status["status"] in {"queued", "running", "cancelling"}:
        await asyncio.to_thread(
            request.app.state.execution_store.recover_expired_lease,
            status["conversation_id"],
        )
        status = await execution.status(run_id)
    if status["status"] == "completed":
        return StreamingResponse(
            _persisted_event_stream(run_id, 0, execution),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    if status["status"] != "interrupted":
        raise HTTPException(status_code=409, detail=f"run 状态不可恢复: {status['status']}")
    if status["write_state"] != "none":
        raise HTTPException(
            status_code=409,
            detail="写工具结果未知，禁止自动恢复；请先按确认幂等键核对业务结果",
        )
    try:
        reserved = await asyncio.to_thread(
            request.app.state.execution_store.resume_run,
            run_id,
            owner_token=execution.owner_token,
            lease_seconds=execution.policy.lease_seconds,
        )
    except UnsafeResume as exc:
        raise HTTPException(status_code=409, detail="写工具结果未知，禁止自动恢复") from exc
    except ResumeConflict as exc:
        raise HTTPException(status_code=409, detail=f"run 状态不可恢复: {exc}") from exc
    except SessionBusy as exc:
        raise HTTPException(status_code=409, detail={
            "message": "会话已有执行中的 run", "run_id": exc.run_id}) from exc

    body = ChatRequest.model_validate(status["request"])
    task = asyncio.create_task(
        _execute_chat_run(request, reserved.run_id, status["conversation_id"], body))
    execution.track(reserved.run_id, task)
    return StreamingResponse(
        _persisted_event_stream(run_id, 0, execution),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/chat/runs/{run_id}/cancel", status_code=202)
async def cancel_chat_run(run_id: str, request: Request):
    execution = request.app.state.execution_service
    try:
        status = await execution.status(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run 不存在") from exc
    if status["write_state"] in {"started", "submitted", "unknown"}:
        raise HTTPException(
            status_code=409,
            detail="写工具已开始或结果未知，不能宣称已取消；请核对幂等结果",
        )
    if status["status"] in _TERMINAL_STATUSES:
        return {"cancel_requested": False, "status": status["status"], "run_id": run_id}
    if not execution.cancel(run_id):
        raise HTTPException(
            status_code=409,
            detail="run 不在本进程执行器中；跨进程取消仅支持租约过期恢复",
        )
    return {"cancel_requested": True, "status": "cancelling", "run_id": run_id}
