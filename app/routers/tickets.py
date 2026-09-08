"""建工单端点:按钮建单(点按钮即用户确认)+ ch08 建工单确认流(预览卡确认/取消)。"""
import json

from fastapi import APIRouter, HTTPException, Request

from app.schemas import TicketCancelRequest, TicketConfirmRequest, TicketRequest

router = APIRouter()


@router.post("/api/tickets")
async def create_ticket(body: TicketRequest, request: Request):
    """ch05 按钮建单:点按钮本身就是用户确认,引擎以 confirmed 放行(审计「成功」)。"""
    store, engine = request.app.state.store, request.app.state.engine
    if body.session_id is not None and not await store.exists(body.session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    conversation_id = await store.resolve(body.session_id)
    raw = await engine.execute("create_ticket", json.dumps({
        "conversation_id": conversation_id, "description": body.description,
        "ticket_type": body.ticket_type,
    }, ensure_ascii=False), conversation_id=conversation_id, confirmed=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None  # 引擎失败收敛为错误字符串,不是 JSON → 与下方同归 502
    if not isinstance(parsed, dict) or "ticket_no" not in parsed:
        raise HTTPException(status_code=502, detail="创建工单失败")
    return parsed


@router.post("/api/tickets/confirm")
async def confirm_ticket(body: TicketConfirmRequest, request: Request):
    """ch08 建工单确认流:前端预览卡「确认提交」回传,引擎以 confirmed 放行真实执行。"""
    store, engine = request.app.state.store, request.app.state.engine
    if not await store.exists(body.conversation_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    raw = await engine.execute("create_ticket", json.dumps({
        "conversation_id": body.conversation_id, "description": body.description,
        "ticket_type": body.ticket_type,
    }, ensure_ascii=False), conversation_id=body.conversation_id, confirmed=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict) or "ticket_no" not in parsed:
        raise HTTPException(status_code=502, detail="创建工单失败")
    return parsed


@router.post("/api/tickets/cancel")
async def cancel_ticket(body: TicketCancelRequest, request: Request):
    """ch08 建工单确认流:用户点「取消」→ 工单不建,本次调用按「权限拒绝」落审计。"""
    store, audit = request.app.state.store, request.app.state.tool_audit_store
    if not await store.exists(body.conversation_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    audit.log(tool_name="create_ticket", tool_source="builtin",
              conversation_id=body.conversation_id, status="权限拒绝",
              error_message=body.reason or "用户取消确认")
    return {"ok": True}
