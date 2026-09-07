"""建工单:仅前端「建工单」按钮触发,后端不自动执行(spec §8)。"""
import json

from fastapi import APIRouter, HTTPException, Request

from app.schemas import TicketRequest

router = APIRouter()


@router.post("/api/tickets")
async def create_ticket(body: TicketRequest, request: Request):
    store, registry = request.app.state.store, request.app.state.registry
    if body.session_id is not None and not await store.exists(body.session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    conversation_id = await store.resolve(body.session_id)
    raw = await registry.execute("create_ticket", json.dumps({
        "conversation_id": conversation_id, "description": body.description,
        "ticket_type": body.ticket_type,
    }, ensure_ascii=False))
    parsed = json.loads(raw)
    if "ticket_no" not in parsed:
        raise HTTPException(status_code=502, detail="创建工单失败")
    return parsed
