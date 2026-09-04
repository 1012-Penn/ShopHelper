# app/routers/sessions.py
from fastapi import APIRouter, HTTPException, Request

from app.schemas import HistoryResponse, MessageItem

router = APIRouter()


@router.get("/api/sessions/{session_id}/history")
async def history(session_id: int, request: Request) -> HistoryResponse:
    store = request.app.state.store
    if not await store.exists(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return HistoryResponse(
        session_id=session_id,
        messages=[MessageItem.model_validate(m) for m in await store.get_history(session_id)],
    )
