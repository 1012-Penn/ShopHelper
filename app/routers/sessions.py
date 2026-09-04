# app/routers/sessions.py
from fastapi import APIRouter, HTTPException, Request

from app.schemas import HistoryResponse, MessageItem

router = APIRouter()


@router.get("/api/sessions/{session_id}/history")
async def history(session_id: str, request: Request) -> HistoryResponse:
    data = await request.app.state.sessions.get(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return HistoryResponse(
        session_id=session_id,
        messages=[MessageItem.model_validate(m) for m in data.messages],
    )
