"""聊天满意度反馈:👎进入后端问题池并回捞当轮召回片段,👍只确认不入池。"""
import asyncio

from fastapi import APIRouter, Request

from app.schemas import FeedbackRequest

router = APIRouter()


@router.post("/api/feedback")
async def feedback(body: FeedbackRequest, request: Request) -> dict:
    if body.rating == "down":
        # 尽力回捞该会话当轮的检索快照;确认没走检索(空快照)与回捞不到(None)都落 NULL
        snapshot = request.app.state.round_capture.lookup(body.session_id, body.question)
        await asyncio.to_thread(
            request.app.state.pool.insert,
            "user_feedback", body.session_id, body.question, "用户反馈未解决", snapshot)
    return {"accepted": True}
