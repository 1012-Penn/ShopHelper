"""聊天满意度反馈:👎进入后端问题池,👍只确认不入池。"""
from fastapi import APIRouter, Request

from app.schemas import FeedbackRequest

router = APIRouter()


@router.post("/api/feedback")
async def feedback(body: FeedbackRequest, request: Request) -> dict:
    if body.rating == "down":
        request.app.state.pool.insert(
            "user_feedback", body.session_id, body.question, "用户反馈未解决", [])
    return {"accepted": True}
