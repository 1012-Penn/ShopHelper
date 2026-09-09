"""按意图查看请求 token 花销。"""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/usage/by-intent")
async def usage_by_intent(request: Request) -> dict:
    return {"items": await request.app.state.usage_store.aggregate_by_intent()}
