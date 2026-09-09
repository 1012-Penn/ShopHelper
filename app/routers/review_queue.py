"""飞轮待审队列后台 API。"""
import asyncio

from fastapi import APIRouter, HTTPException, Request

from app.schemas import ReviewApproveRequest

router = APIRouter(prefix="/api/review-queue")


@router.get("")
async def list_review_queue(request: Request, status: str = "待审") -> dict:
    return {"items": await asyncio.to_thread(request.app.state.flywheel.list_reviews, status)}


@router.get("/{review_id}")
async def review_detail(review_id: int, request: Request) -> dict:
    item = await asyncio.to_thread(request.app.state.flywheel.detail, review_id)
    if item is None:
        raise HTTPException(status_code=404, detail="待审问题不存在")
    return item


@router.post("/{review_id}/approve")
async def approve_review(review_id: int, body: ReviewApproveRequest, request: Request) -> dict:
    # 审核通过内含向量化双写(嵌入 API + Milvus),放线程池避免阻塞事件循环
    try:
        chunk_id = await asyncio.to_thread(
            request.app.state.flywheel.approve, review_id, body.approved_answer)
    except KeyError:
        raise HTTPException(status_code=404, detail="待审问题不存在")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"approved": True, "knowledge_chunk_id": chunk_id}


@router.post("/{review_id}/reject")
async def reject_review(review_id: int, request: Request) -> dict:
    try:
        await asyncio.to_thread(request.app.state.flywheel.reject, review_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="待审问题不存在")
    return {"rejected": True}
