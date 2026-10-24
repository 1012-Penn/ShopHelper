"""飞轮待审队列后台 API。"""
import asyncio
import hashlib

from fastapi import APIRouter, Depends, HTTPException, Request

from app.admin_auth import AdminPrincipal, audit_context, require_csrf, require_operator, require_viewer
from app.admin_store import AdminAuditStore
from app.schemas import ReviewApproveRequest

router = APIRouter(prefix="/api/review-queue")


async def _audit(request: Request, principal: AdminPrincipal, action: str,
                 review_id: int, detail: dict) -> None:
    store: AdminAuditStore = request.app.state.admin_audit_store
    await asyncio.to_thread(
        store.log_best_effort,
        admin_user_id=principal.user_id, username=principal.username,
        role=principal.role, action=action, target_type="review_queue",
        target_id=review_id, detail=detail, **audit_context(request),
    )


@router.get("")
async def list_review_queue(request: Request, status: str = "待审",
                            _principal: AdminPrincipal = Depends(require_viewer)) -> dict:
    return {"items": await asyncio.to_thread(request.app.state.flywheel.list_reviews, status)}


@router.get("/{review_id}")
async def review_detail(review_id: int, request: Request,
                        _principal: AdminPrincipal = Depends(require_viewer)) -> dict:
    item = await asyncio.to_thread(request.app.state.flywheel.detail, review_id)
    if item is None:
        raise HTTPException(status_code=404, detail="待审问题不存在")
    return item


@router.post("/{review_id}/approve")
async def approve_review(review_id: int, body: ReviewApproveRequest, request: Request,
                         principal: AdminPrincipal = Depends(require_operator),
                         _: AdminPrincipal = Depends(require_csrf)) -> dict:
    # 审核通过内含向量化双写(嵌入 API + Milvus),放线程池避免阻塞事件循环
    try:
        chunk_id = await asyncio.to_thread(
            request.app.state.flywheel.approve, review_id, body.approved_answer)
    except KeyError:
        raise HTTPException(status_code=404, detail="待审问题不存在")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    await _audit(request, principal, "review.approve", review_id, {
        "review_id": review_id,
        "knowledge_chunk_id": chunk_id,
        "answer_chars": len(body.approved_answer),
        "answer_sha256": hashlib.sha256(body.approved_answer.encode("utf-8")).hexdigest(),
    })
    return {"approved": True, "knowledge_chunk_id": chunk_id}


@router.post("/{review_id}/reject")
async def reject_review(review_id: int, request: Request,
                        principal: AdminPrincipal = Depends(require_operator),
                        _: AdminPrincipal = Depends(require_csrf)) -> dict:
    try:
        await asyncio.to_thread(request.app.state.flywheel.reject, review_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="待审问题不存在")
    await _audit(request, principal, "review.reject", review_id, {"review_id": review_id})
    return {"rejected": True}
