"""ch10 · 主题分布旁路 API:分布查询 + 手动触发一批归类;实时聊天主链路不经过这里。"""
import asyncio
from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Request

from app.admin_auth import (AdminPrincipal, audit_context, require_csrf, require_operator,
                            require_viewer)
from app.admin_store import AdminAuditStore

router = APIRouter(prefix="/api/topics", tags=["topics"])


class ClassifyBatchRequest(BaseModel):
    limit: Optional[int] = None


@router.get("/distribution")
async def distribution(request: Request,
                       _principal: AdminPrincipal = Depends(require_viewer)) -> dict:
    return request.app.state.topic_store.distribution()


@router.post("/classify-batch")
async def classify_batch(request: Request, payload: ClassifyBatchRequest | None = None,
                         principal: AdminPrincipal = Depends(require_operator),
                         _: AdminPrincipal = Depends(require_csrf)) -> dict:
    service = request.app.state.topic_classifier
    if not service.available():
        raise HTTPException(status_code=503,
                            detail="主题分类模型未就绪:先跑 train/export 两个脚本")
    limit = payload.limit if payload and payload.limit else \
        request.app.state.settings.topic_classify_batch_limit
    # ONNX CPU 推理是重活,扔线程池防堵事件循环(ch09 落池同款惯例)
    classified = await asyncio.to_thread(service.classify_batch,
                                         request.app.state.topic_store, limit)
    audit: AdminAuditStore = request.app.state.admin_audit_store
    await asyncio.to_thread(
        audit.log_best_effort,
        admin_user_id=principal.user_id, username=principal.username,
        role=principal.role, action="topics.classify_batch",
        target_type="topic_pool", detail={"classified": classified, "limit": limit},
        **audit_context(request),
    )
    return {"classified": classified, "limit": limit}
