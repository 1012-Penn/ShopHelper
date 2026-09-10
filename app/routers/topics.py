"""ch10 · 主题分布旁路 API:分布查询 + 手动触发一批归类;实时聊天主链路不经过这里。"""
import asyncio
from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/topics", tags=["topics"])


class ClassifyBatchRequest(BaseModel):
    limit: Optional[int] = None


@router.get("/distribution")
async def distribution(request: Request) -> dict:
    return request.app.state.topic_store.distribution()


@router.post("/classify-batch")
async def classify_batch(request: Request, payload: ClassifyBatchRequest | None = None) -> dict:
    service = request.app.state.topic_classifier
    if not service.available():
        raise HTTPException(status_code=503,
                            detail="主题分类模型未就绪:先跑 train/export 两个脚本")
    limit = payload.limit if payload and payload.limit else \
        request.app.state.settings.topic_classify_batch_limit
    # ONNX CPU 推理是重活,扔线程池防堵事件循环(ch09 落池同款惯例)
    classified = await asyncio.to_thread(service.classify_batch,
                                         request.app.state.topic_store, limit)
    return {"classified": classified, "limit": limit}
