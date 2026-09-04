# app/routers/extract.py
from fastapi import APIRouter, HTTPException, Request

from app.schemas import AfterSaleExtraction, ExtractRequest

router = APIRouter()


@router.post("/api/extract")
async def extract(body: ExtractRequest, request: Request) -> AfterSaleExtraction:
    try:
        result = await request.app.state.extract_model.ainvoke(body.text)
    except Exception as exc:  # 解析失败/上游异常统一 422(spec §4.3)
        raise HTTPException(status_code=422, detail=f"结构化抽取失败:{exc}") from exc
    return result
