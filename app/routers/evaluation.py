"""评估轮次趋势只读接口。"""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/eval-runs")
async def eval_runs(request: Request) -> dict:
    return {"items": request.app.state.eval_store.list_runs()}
