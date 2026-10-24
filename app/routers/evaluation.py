"""评估轮次趋势只读接口。"""
from fastapi import APIRouter, Depends, Request

from app.admin_auth import AdminPrincipal, require_viewer

router = APIRouter()


@router.get("/api/eval-runs")
async def eval_runs(request: Request,
                    _principal: AdminPrincipal = Depends(require_viewer)) -> dict:
    return {"items": request.app.state.eval_store.list_runs()}
