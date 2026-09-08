"""ch06:订单列表(固定 mock 目录),订单选择器的数据源。"""
from fastapi import APIRouter

from app.orders import list_orders

router = APIRouter()


@router.get("/api/orders")
async def orders() -> dict:
    return {"items": list_orders()}
