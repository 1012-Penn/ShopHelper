"""订单固定目录(ch06):不建表、不接真实 API,选择器 /api/orders、query_order、子流程三处共用单一来源。"""
import re

ORDER_CATALOG: dict[str, dict] = {
    "1001": {"order_id": "1001", "product": "无线耳机", "amount": 299.0, "status": "已签收", "created_at": "2026-08-30"},
    "1002": {"order_id": "1002", "product": "机械键盘", "amount": 399.0, "status": "已发货", "created_at": "2026-09-03"},
    "1003": {"order_id": "1003", "product": "硅胶手机壳", "amount": 19.9, "status": "待发货", "created_at": "2026-09-07"},
}

# 「订单1001」「order 1002」带上下文优先;裸 4-6 位独立数字兜底(SH-E300 的 3 位数字不误伤)
_ORDER_ID_CONTEXTUAL = re.compile(r"(?:订单|order)\s*号?\s*(\d{3,6})", re.IGNORECASE)
_ORDER_ID_BARE = re.compile(r"(?<!\d)(\d{4,6})(?!\d)")


def get_order(order_id: str) -> dict:
    order = ORDER_CATALOG.get(str(order_id))
    if order is None:
        return {"order_id": str(order_id), "error": "未找到该订单"}
    return dict(order)


def list_orders() -> list[dict]:
    return [dict(o) for o in ORDER_CATALOG.values()]


def extract_order_id(text: str, require_known: bool = False) -> str | None:
    """require_known:裸数字兜底结果必须命中目录才返回(年份/尾号误抓防御);
    「订单N」上下文形态不受此限(未知单号走未找到话术是合法应答)。"""
    m = _ORDER_ID_CONTEXTUAL.search(text or "")
    if m:
        return m.group(1)
    m = _ORDER_ID_BARE.search(text or "")
    if m:
        oid = m.group(1)
        return oid if (not require_known or oid in ORDER_CATALOG) else None
    return None
